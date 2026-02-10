import csv
import gzip
import logging
from glob import glob
from copy import deepcopy
import random
import os.path
import argparse
import re
from typing import List

import torch
from torch.nn import Module, Embedding, LSTM, Linear
from torch.nn.utils.rnn import pad_sequence, unpad_sequence
from torch.nn.utils import clip_grad_norm_
from torch.nn.functional import cross_entropy
from torch.optim import AdamW
from torch.utils.data import DataLoader

logger = logging.getLogger("train_model")


def vowel_clusters(word: str) -> List[str]:
    """
    Identify vowel clusters in a word with shared consonants between them.
    Returns a list of syllables.
    """
    clusters = re.findall(r"(^|[^aeiou]+)([aeiou]+)([^aeiou]+$)?", word.lower())
    if not clusters:
        return []
    
    processed = []
    prev_end_consonant = ""
    
    for i, (start, vowels, end) in enumerate(clusters):
        if i == 0:
            prev_consonant = start
        else:
            prev_consonant = prev_end_consonant
        
        if i == len(clusters) - 1:
            next_consonant = end if end is not None else ""
        else:
            if clusters[i+1][0]:
                next_consonant = clusters[i+1][0]
                prev_end_consonant = clusters[i+1][0]
            else:
                next_consonant = ""
                prev_end_consonant = ""
        
        processed.append(prev_consonant + vowels + next_consonant)
    
    return processed


def collate(items):
    syllables = pad_sequence([x for x, _, _, _ in items], batch_first=True)
    stresses = pad_sequence([x for _, x, _, _ in items], batch_first=True)
    lengths = torch.tensor([x for _, _, x, _ in items])
    fnames = [x for _, _, _, x in items]
    return (syllables, stresses, lengths, fnames)


class Model(Module):
    def __init__(self, embedding_size, syllable_vocab, hidden_size, stress_vocab, bidirectional):
        super(Model, self).__init__()
        self.syllable_vocab = syllable_vocab
        self.stress_vocab = stress_vocab
        self.embedding = Embedding(len(self.syllable_vocab), embedding_size)
        self.rnn = LSTM(embedding_size, hidden_size, num_layers=1, batch_first=True,
                       bidirectional=bidirectional, dropout=0.0)
        self.head = Linear(hidden_size * (2 if bidirectional else 1), len(self.stress_vocab))

    def forward(self, syllables):
        out, _ = self.rnn(self.embedding(syllables))
        return self.head(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="full_dataset")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--train_proportion", type=float, default=0.8)
    parser.add_argument("--dev_proportion", type=float, default=0.1)
    parser.add_argument("--test_proportion", type=float, default=0.1)
    parser.add_argument("--embedding_size", type=int, default=64)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--learning_rate", type=float, default=0.005)
    parser.add_argument("--use_dev_loss", action="store_true", default=False)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--unidirectional", action="store_true", default=False)
    parser.add_argument("--alternation_weight", type=float, default=0.1, 
                       help="Weight for alternating stress pattern reward (default: 0.1)")
    parser.add_argument("--device")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.seed != None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    torch.set_default_device(device)

    logger.info("Using device = %s", torch.get_default_device())

    stress_vocab = {}
    syllable_vocab = {}
    data = []
    raw_data = []  # Store raw data for saving split info

    for fname in glob(os.path.join(args.data_path, "*csv")):
        if os.path.basename(fname) == "combined.csv":
            continue
        with open(fname, "rt") as ifd:
            for row in csv.DictReader(ifd):
                o_text = row["OXFORD_TEXT"]
                o_words = o_text.split()
                o_stresses = row["OXFORD_SCANSION"].split()
                
                # Convert words to syllables
                all_syllables = []
                all_syllable_stresses = []
                
                # Check if we have matching words and stress patterns
                if len(o_words) != len(o_stresses):
                    continue
                
                for word, stress_pattern in zip(o_words, o_stresses):
                    syllables = vowel_clusters(word)
                    
                    # If no syllables found or stress pattern length doesn't match, skip
                    if not syllables or len(syllables) != len(stress_pattern):
                        break
                    
                    # Map each stress character to its corresponding syllable
                    for syllable, stress_char in zip(syllables, stress_pattern):
                        all_syllables.append(syllable)
                        all_syllable_stresses.append(stress_char)
                else:
                    # Only add if we didn't break (all words processed successfully)
                    if all_syllables:
                        data.append(
                            (
                                torch.tensor([syllable_vocab.setdefault(s, len(syllable_vocab)) 
                                            for s in all_syllables]),
                                torch.tensor([stress_vocab.setdefault(s, len(stress_vocab)) 
                                            for s in all_syllable_stresses]),
                                len(all_syllables),
                                fname
                            )
                        )
                        # Store raw data info
                        raw_data.append({
                            'text': o_text,
                            'scansion': row["OXFORD_SCANSION"],
                            'filename': fname
                        })

    # Shuffle both data lists together
    combined = list(zip(data, raw_data))
    random.shuffle(combined)
    data, raw_data = zip(*combined)
    data = list(data)
    raw_data = list(raw_data)

    train_count = int(args.train_proportion * len(data))
    dev_count = int(args.dev_proportion * len(data))
    test_count = int(args.dev_proportion * len(data))

    train = data[:train_count]
    dev = data[train_count:train_count + dev_count]
    test = data[train_count + dev_count:]

    logger.info(
        "Split data into %d/%d/%d train/dev/test lines, syllable vocabulary size=%d, stress pattern vocabulary size=%d",
        len(train), len(dev), len(test), len(syllable_vocab), len(stress_vocab)
    )

    train_dl = DataLoader(train, batch_size=args.batch_size, collate_fn=collate,
                         shuffle=True, generator=torch.Generator(device=device))
    dev_dl = DataLoader(dev, batch_size=args.batch_size, collate_fn=collate)
    test_dl = DataLoader(test, batch_size=args.batch_size, collate_fn=collate)

    model = Model(args.embedding_size, syllable_vocab, args.hidden_size, stress_vocab,
                 not args.unidirectional)
    opt = AdamW(model.parameters(), lr=args.learning_rate)

    best_dev_score = None
    best_state_dict = None

    for epoch in range(args.max_epochs):
        train_losses = []
        dev_losses = []
        dev_guesses = []
        dev_golds = []

        model.train()
        for syllables, stresses, lengths, fnames in train_dl:
            out = model(syllables)
            ces = cross_entropy(torch.transpose(out, 1, 2), stresses, reduction="none")
            uces = unpad_sequence(ces, lengths, batch_first=True)
            
            # Calculate alternation reward
            # Get predicted stress labels
            predictions = out.argmax(dim=2)
            reverse_stress_vocab = {v: k for k, v in stress_vocab.items()}
            
            alternation_penalty = 0.0
            for i, (pred_seq, length) in enumerate(zip(predictions, lengths)):
                pred_seq = pred_seq[:length]
                # Convert to stress labels
                stress_labels = [reverse_stress_vocab[idx.item()] for idx in pred_seq]
                
                # Filter out 'x' and check alternation
                filtered = [s for s in stress_labels if s != 'x']
                
                # Count violations of alternation (consecutive S or u)
                violations = 0
                for j in range(len(filtered) - 1):
                    if filtered[j] == filtered[j + 1]:
                        violations += 1
                
                # Add penalty proportional to violations
                alternation_penalty += violations
            
            # Combine cross-entropy loss with alternation penalty
            # The penalty encourages alternating patterns
            ce_loss = sum([x.sum() for x in uces])
            loss = ce_loss + args.alternation_weight * alternation_penalty  # Weight the penalty
            
            loss.backward()
            clip_grad_norm_(model.parameters(), 3)
            opt.step()
            opt.zero_grad()
            train_losses.append(loss.detach().item())

        train_loss = sum(train_losses) / (len(train_losses) * args.batch_size)

        model.eval()
        for syllables, stresses, lengths, fnames in dev_dl:
            out = model(syllables)
            dev_guesses.append(torch.cat(unpad_sequence(out.argmax(dim=2).detach(), lengths, batch_first=True)))
            dev_golds.append(torch.cat(unpad_sequence(stresses.detach(), lengths, batch_first=True)))
            kls = cross_entropy(torch.transpose(out, 1, 2), stresses, reduction="none")
            loss = sum([x.sum() for x in unpad_sequence(kls, lengths, batch_first=True)])
            dev_losses.append(loss.detach().item())

        dev_loss = sum(dev_losses) / (len(dev_losses) * args.batch_size)
        dev_guesses = torch.cat(dev_guesses)
        dev_golds = torch.cat(dev_golds)
        dev_acc = (dev_guesses == dev_golds).sum() / dev_guesses.shape[0]

        logger.info(
            "Epoch %d average instance loss (train/dev): %f/%f\n Dev accuracy: %f",
            epoch + 1, train_loss, dev_loss, dev_acc
        )

        dev_score = -dev_loss if args.use_dev_loss else dev_acc
        if not best_dev_score or dev_score > best_dev_score:
            logger.info("Saving new best model")
            best_dev_score = dev_score
            best_state_dict = deepcopy(model.state_dict())

    model.load_state_dict(best_state_dict)
    model.eval()

    test_losses = []
    test_guesses = []
    test_golds = []
    test_fnames = []

    for syllables, stresses, lengths, fnames in test_dl:
        out = model(syllables)
        test_fnames.append(sum([[f] * l for l, f in zip(lengths, fnames)], []))
        test_guesses.append(torch.cat(unpad_sequence(out.argmax(dim=2).detach(), lengths, batch_first=True)))
        test_golds.append(torch.cat(unpad_sequence(stresses.detach(), lengths, batch_first=True)))
        kls = cross_entropy(torch.transpose(out, 1, 2), stresses, reduction="none")
        loss = sum([x.sum() for x in torch.nn.utils.rnn.unpad_sequence(kls, lengths, batch_first=True)])
        test_losses.append(loss.detach().item())

    test_fnames = sum(test_fnames, [])
    test_guesses = torch.cat(test_guesses)
    test_golds = torch.cat(test_golds)
    test_acc = (test_guesses == test_golds).sum() / test_guesses.shape[0]

    logger.info("Final average instance loss (test): %f\n Test accuracy: %f",
               sum(test_losses) / (len(test_losses) * args.batch_size), test_acc)

    indices = {}
    for i, fname in enumerate(test_fnames):
        indices[fname] = indices.get(fname, [])
        indices[fname].append(i)

    fname_accs = {}
    for fname, idxs in indices.items():
        fname_accs[fname] = (test_guesses[idxs] == test_golds[idxs]).sum() / len(idxs)

    logger.info("Accuracy by section:")
    for fname, acc in reversed(sorted(fname_accs.items(), key=lambda x: x[1])):
        print("{:.3f} : {}".format(acc, os.path.basename(fname).replace("_gui_complete.csv", "")))

    # Save model and metadata including data split
    save_dict = {
        'model': model,
        'syllable_vocab': syllable_vocab,
        'stress_vocab': stress_vocab,
        'train_indices': list(range(0, train_count)),
        'dev_indices': list(range(train_count, train_count + dev_count)),
        'test_indices': list(range(train_count + dev_count, len(data))),
        'raw_data': raw_data,  # All data in shuffled order
        'args': {
            'seed': args.seed,
            'train_proportion': args.train_proportion,
            'dev_proportion': args.dev_proportion,
            'test_proportion': args.test_proportion,
            'alternation_weight': args.alternation_weight
        }
    }

    with open(args.output, "wb") as ofd:
        torch.save(save_dict, ofd)

    logger.info("Saved model and split information to %s", args.output)
