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
    clusters = re.findall(r"(^|[^aeou]+)([aeou]+)([^aeou]+$)?", word.lower())
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
    
    if len(vowels)>1 and not (len(vowels)==2 and vowels[1] == 'u') and not (len(vowels)==2 and vowels[0] == vowels[1]) and not (len(vowels) == 2 and prev_consonant[-1] == 'q' and vowels[0] == 'u'):
        print(vowels)
        print(word)
    return processed


def assign_char_stresses(word: str, stress_pattern: str) -> List[str]:
    """
    Assign stress markers to each character in a word based on syllables.
    
    Rules:
    - First vowel in syllable gets the syllable's stress (S, u, or x)
    - Characters that overlap between syllables get '|'
    - Other characters get '-'
    - Spaces get 'SPACE'
    """
    syllables = vowel_clusters(word)
    
    if not syllables or len(syllables) != len(stress_pattern):
        return None
    
    word_lower = word.lower()
    char_stresses = [None] * len(word_lower)  # Initialize with None
    
    # Track which character positions have been assigned
    char_pos = 0
    
    for syll_idx, (syllable, stress) in enumerate(zip(syllables, stress_pattern)):
        # Find first vowel in this syllable
        first_vowel_idx = None
        for i, c in enumerate(syllable):
            if c in 'aeiou':
                first_vowel_idx = i
                break
        
        # Determine overlap with next syllable
        overlap_len = 0
        if syll_idx < len(syllables) - 1:
            next_syll = syllables[syll_idx + 1]
            # Find longest overlap
            for i in range(1, min(len(syllable), len(next_syll)) + 1):
                if syllable[-i:] == next_syll[:i]:
                    overlap_len = i
        
        # Assign stress to characters in this syllable (excluding overlap)
        syll_len_no_overlap = len(syllable) - overlap_len
        
        for i in range(syll_len_no_overlap):
            if char_stresses[char_pos + i] is None:  # Only assign if not already set
                if i == first_vowel_idx:
                    char_stresses[char_pos + i] = stress
                else:
                    char_stresses[char_pos + i] = '-'
        
        # Mark overlap characters as '|'
        for i in range(overlap_len):
            overlap_pos = char_pos + syll_len_no_overlap + i
            if overlap_pos < len(char_stresses):
                char_stresses[overlap_pos] = '|'
        
        # Move to next position (only advance by non-overlapping part)
        char_pos += syll_len_no_overlap
    
    # Fill in any remaining None values with '-'
    char_stresses = [s if s is not None else '-' for s in char_stresses]
    
    return char_stresses


def collate(items):
    words = pad_sequence([x for x, _, _, _ in items], batch_first=True)
    stresses = pad_sequence([x for _, x, _, _ in items], batch_first=True)
    lengths = torch.tensor([x for _, _, x, _ in items])
    fnames = [x for _, _, _, x in items]
    return (words, stresses, lengths, fnames)


class Model(Module):
    def __init__(self, embedding_size, word_vocab, hidden_size, stress_vocab, bidirectional):
        super(Model, self).__init__()
        self.word_vocab = word_vocab
        self.stress_vocab = stress_vocab
        self.embedding = Embedding(len(self.word_vocab), embedding_size)
        self.rnn = LSTM(embedding_size, hidden_size, num_layers=1, batch_first=True,
                       bidirectional=bidirectional, dropout=0.0)
        self.head = Linear(hidden_size * (2 if bidirectional else 1), len(self.stress_vocab))

    def forward(self, words):
        out, _ = self.rnn(self.embedding(words))
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
    word_vocab = {}  # Character vocab
    data = []
    raw_data = []

    skipped = 0
    processed = 0

    for fname in glob(os.path.join(args.data_path, "*csv")):
        if os.path.basename(fname) == "combined.csv":
            continue
        with open(fname, "rt") as ifd:
            for row in csv.DictReader(ifd):
                processed += 1
                o_text = row["OXFORD_TEXT"]
                o_words = o_text.split()
                o_stresses = row["OXFORD_SCANSION"].split()
                
                if len(o_words) != len(o_stresses):
                    skipped += 1
                    #print('len mismatch')
                    continue
                
                # Build character-level representation with syllable-aware stress
                all_char_stresses = []
                valid = True
                
                for word, stress_pattern in zip(o_words, o_stresses):
                    char_stresses = assign_char_stresses(word, stress_pattern)
                    if char_stresses is None:
                        valid = False
                        break
                    all_char_stresses.extend(char_stresses)
                    # Add space marker between words (except after last word)
                    if word != o_words[-1]:
                        all_char_stresses.append('SPACE')
                
                if not valid:
                    skipped += 1
                    #print('invalid')
                    continue
                
                # Create character list (with spaces)
                chars = []
                for i, word in enumerate(o_words):
                    chars.extend(list(word.lower()))
                    if i < len(o_words) - 1:
                        chars.append(' ')
                
                if len(chars) != len(all_char_stresses):
                    skipped += 1
                    continue
                
                data.append(
                    (
                        torch.tensor([word_vocab.setdefault(c, len(word_vocab)) for c in chars]),
                        torch.tensor([stress_vocab.setdefault(s, len(stress_vocab)) for s in all_char_stresses]),
                        len(chars),
                        fname
                    )
                )
                raw_data.append({
                    'text': o_text,
                    'scansion': row["OXFORD_SCANSION"],
                    'filename': fname
                })

    logger.info(f"Processed {processed} lines, kept {len(data)}, skipped {skipped}")

    # Shuffle both data lists together
    combined = list(zip(data, raw_data))
    random.shuffle(combined)
    data, raw_data = zip(*combined)
    data = list(data)
    raw_data = list(raw_data)

    train_count = int(args.train_proportion * len(data))
    dev_count = int(args.dev_proportion * len(data))
    test_count = len(data) - train_count - dev_count  # Use remaining for test

    train = data[:train_count]
    dev = data[train_count:train_count + dev_count]
    test = data[train_count + dev_count:]

    logger.info(
        "Split data into %d/%d/%d train/dev/test lines, character vocabulary size=%d, stress pattern vocabulary size=%d",
        len(train), len(dev), len(test), len(word_vocab), len(stress_vocab)
    )
    
    if len(test) == 0:
        logger.warning("WARNING: Test set is empty! Consider adjusting proportions or getting more data.")

    train_dl = DataLoader(train, batch_size=args.batch_size, collate_fn=collate,
                         shuffle=True, generator=torch.Generator(device=device))
    dev_dl = DataLoader(dev, batch_size=args.batch_size, collate_fn=collate)
    test_dl = DataLoader(test, batch_size=args.batch_size, collate_fn=collate)

    model = Model(args.embedding_size, word_vocab, args.hidden_size, stress_vocab,
                 not args.unidirectional)
    opt = AdamW(model.parameters(), lr=args.learning_rate)

    best_dev_score = None
    best_state_dict = None

    for epoch in range(args.max_epochs):
        train_losses = []
        train_alternation_penalties = []
        dev_losses = []
        dev_guesses = []
        dev_golds = []

        model.train()
        for words, stresses, lengths, fnames in train_dl:
            out = model(words)
            ces = cross_entropy(torch.transpose(out, 1, 2), stresses, reduction="none")
            uces = unpad_sequence(ces, lengths, batch_first=True)
            
            # Calculate alternation reward (only on S/u/x, ignoring - and |)
            predictions = out.argmax(dim=2)
            reverse_stress_vocab = {v: k for k, v in stress_vocab.items()}
            
            alternation_penalty = 0.0
            for i, (pred_seq, length) in enumerate(zip(predictions, lengths)):
                pred_seq = pred_seq[:length]
                stress_labels = [reverse_stress_vocab[idx.item()] for idx in pred_seq]
                
                # Filter out '-', '|', 'SPACE' - keep only S, u, x
                filtered = [s for s in stress_labels if s in ['S', 'u', 'x']]
                
                # Further filter out 'x' for alternation check
                alternating = [s for s in filtered if s != 'x']
                
                # Count violations
                violations = 0
                for j in range(len(alternating) - 1):
                    if alternating[j] == alternating[j + 1]:
                        violations += 1
                
                alternation_penalty += violations
            
            ce_loss = sum([x.sum() for x in uces])
            loss = ce_loss + args.alternation_weight * alternation_penalty
            
            loss.backward()
            clip_grad_norm_(model.parameters(), 3)
            opt.step()
            opt.zero_grad()
            train_losses.append(loss.detach().item())
            train_alternation_penalties.append(alternation_penalty)

        train_loss = sum(train_losses) / (len(train_losses) * args.batch_size)
        avg_alternation_penalty = sum(train_alternation_penalties) / len(train_alternation_penalties)

        model.eval()
        for words, stresses, lengths, fnames in dev_dl:
            out = model(words)
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
            "Epoch %d average instance loss (train/dev): %f/%f\n Dev accuracy: %f\n Avg alternation violations per batch: %f",
            epoch + 1, train_loss, dev_loss, dev_acc, avg_alternation_penalty
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

    for words, stresses, lengths, fnames in test_dl:
        out = model(words)
        test_fnames.append(sum([[f] * l for l, f in zip(lengths, fnames)], []))
        test_guesses.append(torch.cat(unpad_sequence(out.argmax(dim=2).detach(), lengths, batch_first=True)))
        test_golds.append(torch.cat(unpad_sequence(stresses.detach(), lengths, batch_first=True)))
        kls = cross_entropy(torch.transpose(out, 1, 2), stresses, reduction="none")
        loss = sum([x.sum() for x in torch.nn.utils.rnn.unpad_sequence(kls, lengths, batch_first=True)])
        test_losses.append(loss.detach().item())

    if not test_guesses:
        logger.warning("No test data available! Skipping test evaluation.")
        test_acc = 0.0
    else:
        test_fnames = sum(test_fnames, [])
        test_guesses = torch.cat(test_guesses)
        test_golds = torch.cat(test_golds)
        test_acc = (test_guesses == test_golds).sum() / test_guesses.shape[0]

        logger.info("Final average instance loss (test): %f\n Test accuracy: %f",
                   sum(test_losses) / (len(test_losses) * args.batch_size), test_acc)

    if test_guesses:
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

    # Save model and metadata
    save_dict = {
        'model': model,
        'word_vocab': word_vocab,
        'stress_vocab': stress_vocab,
        'train_indices': list(range(0, train_count)),
        'dev_indices': list(range(train_count, train_count + dev_count)),
        'test_indices': list(range(train_count + dev_count, len(data))),
        'raw_data': raw_data,
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
