import json
import gzip
import logging
from copy import deepcopy
import random
import os.path
import argparse
import torch
from torch.nn import Module, Embedding, LSTM, Linear
from torch.nn.utils.rnn import pad_sequence, unpad_sequence
from torch.nn.utils import clip_grad_norm_
from torch.nn.functional import cross_entropy
from torch.optim import AdamW
from torch.utils.data import DataLoader
from rnn import RNNModel


logger = logging.getLogger("train_model")


def collate(items):
    words = pad_sequence([x for x, _, _, _, _ in items], batch_first=True)
    stresses = pad_sequence([x for _, x, _, _, _ in items], batch_first=True)
    lengths = torch.tensor([x for _, _, x, _, _ in items])
    books = [x for _, _, _, x, _ in items]
    return (words, stresses, lengths, books)
        

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--train")
    parser.add_argument("--dev")
    parser.add_argument("--test")    
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--embedding_size", type=int, default=64)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--learning_rate", type=float, default=0.005)
    parser.add_argument("--use_loss_for_lr", action="store_true", default=False)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--unidirectional", action="store_true", default=False)
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

    
    stress_vocab = {None : 0}
    word_vocab = {None : 0}
    splits = {}

    for split, fname in [
            ("train", args.train),
            ("dev", args.dev),
            ("test", args.test)
            ]:
        splits[split] = []
        with gzip.open(fname, "rt") as ifd:
            for line in ifd:
                j = json.loads(line)
                words = j["text"].split()
                stresses = j["scansion"].split()
                if len(words) == len(stresses):
                    splits[split].append(
                        (
                            torch.tensor([word_vocab.setdefault(w, len(word_vocab)) for w in words]),
                            torch.tensor([stress_vocab.setdefault(s, len(stress_vocab)) for s in stresses]),
                            len(words),
                            j["source"],
                            j["line"]
                        )
                    )
    logger.info(
        "Split data into %d/%d/%d train/dev/test lines, word vocabulary size=%d, stress pattern vocabulary size=%d",
        len(splits["train"]),
        len(splits["dev"]),
        len(splits["test"]),
        len(word_vocab),
        len(stress_vocab)
    )
    
    train_dl = DataLoader(splits["train"], batch_size=args.batch_size, collate_fn=collate, shuffle=True, generator=torch.Generator(device=device))
    dev_dl = DataLoader(splits["dev"], batch_size=args.batch_size, collate_fn=collate)
    test_dl = DataLoader(splits["test"], batch_size=args.batch_size, collate_fn=collate)

    model = RNNModel(args.embedding_size, word_vocab, args.hidden_size, stress_vocab, not args.unidirectional)
    
    opt = AdamW(model.parameters(), lr=args.learning_rate)

    best_dev_score = None
    best_state_dict = None
    for epoch in range(args.max_epochs):
        train_losses = []
        dev_losses = []
        dev_guesses = []
        dev_golds = []
        model.train()
        for words, stresses, lengths, fnames in train_dl:
            out = model(words)
            ces = cross_entropy(torch.transpose(out, 1, 2), stresses, reduction="none")
            uces = unpad_sequence(ces, lengths, batch_first=True)
            loss = sum([x.sum() for x in uces])
            loss.backward()
            clip_grad_norm_(model.parameters(), 3)
            opt.step()
            opt.zero_grad()
            train_losses.append(loss.detach().item())
        train_loss = sum(train_losses) / (len(train_losses) * args.batch_size)
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
            "Epoch %d average instance loss (train/dev): %f/%f\n  Dev accuracy: %f",
            epoch + 1,
            train_loss,
            dev_loss,
            dev_acc
        )        
        dev_score = -dev_loss if args.use_loss_for_lr else dev_acc
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
        
    test_fnames = sum(test_fnames, [])
    test_guesses = torch.cat(test_guesses)
    test_golds = torch.cat(test_golds)
    test_acc = (test_guesses == test_golds).sum() / test_guesses.shape[0]        
    logger.info("Final average instance loss (test): %f\n  Test accuracy: %f", sum(test_losses) / (len(test_losses) * args.batch_size), test_acc)
    
    with open(args.output, "wb") as ofd:
        torch.save(model, ofd)
