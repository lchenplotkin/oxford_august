import json
import gzip
import logging
import random
import os.path
import argparse
import torch
from torch.nn.utils.rnn import pad_sequence, unpad_sequence
from torch.utils.data import DataLoader
from rnn import RNNModel


logger = logging.getLogger("apply_model")


def collate(items):
    words = pad_sequence([x for x, _, _, _, _, _ in items], batch_first=True)
    stresses = pad_sequence([x for _, x, _, _, _, _ in items], batch_first=True)
    lengths = torch.tensor([x for _, _, x, _, _, _ in items])
    books = [x for _, _, _, x, _, _ in items]
    lines = [x for _, _, _, _, x, _ in items]
    fnames = [x for _, _, _, _, _, x in items]
    return (words, stresses, lengths, books, lines, fnames)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=1024)
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

    model = torch.load(args.model, weights_only=False)

    data = []
    for fname in args.inputs:
        with gzip.open(fname, "rt") as ifd:
            for line in ifd:
                j = json.loads(line)
                words = j["text"].split()
                stresses = j["scansion"].split()
                if len(words) == len(stresses):
                    data.append(
                        (
                            torch.tensor([model.word_vocab.get(w, 0) for w in words]),
                            torch.tensor([model.stress_vocab.get(s, 0) for s in stresses]),
                            len(words),
                            j["source"],
                            j["line"],
                            fname
                        )
                    )

    logger.info(
        "Applying model to %d lines",
        len(data),
    )
    
    dl = DataLoader(data, batch_size=args.batch_size, collate_fn=collate, shuffle=True, generator=torch.Generator(device=device))
    
    model.eval()

    r_word_vocab = {v : k for k, v in model.word_vocab.items()}
    r_stress_vocab = {v : k for k, v in model.stress_vocab.items()}
    with gzip.open(args.output, "wt") as ofd:
        for words, stresses, lengths, books, lines, fnames in dl:
            out = model(words)
            res = unpad_sequence(out.argmax(dim=2).detach(), lengths, batch_first=True)
            for i, line in enumerate(lines):
                ofd.write(
                    json.dumps(
                        {
                            "words" : [r_word_vocab.get(int(w)) for w in words[i][:lengths[i]]],
                            "true_stresses" : [r_stress_vocab.get(int(s)) for s in stresses[i][:lengths[i]]],
                            "inferred_stresses" : [r_stress_vocab.get(int(s)) for s in res[i][:lengths[i]]],
                            "book" : books[i],
                            "line" : line,
                            "split" : os.path.basename(fnames[i])
                        }
                    ) + "\n"
                )
