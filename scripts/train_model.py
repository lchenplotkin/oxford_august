import json
import gzip
import logging
from copy import deepcopy
import random
from itertools import accumulate
import os.path
import re
import argparse
import torch
from torch.nn import Module, Embedding, LSTM, Linear
from torch.nn.utils.rnn import pad_sequence, unpad_sequence
from torch.nn.utils import clip_grad_norm_
from torch.nn.functional import cross_entropy
from torch.optim import AdamW
import models
from utils import run_epoch
from data import DataSet, DataLoader


logger = logging.getLogger("train_model")


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
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--limit", type=int, help="Limit each of train/dev/test to this number of lines")
    parser.add_argument("--validation_metric", choices=["loss", "stress_accuracy", "scan_accuracy"], default="stress_accuracy")
    parser.add_argument("--unidirectional", action="store_true", default=False)
    parser.add_argument("--device")
    parser.add_argument("--granularity", choices=["word_scan", "word_stress", "char_scan", "char_stress", "hierarchical_scan", "hierarchical_stress"], default="word_scan")
    args = parser.parse_args()

    metric_weight = 1 if args.validation_metric == "loss" else -1
    
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
    logger.info("Using device = %s", device) #torch.get_default_device())

    model_class = getattr(models, "{}Model".format(args.granularity.replace("_", " ").title().replace(" ", ""))) #[args.model_type]

    scan_vocab = {"" : 0}
    stress_vocab = {None : 0, False : 1}
    char_vocab = {None : 0}
    word_vocab = {None : 0}
    splits = {}

    splits["train"] = DataSet.from_files(args.train, limit=args.limit, zeroing_proportion=0.1).to(device)
    for split_name, fname in [
            ("dev", args.dev),
            ("test", args.test)
            ]:
        splits[split_name] = DataSet.from_files(fname, reference=splits["train"], limit=args.limit).to(device)

    logger.info(
        "Split data into %d/%d/%d train/dev/test lines, #unique words=%d, characters=%d, word scans=%d, stress marks=%d",
        len(splits["train"]),
        len(splits["dev"]),
        len(splits["test"]),
        len(splits["train"]._id2word),
        len(splits["train"]._id2char),
        len(splits["train"]._scan2id),
        len(splits["train"]._id2stress),        
    )

    train_dl = DataLoader(splits["train"], batch_size=args.batch_size, shuffle=True, device=device)
    dev_dl = DataLoader(splits["dev"], batch_size=args.batch_size, device=device)
    test_dl = DataLoader(splits["test"], batch_size=args.batch_size, device=device)

    model = model_class(
        args.embedding_size,
        splits["train"]._word2id,
        splits["train"]._char2id,
        splits["train"]._scan2id,
        splits["train"]._stress2id,
        args.hidden_size,
        not args.unidirectional
    ).to(device)
    
    print(model)

    opt = AdamW(model.parameters(), lr=args.learning_rate)
    
    best_dev_score = None
    best_state_dict = None
    for epoch in range(args.max_epochs):
        train_results = run_epoch(model, train_dl, optimizer=opt)
        dev_results = run_epoch(model, dev_dl)
        logger.info(
            "Epoch %d average loss per word (train/dev): %.4f/%.4f\n  Dev stress/scan accuracy: %.4f/%.4f",
            epoch + 1,
            train_results["loss"],
            dev_results["loss"],
            dev_results["stress_accuracy"],
            dev_results["scan_accuracy"]
        )        
        if not best_dev_score or (metric_weight * dev_results[args.validation_metric]) < best_dev_score:
            logger.info("Saving new best model")
            best_dev_score = metric_weight * dev_results[args.validation_metric]
            best_state_dict = deepcopy(model.state_dict())

    model.load_state_dict(best_state_dict)
    test_results = run_epoch(model, test_dl)

    logger.info("Final test performance:\n  Average loss per word: %.4f\n  Stress/scan accuracy: %.4f/%.4f", test_results["loss"], test_results["stress_accuracy"], test_results["scan_accuracy"])
    
    with open(args.output, "wb") as ofd:
        torch.save(model, ofd)
