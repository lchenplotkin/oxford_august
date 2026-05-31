from itertools import accumulate
import json
import gzip
import logging
import random
import os.path
import argparse
import re
import torch
from torch.nn.utils.rnn import pad_sequence, unpad_sequence
import models
from utils import run_epoch
from data import DataSet, DataLoader


logger = logging.getLogger("apply_model")


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

    model = torch.load(args.model, weights_only=False, map_location=device)

    data = DataSet.from_files(args.inputs, reference=model).to(device)
    dl = DataLoader(data, batch_size=args.batch_size, device=device)

    logger.info(
        "Applying model to %d lines",
        data._length,
    )
    
    model.eval()

    with gzip.open(args.output, "wt") as ofd:
        scores = run_epoch(model, dl, out=ofd)
        print("Stress/scan accuracy: {:.3f}/{:.3f}".format(scores["stress_accuracy"], scores["scan_accuracy"]))
