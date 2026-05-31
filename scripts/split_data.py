import json
import csv
import gzip
import logging
from glob import glob
import random
import os.path
import argparse


logger = logging.getLogger("split_data")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--proportions", nargs="+", type=float, required=True)
    parser.add_argument("--outputs", nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if len(args.proportions) != len(args.outputs):
        raise Exception("The number of proportions and the number of file outputs must be the same!")
    
    if args.seed != None:
        random.seed(args.seed)
        
    data = []
    with gzip.open(args.input, "rt") as ifd:
        for line in ifd:
            data.append(json.loads(line))

    random.shuffle(data)

    counts = [int(p * len(data)) for p in args.proportions]

    for fname, ct in zip(args.outputs, counts):
        with gzip.open(fname, "wt") as ofd:
            for item in data[:ct]:
                ofd.write(json.dumps(item) + "\n")
            data = data[ct:]
            logger.info("Wrote %d instances to %s", ct, fname)
