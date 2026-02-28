import json
import csv
import gzip
import logging
from glob import glob
import random
import os.path
import argparse


logger = logging.getLogger("preprocess_data")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="full_dataset")
    parser.add_argument("--proportions", nargs="+", type=float, required=True)
    parser.add_argument("--outputs", nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    
    if args.seed != None:
        random.seed(args.seed)
        
    stress_vocab = {}
    word_vocab = {}
    data = []
    for fname in glob(os.path.join(args.data_path, "*csv")):
        if os.path.basename(fname) == "combined.csv":
            continue
        with open(fname, "rt") as ifd:
            for row in csv.DictReader(ifd):
                o_words = row["OXFORD_TEXT"].split()
                r_words = row["RIVERSIDE_TEXT"].split()
                o_stresses = row["OXFORD_SCANSION"].split()
                r_stresses = row["RIVERSIDE_SCANSION"].split()
                words = o_words
                stresses = o_stresses
                if len(words) == len(stresses):
                    data.append(
                        {
                            "line" : row["LINE_NUMBER"],
                            "text" : row["OXFORD_TEXT"],
                            "scansion" : row["OXFORD_SCANSION"],
                            "source" : os.path.basename(fname)
                        }
                    )

    random.shuffle(data)

    counts = [int(p * len(data)) for p in args.proportions]

    for fname, ct in zip(args.outputs, counts):
        with gzip.open(fname, "wt") as ofd:
            for item in data[:ct]:
                ofd.write(json.dumps(item) + "\n")
            data = data[ct:]
            logger.info("Wrote %d instances to %s", ct, fname)
