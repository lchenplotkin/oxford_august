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
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
        
    with gzip.open(args.output, "wt") as ofd:
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
                    ofd.write(
                        json.dumps(
                            {
                                "line" : row["LINE_NUMBER"],
                                "text" : row["OXFORD_TEXT"],
                                "scansion" : row["OXFORD_SCANSION"],
                                "source" : os.path.basename(fname),
                                "consistent" : len(words) == len(stresses),
                                "original" : row
                            }
                        ) + "\n"
                    )
