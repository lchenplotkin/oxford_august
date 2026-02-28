import json
import gzip
import logging
import argparse


logger = logging.getLogger("analyze_results")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    lines = []
    with gzip.open(args.input, "rt") as ifd:
        for line in ifd:
            lines.append(json.loads(line))

    # any kind of error analysis can be performed and then written out here
            
    with open(args.output, "wt") as ofd:
        pass
