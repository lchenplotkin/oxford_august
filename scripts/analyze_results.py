import sys
import re
import json
import gzip
import logging
import argparse
import pandas


logger = logging.getLogger("analyze_results")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # predictions
    
    locations = []
    stress_attempts = 0
    stress_correct = 0
    scan_attempts = 0
    scan_correct = 0
    with gzip.open(args.input, "rt") as ifd:
        for line in ifd:
            j = json.loads(line)
            o_scan = j["original"]["scansion"].split()
            #for i, (word, golds, guesses) in enumerate(zip(j["words"], j["stresses"], j["predictions"])):
            annotated = len(j["stresses"]) == len(j["words"])
                   
            for word_num, word in enumerate(j["words"]):
                guesses = j["predictions"][word_num]
                
                #
                if annotated:
                    golds = j["stresses"][word_num]
                    scan_attempts += 1
                    if golds == guesses:
                        scan_correct += 1
                    stress_attempts += max([len(golds), len(guesses)])                        
                    for gold, guess in zip(golds, guesses):
                        if gold == guess:
                            stress_correct += 1

                if word_num < len(o_scan):
                    scan = o_scan[word_num]
                else:
                    scan = None

                locations.append(
                    {
                        "document" : re.sub(r"^(oxford_txts/)?(.*?)(_gui_complete\.csv|\.txt)$", r"\2", j["original"]["source"]),
                        "line_number" : int(re.sub(r"^(.*\s+)?(\d+)$", r"\2", j["original"]["line"])),
                        "gold" : "".join(golds) if annotated else None,
                        "guess" : "".join([g for g in guesses if g]),
                        "word_number" : word_num,
                        "word" : word,
                        "human_scan" : scan
                    }
                )
    print("Stress accuracy: {:.3f}\nScan accuracy: {:.3f}".format(stress_correct / stress_attempts, scan_correct / scan_attempts))
    df = pandas.DataFrame(locations)
    sys.exit()
    print(df.groupby("document").count())
    print(len(df))
    dd = df[df.gold != df.guess] #.groupby("document").count().line_number
    dd = dd[dd.gold != ""]
    print(len(dd))
    
    with open(args.output, "wt") as ofd:
        pass
