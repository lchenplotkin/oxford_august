import re
import csv
from itertools import product

# Load variable stress patterns from CSV
def load_variable_stress_dict(path):
    var_dict = {}
    with open(path, encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            word = row[0].strip().lower()
            patterns = [p.strip() for p in row[1:] if p.strip()]
            if patterns:
                var_dict[word] = patterns
    return var_dict

# Simple syllable-based approximation
def syllabify(word):
    return re.findall(r'[^aeiouy]*[aeiouy]+(?:[^aeiouy]*)?', word.lower()) or [word]

def is_final_e(word, sylls):
    return word.endswith("e") and len(sylls) > 1 and re.search(r'[^aeiouy]e$', word)

# Generate stress options for a word
def generate_options(word, next_word, var_dict):
    lw = word.lower()
    if lw in var_dict:
        return var_dict[lw]
    sylls = syllabify(word)
    options = []

    # Regular syllabification
    options.append("u" * len(sylls))

    # Final -e elision default if next word starts with vowel/h
    if is_final_e(word, sylls):
        if next_word and next_word[0].lower() in "aeiouh":
            options.append("u" * (len(sylls) - 1))  # elide
        else:
            options.insert(0, "u" * (len(sylls) - 1))  # try it early if needed
    return options

# Ideal alternating patterns
def alternating(length, headless=False):
    base = ['S', 'u'] if headless else ['u', 'S']
    return (base * ((length + 1) // 2))[:length]

def matches_alternating(stress_list, headless=False):
    flat = list(''.join(stress_list))
    return flat == alternating(len(flat), headless)

# Try scanning a line with both normal and headless meters
def try_all_scans(words, var_dict, target=10):
    next_words = words[1:] + [None]
    all_options = [generate_options(w, nw, var_dict) for w, nw in zip(words, next_words)]

    for combo in product(*all_options):
        total = sum(len(p) for p in combo)

        # Try regular 10-syllable scan
        if total == target and matches_alternating(combo, headless=False):
            return list(zip(words, combo)), "normal"

        # Try headless 9-syllable scan
        if total == target - 1 and matches_alternating(combo, headless=True):
            return list(zip(words, combo)), "headless"

    return None, "failed"

# Scan a text file and write output
def scan_file(input_file, dict_file, output_csv, target=10):
    var_dict = load_variable_stress_dict(dict_file)

    with open(input_file, encoding='utf-8') as f, open(output_csv, 'w', newline='', encoding='utf-8') as out:
        writer = csv.writer(out)
        for line in f:
            line = line.strip()
            if not line:
                continue
            words = re.findall(r"\b[\w']+\b", line)
            result, status = try_all_scans(words, var_dict, target)

            if result:
                writer.writerow([w for w, _ in result])
                writer.writerow([s for _, s in result])
                writer.writerow([])  # Blank line between entries
            else:
                print(f"⚠️ Could not scan (target={target}): {line}")
