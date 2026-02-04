import os
import json
from collections import defaultdict, Counter

def parse_line(line):
    """Extract (word, tag) pairs from a line like 'Whan{*whan@adv&conj*}'."""
    tokens = line.strip().split()
    result = []
    for token in tokens:
        if '{*' in token and '*}' in token:
            word_part, tag_part = token.split('{*')
            tag = tag_part.rstrip('*}')
            result.append((word_part.lower(), tag))
    return result

def build_tag_index_from_directory(directory_path):
    """Read all files in a directory and return a word → tag frequency index."""
    index = defaultdict(Counter)

    for filename in os.listdir(directory_path):
        full_path = os.path.join(directory_path, filename)
        if os.path.isfile(full_path) and filename.lower().endswith('.cat'):
            with open(full_path, 'r', encoding='utf-8') as f:
                for line in f:
                    for word, tag in parse_line(line):
                        index[word][tag] += 1

    return index

def save_index(index, output_path):
    """Save the index in the 'word total;tag count,...' format."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for word in sorted(index):
            total = sum(index[word].values())
            tags = ','.join(f'{tag} {count}' for tag, count in index[word].items())
            f.write(f'{word} {total};{tags}\n')

def load_index(filepath):
    """Reload the saved index from file."""
    index = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            word, rest = line.strip().split(' ', 1)
            count_part, tag_part = rest.split(';')
            tags = dict(tag.strip().split(' ') for tag in tag_part.split(','))
            index[word] = {k: int(v) for k, v in tags.items()}
    return index

# Example usage:
directory_path = 'data/oxford_cats'  # Replace with your directory path
output_file = 'data/oxford_prelim.json'

index = build_tag_index_from_directory(directory_path)
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(index, f, indent=2)

