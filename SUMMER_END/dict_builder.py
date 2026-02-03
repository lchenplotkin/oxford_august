import os
import pandas as pd
import json
from collections import defaultdict, Counter

def build_tag_index_from_directory(directory_path):
	"""Read all files in a directory and return a word → tag frequency index."""
	index = defaultdict(Counter)

	for filename in os.listdir(directory_path):
		full_path = os.path.join(directory_path, filename)
		if os.path.isfile(full_path) and filename.lower().endswith('complete.csv'):
			df = pd.read_csv(full_path,encoding='utf-8')
			for idx, row in df.iterrows():
				ox_words = row['OXFORD_TEXT'].split(' ')	
				ox_tags = row['OXFORD_TAGGING'].split(' ')
				
				for ox_word, ox_tag in zip(ox_words,ox_tags):
					index[ox_word][ox_tag]+=1

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
directory_path = 'for_gui/done'  # Replace with your directory path
output_file = 'data/oxford.json'

index = build_tag_index_from_directory(directory_path)
with open(output_file, 'w', encoding='utf-8') as f:
	json.dump(index, f, indent=2)

