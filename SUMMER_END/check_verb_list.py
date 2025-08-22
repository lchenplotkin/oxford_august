import os
import string
import csv
import pandas as pd
from collections import defaultdict, Counter
import matplotlib.pyplot as plt


def parse_tagged_text(oxford_text, oxford_tagging):
	"""Extract words from oxford_text and tags from oxford_tagging"""
	if pd.isna(oxford_text) or pd.isna(oxford_tagging) or oxford_text == '' or oxford_tagging == '':
		return [], [], []

	# Clean and split the text
	words = oxford_text.lower().replace(',', '').replace('.', '').replace('!', '').replace('?', '').replace('°', '').replace('¶', '').strip().split()

	# Parse tags from oxford_tagging
	tag_tokens = oxford_tagging.strip().split()
	tags = []
	headwords = []

	for token in tag_tokens:
		if '@' in token:
			if token not in ["--@dash", ".@ellipsis"]:
				parts = token.split('@')
				headword = parts[0].lower()
				tag_part = parts[1] if len(parts) > 1 else ''
				#tag_part = re.sub('r\d','',tag_part)
				#tag_part = tag_part.replace('_','')

				tag = tag_part

				headwords.append(headword)
				tags.append(tag)

	# Ensure all lists are the same length (trim to shortest)
	if len(words) != len(tags):
		print(oxford_text, oxford_tagging, words, tags)
		print(len(words), len(tags))
	min_len = min(len(words), len(tags), len(headwords))
	return words[:min_len], headwords[:min_len], tags[:min_len]

#import docx

# Initialize Word document
#doc = docx.Document()
#doc.add_heading('Verb Rules Analysis in the Oxford Chaucer', 0)
#
# Configuration
base_csv_dir = 'data/csvs'
verb_list_file = 'verb_forms_gold.csv'
verb_list_file = 'fixed_with_headwords.csv'
verb_df = pd.read_csv(verb_list_file, encoding='utf-8')
known_headwords = list(verb_df['headword'])

sucess = 0
fail = 0
for root, dirs, files in os.walk(base_csv_dir):
	for file in files:
		if not file.endswith('_gui.csv'):
			continue

		csv_path = os.path.join(root, file)
		rel_path = os.path.relpath(root, base_csv_dir)
		file_id = os.path.join(rel_path, file) if rel_path != '.' else file

		df = pd.read_csv(csv_path, encoding='utf-8')

		for idx, row in df.iterrows():
			if row["MATCH"] != "DIFF":
				oxford_text = row['OXFORD_TEXT']
				oxford_tagging = row['OXFORD_TAGGING']
				line_number = row['LINE_NUMBER']
				filename = row['OXFORD_FILENAME']

				words, headwords, tags = parse_tagged_text(oxford_text, oxford_tagging)

				for j, (word, headword, tag) in enumerate(zip(words, headwords, tags)):
					if tag.startswith('v'):
						if headword in known_headwords:
							sucess += 1
						else:
							print(word,headword,tag)
							fail+=1
							known_headwords.append(headword)
							
print(sucess, fail)
#print(sucess/(sucess+fail))

