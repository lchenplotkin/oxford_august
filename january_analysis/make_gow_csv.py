from pathlib import Path
import csv
from scansion import *
import re
import string
import numpy as np

folder = Path("gow")
target = 8

def strip_annotations(line):
	# Add a marker `|||` after each word ending in a digit OR starting with digits followed by a single letter
	# This pattern matches:
	# - \w*\d\b: words ending in a digit (original pattern)
	# - \d+[a-zA-Z]\b: words starting with digits followed by a single letter (new pattern)
	# - \w*\d\*\b: words ending in a digit followed by * (new requirement)
	marked = re.sub(r'\b(\w*\d(?:\*)?|\d+[a-zA-Z])\b\s+', r'\1||| ', line)
	print(line.split('-gow'))
	line_num = line.split('-gow')[0] + '-gow' + line.split('-gow')[1].split(' ')[1]
	line = ' '.join(line.split('-gow')[1].split(' ')[2:])

	# Split on the marker and take the second part (index 1)
	parts = line_num, line

	line = parts[1]
	line = line.replace('{', ' ').split(' ')
	newline = ""

	for word in line:
		if '}' not in word:
			newline += word + ' '

	newline = newline.strip()

	# Return the processed line and all split parts
	return newline, parts[0]

def extract_tags(line):
	return re.findall(r'\{\*.*?\*\}', line)

for file_path in folder.rglob("*.cat"):
	with open(file_path, encoding='utf-8') as f:
		cat_lines = [line.rstrip('\n') for line in f if line.strip() != '']
	
	with open('gow_csvs/'+file_path.name.strip('.cat')+'.csv','w',newline='') as csvfile:
		writer = csv.writer(csvfile)
		writer.writerow(['FILENAME','LINE_NUMBER','OG_TEXT','TEXT','NUM_SYBS','SCANSION','TAGGING'])
		for cat_line in cat_lines:
			gow_line, line_num = strip_annotations(cat_line) 
			og_gow_line = gow_line
			gow_line = ''.join(ch for ch in gow_line if ch in string.ascii_letters + ' \n')
			gow_words = gow_line.split()
			tags = extract_tags(cat_line)
			
			stresses, num_sybs = scan(gow_line, target)
			row = [file_path, line_num, og_gow_line, gow_line]
			stress_str = ' '.join(stress for stress in stresses)
			row.append(num_sybs)
			row.append(stress_str)
			tag_str = ' '.join(tag.strip("{}*") for tag in tags)
			row.append(tag_str)
			writer.writerow(row)
