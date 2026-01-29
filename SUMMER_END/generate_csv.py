import csv
from scansion import *
import re
import string
import numpy as np
from sheets_to_gui import *



def norm_word(word):
	word = word.lower()
	if word.strip() in ['it','hyt']:
		return 'it'
	if word.strip() in ['be','ben']:
		return 'ben'
	if word.strip() in ['foloweth','folweth']:
		return 'folweth'
	if word.strip() in ['do','don']:
		return 'don'
	if word.strip() in ['han','have']:
		return 'han'
	if word.strip() in ['fro','from']:
		return 'from'
	if word.strip() in ['service','servyce','servise','servyse']:
		return 'service'
	if word.strip() in ['wonderly','wonderlich']:
		return 'wonderly'
	if word.strip() in ['euery','every']:
		return 'every'
	if word.strip() in ['sorowe','sorwe']:
		return 'sorwe'
	if word.strip() in ['such','swich']:
		return 'such' 
	if word.strip() in ['compaignye','companie']:
		return 'companie'


	word = ''.join(ch for ch in word if ch in string.ascii_lowercase + ' \n')
	word = re.sub(r'(.)\1+', r'\1', word)
	word = word.replace('y', 'i').replace('z', 's').replace('k','c')
	word = word.replace('uw', 'u').replace('ow', 'ou').replace('ov', 'ou').replace('gh', 'w')
	word = word.replace('a','e').replace('e','o').replace('o','u').replace('u','i').replace('i','')
	return word.strip()

def normalize(text):

	text = text.lower()
	text = ''.join(ch for ch in text if ch in string.ascii_lowercase + ' \n')

	'''
	new_text = ''
	for word in text.split(' '):
		new_text += ' '
		new_text += norm_word(word).replace('e','')

	return new_text.strip()
	'''

	text = re.sub(r'(.)\1+', r'\1', text)
	text = text.replace('y', 'i').replace('z', 's').replace('k','c')
	text = text.replace('uw', 'u').replace('ow', 'ou').replace('ov', 'ou').replace('gh', 'w')
	text = text.replace('a', '').replace('e','').replace('i','').replace('o','').replace('u','')
	return text.strip()



def strip_annotations(line):
	# Add a marker `|||` after each word ending in a digit OR starting with digits followed by a single letter
	# This pattern matches:
	# - \w*\d\b: words ending in a digit (original pattern)
	# - \d+[a-zA-Z]\b: words starting with digits followed by a single letter (new pattern)
	marked = re.sub(r'\b(\w*\d|\d+[a-zA-Z])\b\s+', r'\1||| ', line)

	# Split on the marker and take the second part (index 1)
	parts = marked.split('|||')
	if len(parts) < 2:
		# If no marker found, return original line and empty split
		return line.strip(), [line]

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

def make_formatted(riverside_file, oxford_file, output_csv, output_cat):
	cat_output = ""
	with open(riverside_file, encoding='utf-8') as riv:
		cat_lines = [line.rstrip('\n') for line in riv if line.strip() != '']

	with open(oxford_file, 'r', encoding='utf-8') as oxford:
		oxford_lines = [line.rstrip('\n') for line in oxford if line.strip()!='']

	with open(output_csv, 'w', newline='') as csvfile:
		writer = csv.writer(csvfile)
		for cat_line, ox_line in zip(cat_lines, oxford_lines):
			riv_line, numbering = strip_annotations(cat_line)
			line_num = numbering
			og_riv_line = riv_line
			og_ox_line = ox_line

			row = [og_riv_line, "||", og_ox_line]
			writer.writerow(row)

			riv_line = ''.join(ch for ch in riv_line if ch in string.ascii_letters + ' \n')
			ox_line = ''.join(ch for ch in ox_line if ch in string.ascii_letters + ' \n')
			ox_words = ox_line.split()
			riv_words = riv_line.split()
			tags = extract_tags(cat_line)

			if normalize(ox_line) == normalize(riv_line):
				matched = ''
			else:
				matched = 'DIFF'

			flag = ''  # Default blank flag

			# Riverside words
			stresses, num_sybs = scan(riv_line, target)
			row = [riverside_file, oxford_file, output_csv[:-4], line_num, matched, flag, num_sybs]
			row.extend(riv_words)
			writer.writerow(row)

			# Riverside stresses
			row = [riverside_file, oxford_file, output_csv[:-4], line_num, matched, flag, num_sybs]
			row.extend(stresses)
			writer.writerow(row)

			# Riverside tags
			row = [riverside_file, oxford_file, output_csv[:-4], line_num, matched, flag, num_sybs]
			row.extend(tag.strip("{}*") for tag in tags)
			writer.writerow(row)

			# Oxford words
			stresses, num_sybs = scan(ox_line, target)
			row = [riverside_file, oxford_file, output_csv[:-4], line_num, matched, flag, num_sybs]
			row.extend(ox_words)
			writer.writerow(row)

			# Oxford stresses
			row = [riverside_file, oxford_file, output_csv[:-4], line_num, matched, flag, num_sybs]
			row.extend(stresses)
			writer.writerow(row)

			# Oxford tags
			row = [riverside_file, oxford_file, output_csv[:-4], line_num, matched, flag, num_sybs]
			if matched == "DIFF":
				riv_tags = list(tags)
				tags = []
				riv_words_normed = []
				for word in riv_words:
					riv_words_normed.append(norm_word(word))

				for word in ox_words:
					if norm_word(word) in riv_words_normed:
						tags.append(riv_tags[riv_words_normed.index(norm_word(word))])
					else:
						tags.append('')

			row.extend(tag.strip("{}*") for tag in tags)
			writer.writerow(row)


			# CAT reconstruction if match
			if matched != "DIFF":
				marked = re.sub(r'\b(\w*\d)\b\s+', r'\1|||', cat_line)
				cleaned_split = marked.split('|||')
				out_line = ""
				#out_line = cleaned_split[0] + ' ' + cleaned_split[1] + ' '
				for word, tag in zip(ox_words, tags):
					out_line += word + tag + ' '
				cat_output += numbering + ' ' + out_line + '\n'
			else:
				cat_output += numbering + ' \n'

			writer.writerow('')

	with open(output_cat, "w") as f:
		f.write(cat_output)


# Run
#target = 8
#make_formatted('data/riverside_cats/BD_riv.cat','data/oxford_txts/BD_oxford.txt','data/csvs/BD.csv','data/oxford_cats/BD_oxford.cat')
#convert_file('data/csvs/BD.csv', 'data/csvs/BD_gui.csv')

target = 10


#for name in ['MLT','LGW_FPro','GP','HF','PF','TC1','TC2','TC3','TC4','TC5','ClT','CYT','KnT','MancT','MilT','MkT','NPT','PardT','PhyT','PrT','RvT','ShipT','SNT','SqT','Thop','FranT','FriT','MerT','SumT','WBPro','WBT']:
for name in ['ShipT','SqPro','MerEpi','FranPro','PardPro','PrPro']:
	make_formatted('data/riverside_cats/'+name+'_riv.cat','data/oxford_txts/'+name+'_oxford.txt','data/csvs/'+name+'.csv','data/oxford_cats/'+name+'_oxford.cat')
	convert_file('data/csvs/'+name+'.csv','for_gui/to_do/'+name+'_gui.csv')
