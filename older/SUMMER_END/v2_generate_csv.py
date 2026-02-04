import csv
from scansion import *
import re
import string
import numpy as np
from sheets_to_gui import *

# Mapping of Riverside variants to Oxford normalized forms
VARIANT_MAPPINGS = {
	('naught', 'not'): ('nought@adv', 'not@adv'),
	('naught', 'nat'): ('nought@adv', 'not@adv'),
	('nought', 'not'): ('nought@adv', 'not@adv'),
	('nought', 'nat'): ('nought@adv', 'not@adv'),
	('nothing', 'no thing'): ('no-thing%adv', 'no@gram_adj thing@n'),
	('often', 'ofte'): ('often@adv', 'ofte@adv'),
	('namore', 'no more'): ('no_more@adv%comp', 'no@gram_adj more@adv%comp'),
	('farewell', 'fare well'): ('farewel@interj', 'faren@v%imp wel@adv'),
}

def get_variant_tag(riv_word, ox_word, riv_tag):
	"""
	Check if the word pair is a known variant and return the appropriate Oxford tag.
	Returns (oxford_tag, is_variant) where is_variant is True if this is a known variant.
	"""
	riv_lower = riv_word.lower().strip()
	ox_lower = ox_word.lower().strip()
	
	# Check direct mappings
	if (riv_lower, ox_lower) in VARIANT_MAPPINGS:
		riv_pattern, ox_pattern = VARIANT_MAPPINGS[(riv_lower, ox_lower)]
		# Extract the tag type from riverside tag
		if riv_tag.startswith('{*') and riv_tag.endswith('*}'):
			riv_tag_content = riv_tag[2:-2]
			# Check if this matches the expected pattern
			if riv_tag_content.startswith(riv_pattern.split('@')[0]):
				return '{*' + ox_pattern + '*}', True
	
	return riv_tag, False

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
	if word.strip() in ['compaignie','compaignye','companie']:
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
	text = re.sub(r'(.)\1+', r'\1', text)
	text = text.replace('y', 'i').replace('z', 's').replace('k','c')
	text = text.replace('uw', 'u').replace('ow', 'ou').replace('ov', 'ou').replace('gh', 'w')
	text = text.replace('a', '').replace('e','').replace('i','').replace('o','').replace('u','')
	return text.strip()

def strip_annotations(line):
	marked = re.sub(r'\b(\w*\d|\d+[a-zA-Z])\b\s+', r'\1||| ', line)
	parts = marked.split('|||')
	if len(parts) < 2:
		return line.strip(), [line]

	line = parts[1]
	line = line.replace('{', ' ').split(' ')
	newline = ""

	for word in line:
		if '}' not in word:
			newline += word + ' '

	newline = newline.strip()
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
			has_green_variant = False


			# Oxford tags - with variant handling
			oxford_tags = []
			if matched == "DIFF":
				riv_tags = list(tags)
				riv_words_normed = []
				for word in riv_words:
					riv_words_normed.append(norm_word(word))

				ind = 0
				matched = ""
				for ox_word in ox_words:
					if len(ox_words) == len(riv_words):
						if norm_word(ox_word)!=riv_words_normed[ind]:
							matched = "DIFF"
					ind+=1

				if matched == "DIFF":
					for ox_word in ox_words:
						if norm_word(ox_word) in riv_words_normed:
							idx = riv_words_normed.index(norm_word(ox_word))
							riv_tag = riv_tags[idx]
							riv_word = riv_words[idx]
							
							# Check if this is a known variant
							ox_tag, is_variant = get_variant_tag(riv_word, ox_word, riv_tag)
							oxford_tags.append(ox_tag)
								
							if is_variant:
								has_green_variant = True
							else:
								oxford_tags.append('')
				else:
					oxford_tags = tags
			else:
				oxford_tags = tags

			# Set flag to green if we found a known variant
			if has_green_variant:
				flag = 'GREEN'


			
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
			# Write all rows with updated flag
			row = [riverside_file, oxford_file, output_csv[:-4], line_num, matched, flag, num_sybs]
			row.extend(tag.strip("{}*") for tag in oxford_tags)
			writer.writerow(row)


			# CAT reconstruction if match
			if matched != "DIFF":
				marked = re.sub(r'\b(\w*\d)\b\s+', r'\1|||', cat_line)
				cleaned_split = marked.split('|||')
				out_line = ""
				for word, tag in zip(ox_words, oxford_tags):
					out_line += word + tag + ' '
				cat_output += numbering + ' ' + out_line + '\n'
			else:
				cat_output += numbering + ' \n'

			writer.writerow('')

	with open(output_cat, "w") as f:
		f.write(cat_output)


# Run
target = 8
make_formatted('data/riverside_cats/BD_riv.cat','data/oxford_txts/BD_oxford.txt','data/csvs/BD.csv','data/oxford_cats/BD_oxford.cat')
convert_file('data/csvs/BD.csv', 'data/csvs/BD_gui.csv')

target = 10

for name in ['GP','HF','PF','TC1','TC2','TC3','TC4','TC5','ClT','CYT','KnT','MancT','MilT','MkT','NPT','PardT','PhyT','PrT','RvT','ShipT','SNT','SqT','Thop','FranT','FriT','MerT','SumT','WBPro','WBT']:
	make_formatted('data/riverside_cats/'+name+'_riv.cat','data/oxford_txts/'+name+'_oxford.txt','data/csvs/'+name+'.csv','data/oxford_cats/'+name+'_oxford.cat')
	convert_file('data/csvs/'+name+'.csv','data/csvs/'+name+'_gui.csv')
