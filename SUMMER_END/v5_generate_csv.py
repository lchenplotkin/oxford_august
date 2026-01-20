import csv
from scansion import *
import re
import string
import numpy as np
from sheets_to_gui import *
import json


# Load the oxford_prelim.json file at the start
with open('for_gui/oxford_prelim.json', 'r', encoding='utf-8') as f:
	oxford_prelim = json.load(f)


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
	if word.strip() in ['eyen','eighen']:
		return 'eyen'

	word = ''.join(ch for ch in word if ch in string.ascii_lowercase + ' \n')
	word = re.sub(r'(.)\1+', r'\1', word)
	word = word.replace('y', 'i').replace('z', 's').replace('k','c')
	word = word.replace('uw', 'u').replace('ow', 'ou').replace('ov', 'ou').replace('gh', 'w')
	word = word.replace('a','e').replace('e','o').replace('o','u').replace('u','i').replace('i','')
	return word.strip()

def preprocess_variants(text):
	"""Normalize only specific word variants before comparison"""
	replacements = {
		'hyt': 'it',
		'hit':'it',
		'ben': 'be',
		'been':'be',
		'doon':'don',
		'done':'don',
		'foloweth': 'folweth',
		'do': 'don',
		'have': 'han',
		'fro': 'from',
		'defaulte': 'defaute',
		'sorow':'sorwe',
		'servyce': 'service',
		'servise': 'service',
		'servyse': 'service',
		'wonderlich': 'wonderly',
		'euery': 'every',
		'sorowe': 'sorwe',
		'swich': 'such',
		'compaignie': 'companie',
		'compaignye': 'companie',
		'eighen': 'eyen',
		'i':'ich',
		'comen':'come',
		'cruwel':'cruel',
		'crewel':'cruel'
	}
	
	words = text.lower().split()
	i=0
	for word in words:
		word = ''.join(ch for ch in word if ch in string.ascii_lowercase + ' \n')
		word = re.sub(r'(.)\1+', r'\1', word)
		words[i]=word
		i+=1

	normalized_words = [replacements.get(word, word) for word in words]
	return ' '.join(normalized_words)

def normalize(text):

	text = preprocess_variants(text)
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

def check_naught_not_only_diff(riv_words, ox_words, riv_line, ox_line):
	"""
	Check if the only difference between lines (besides spelling) is naught/nought vs not/nat.
	Returns True if this is the only difference, False otherwise.
	"""
	# Normalize the full lines
	riv_normalized = normalize(riv_line)
	ox_normalized = normalize(ox_line)
	
	# If they're identical after normalization, not a naught/not case
	if riv_normalized == ox_normalized:
		return False
	
	# Check if word counts match
	if len(riv_words) != len(ox_words):
		return False
	
	# Track differences
	differences = []
	for i, (riv_w, ox_w) in enumerate(zip(riv_words, ox_words)):
		riv_clean = riv_w.lower().strip()
		ox_clean = ox_w.lower().strip()
		
		# Check if words are different after normalization
		if normalize(riv_clean) != normalize(ox_clean):
			# Check if it's a naught/not variant pair
			naught_variants = {'naught', 'nought'}
			not_variants = {'not', 'nat'}
			
			riv_is_naught = riv_clean in naught_variants
			riv_is_not = riv_clean in not_variants
			ox_is_naught = ox_clean in naught_variants
			ox_is_not = ox_clean in not_variants
			
			# Valid if one is naught and other is not
			if (riv_is_naught and ox_is_not) or (riv_is_not and ox_is_naught):
				differences.append(i)
			else:
				# This is a different kind of difference
				return False
	
	# Return True only if we found exactly the naught/not differences
	return len(differences) > 0

def make_formatted(riverside_file, oxford_file, output_csv, output_cat):
	tolerance = 100
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
			ox_line = ox_line.replace("’"," ")
			riv_line = ''.join(ch for ch in riv_line if ch in string.ascii_letters + ' \n')
			ox_line = ''.join(ch for ch in ox_line if ch in string.ascii_letters + ' \n')
			ox_words = ox_line.split()
			riv_words = riv_line.split()
			tags = extract_tags(cat_line)

			row = [og_riv_line, "||", og_ox_line]
			writer.writerow(row)

			if normalize(ox_line) == normalize(riv_line):
				matched = ''
				flag = ''
			else:
				matched = 'DIFF'
				flag = 'yellow'
				
			
			

			# Oxford tags
			if matched == "DIFF":
				riv_tags = list(tags)
				tags = []
				riv_words_normed = []
				for word in riv_words:
					riv_words_normed.append(norm_word(word))

				# Track which Riverside indices have been used
				used_indices = []

				for i, word in enumerate(ox_words):
					ox_clean = word.lower().strip()
					# Default behavior: match by normalized word
					normed_ox_word = norm_word(word)
					do_extra = True
					if normed_ox_word in riv_words_normed and flag!="green":
						# Find the first unused occurrence of this word
						found_index = None
						for j, riv_normed in enumerate(riv_words_normed):
							if riv_normed == normed_ox_word and j not in used_indices:
								found_index = j
								used_indices.append(j)
								break
						
						if found_index is not None:
							tags.append(riv_tags[found_index])
							do_extra=False
						#else:
							# All occurrences used, append empty
							#tags.append('')
							#flag = ""
					if do_extra == True:
						# Look through the oxford_prelim.json
						if ox_clean in oxford_prelim:
							word_tags = oxford_prelim[ox_clean]
							
							# Check if there's only one option or one option is much more likely
							if len(word_tags) == 1:
								# Only one option, use it
								tag_name = list(word_tags.keys())[0]
								tags.append('{*' + tag_name + '*}')
							else:
								# Find the most common tag and check if it's tolerance times more likely
								sorted_tags = sorted(word_tags.items(), key=lambda x: x[1], reverse=True)
								most_common_tag, most_common_count = sorted_tags[0]
								second_most_common_count = sorted_tags[1][1] if len(sorted_tags) > 1 else 0
								
								if most_common_count >= tolerance * second_most_common_count:
									# High confidence tag found
									tags.append('{*' + most_common_tag + '*}')
								else:
									found_it = False
									remaining_tags = [
									    tag for i, tag in enumerate(riv_tags)
									    if i not in used_indices
									]
									remaining_tags = [
									    tag.removeprefix("{*").removesuffix("*}")
									    for tag in remaining_tags
									]

									if "nought@adv" in remaining_tags: 
										for sorted_tag in sorted_tags:
											the_tag = sorted_tag[0]
											if the_tag == "not@adv" and not found_it:
												tags.append(the_tag)
												found_it = True

									elif "not@adv" in remaining_tags: 
										for sorted_tag in sorted_tags:
											the_tag = sorted_tag[0]
											if the_tag == "nought@adv" and not found_it:
												tags.append(the_tag)
												found_it = True
												
									for sorted_tag in sorted_tags:
										the_tag = sorted_tag[0]
										if the_tag in remaining_tags and not found_it:
											tags.append(the_tag)
											found_it = True
									
									if not found_it:	
										# No high confidence tag
										flag = ""
										tags.append('')
						else:
							# Word not found in oxford_prelim
							flag = ""
							tags.append('')
			if flag == "yellow":
				flag = "green"


						
			riv_line = ''.join(ch for ch in riv_line if ch in string.ascii_letters + ' \n')
			#ox_line = ''.join(ch for ch in ox_line if ch in string.ascii_letters + ' \n')
			#ox_words = ox_line.split()
			riv_words = riv_line.split()
			tags_riv = extract_tags(cat_line)

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
			row.extend(tag.strip("{}*") for tag in tags_riv)
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

			row = [riverside_file, oxford_file, output_csv[:-4], line_num, matched, flag, num_sybs]
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
target = 8
make_formatted('data/riverside_cats/BD_riv.cat','data/oxford_txts/BD_oxford.txt','data/csvs/BD.csv','data/oxford_cats/BD_oxford.cat')
convert_file('data/csvs/BD.csv', 'for_gui/to_do/BD_gui.csv')
make_formatted('data/riverside_cats/HF_riv.cat','data/oxford_txts/HF_oxford.txt','data/csvs/HF.csv','data/oxford_cats/HF_oxford.cat')
convert_file('data/csvs/HF.csv', 'for_gui/to_do/HF_gui.csv')

target = 10


for name in ['MLT','LGW_FPro','GP','PF','TC1','TC2','TC3','TC4','TC5','ClT','CYT','KnT','MancT','MilT','MkT','NPT','PardT','PhyT','PrT','RvT','ShipT','SNT','SqT','Thop','FranT','FriT','MerT','SumT','WBPro','WBT']:
	make_formatted('data/riverside_cats/'+name+'_riv.cat','data/oxford_txts/'+name+'_oxford.txt','data/csvs/'+name+'.csv','data/oxford_cats/'+name+'_oxford.cat')
	convert_file('data/csvs/'+name+'.csv','for_gui/to_do/'+name+'_gui.csv')
