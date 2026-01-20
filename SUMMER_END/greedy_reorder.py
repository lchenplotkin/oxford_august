import re
import string


# ---------- NORMALIZATION (aligned with your pipeline) ----------

def preprocess_variants(text):
	replacements = {
		'hyt': 'it', 'hit':'it',
		'ben': 'be', 'been':'be',
		'doon':'don', 'done':'don', 'do':'don',
		'foloweth': 'folweth',
		'have': 'han',
		'fro': 'from',
		'defaulte': 'defaute',
		'sorow':'sorwe', 'sorowe':'sorwe',
		'servyce': 'service', 'servise': 'service', 'servyse': 'service',
		'wonderlich': 'wonderly',
		'euery': 'every',
		'swich': 'such',
		'compaignie': 'companie', 'compaignye': 'companie',
		'eighen': 'eyen',
		'i':'ich',
		'comen':'come',
		'cruwel':'cruel', 'crewel':'cruel'
	}

	words = text.lower().split()
	out = []
	for w in words:
		w = ''.join(ch for ch in w if ch in string.ascii_lowercase)
		w = re.sub(r'(.)\1+', r'\1', w)
		out.append(replacements.get(w, w))
	return ' '.join(out)


def normalize(text):
	text = preprocess_variants(text)
	text = ''.join(ch for ch in text if ch in string.ascii_lowercase + ' ')
	text = re.sub(r'(.)\1+', r'\1', text)
	text = text.replace('y','i').replace('z','s').replace('k','c')
	text = text.replace('uw','u').replace('ow','ou').replace('ov','ou').replace('gh','w')
	text = text.replace('a','').replace('e','').replace('i','').replace('o','').replace('u','')
	return text.strip()


# ---------- CAT HANDLING ----------

def split_cat_line(line):
	"""
	Returns:
		line_number, full_payload_with_tags
	"""
	m = re.match(r'^(\S+)\s+(.*)$', line)
	if m:
		return m.group(1), m.group(2)
	return "", line


def strip_for_matching(payload):
	"""
	Remove tags for comparison only.
	"""
	text = payload.replace('{', ' ').split()
	text = text[2:]
	return " ".join(w for w in text if '}' not in w)


# ---------- CORE SWAP ALGORITHM ----------

def reorder_by_greedy_swaps(riv_cat, ox_txt, output_cat):
	# Load files
	with open(riv_cat, encoding="utf-8") as f:
		riv_lines = [l.rstrip("\n") for l in f if l.strip()]

	with open(ox_txt, encoding="utf-8") as f:
		ox_lines = [l.rstrip("\n") for l in f if l.strip()]

	# Decompose Riverside
	line_numbers = []
	payloads = []
	match_texts = []

	for line in riv_lines:
		num, payload = split_cat_line(line)
		line_numbers.append(num)
		payloads.append(payload)
		match_texts.append(normalize(strip_for_matching(payload)))

	ox_norm = [normalize(l) for l in ox_lines]

	n = min(len(payloads), len(ox_norm))

	# Greedy pass
	for i in range(n):

		if match_texts[i] == ox_norm[i]:
			continue

		# Search Riverside for a matching line
		found = None
		for j in range(len(match_texts)):
			if match_texts[j] == ox_norm[i]:
				found = j
				break

		# Swap payloads (NOT line numbers)
		if found is not None:
			payloads[i], payloads[found] = payloads[found], payloads[i]
			match_texts[i], match_texts[found] = match_texts[found], match_texts[i]
			print('swapped line', i,'with line',found)

	# Write output CAT
	with open(output_cat, "w", encoding="utf-8") as f:
		for num, payload in zip(line_numbers, payloads):
			if num:
				f.write(f"{num} {payload}\n")
			else:
				f.write(payload + "\n")


# ---------- RUN EXAMPLE ----------

if __name__ == "__main__":
	reorder_by_greedy_swaps(
		"data/riverside_cats/MkT_riv.cat",
		"data/oxford_txts/MkT_oxford.txt",
		"data/riverside_cats/MkT_riverside_reordered.cat"
	)

