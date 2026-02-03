import re
import string
from collections import defaultdict

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


# ---------- CAT UTILITIES ----------

def strip_annotations_for_matching(line):
	"""
	Removes numbering and tags.
	Used ONLY for comparison.
	"""
	marked = re.sub(r'\b(\w*\d|\d+[a-zA-Z])\b\s+', r'\1||| ', line)
	parts = marked.split('|||')

	text = parts[1] if len(parts) > 1 else parts[0]
	text = text.replace('{', ' ').split()
	return " ".join(w for w in text if '}' not in w).strip()


def extract_line_number(line):
	m = re.match(r'^(\S+)\s+', line)
	return m.group(1) if m else ""


def remove_line_number(line):
	m = re.match(r'^(\S+)\s+(.*)$', line)
	return m.group(2) if m else line


# ---------- ALIGNMENT LOGIC ----------

def build_oxford_index(ox_norm):
	index = defaultdict(list)
	for j, line in enumerate(ox_norm):
		if line:
			index[line].append(j)
	return index


def assign_best_anchor(riv_norm, ox_index):
	anchors = []
	for line in riv_norm:
		if not line or line not in ox_index:
			anchors.append(None)
		else:
			hits = sorted(ox_index[line])
			anchors.append(hits[len(hits)//2])  # median hit
	return anchors


def smooth_anchors(anchors):
	out = anchors[:]

	last = None
	for i in range(len(out)):
		if out[i] is not None:
			last = out[i]
		elif last is not None:
			out[i] = last

	last = None
	for i in range(len(out) - 1, -1, -1):
		if anchors[i] is not None:
			last = anchors[i]
		elif last is not None:
			out[i] = last

	return out


# ---------- MAIN REORDERING FUNCTION ----------

def reorder_riverside_cat(riv_cat, ox_txt, output_cat):
	# Load Riverside CAT
	with open(riv_cat, encoding="utf-8") as f:
		riv_lines = [l.rstrip("\n") for l in f if l.strip()]

	# Load Oxford TXT
	with open(ox_txt, encoding="utf-8") as f:
		ox_lines = [l.rstrip("\n") for l in f if l.strip()]

	# Decompose Riverside lines
	line_numbers = []
	full_payloads = []
	match_texts = []

	for line in riv_lines:
		line_numbers.append(extract_line_number(line))
		payload = remove_line_number(line)
		full_payloads.append(payload)
		match_texts.append(strip_annotations_for_matching(line))

	# Normalize for matching
	riv_norm = [normalize(t) for t in match_texts]
	ox_norm  = [normalize(l) for l in ox_lines]

	# Anchor Riverside lines to Oxford
	ox_index = build_oxford_index(ox_norm)
	anchors = assign_best_anchor(riv_norm, ox_index)
	anchors = smooth_anchors(anchors)

	# Compute reorder permutation
	order = sorted(
		range(len(full_payloads)),
		key=lambda i: (anchors[i] is None, anchors[i] if anchors[i] is not None else i)
	)

	reordered_payloads = [full_payloads[i] for i in order]

	# Reassemble CAT: numbering fixed, payloads moved
	with open(output_cat, "w", encoding="utf-8") as f:
		for num, payload in zip(line_numbers, reordered_payloads):
			if num:
				f.write(f"{num} {payload}\n")
			else:
				f.write(payload + "\n")


if __name__ == "__main__":
	reorder_riverside_cat(
		"data/riverside_cats/MkT_riv.cat",
		"data/oxford_txts/MkT_oxford.txt",
		"data/riverside_cats/MkT_riverside_reordered.cat"
	)

