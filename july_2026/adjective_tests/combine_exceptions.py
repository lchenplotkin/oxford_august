"""Combine exception instances from scan_adjs.py (scansion-aware) and 2adjs.py
(spelling-only) into a single CSV, flagging which script(s) treat each
instance as a declension exception.
"""
import csv
import os

SCAN_DIR = 'scan_adjective_analysis_output'
SPELL_DIR = 'adjective_analysis_output'
OUTPUT_PATH = 'combined_exceptions.csv'
SCANSION_LOOKUP_FILE = 'all_instances_scansion.csv'

EXCEPTION_FILES = {
	'weak_no_e': 'weak_no_e_all_instances.csv',
	'plural_no_e': 'plural_no_e_all_instances.csv',
	'strong_with_e': 'strong_with_e_all_instances.csv',
}


def load_exceptions(output_dir):
	"""Load {key: (exception_type, row)} for every exception instance in a directory."""
	instances = {}
	for exception_type, filename in EXCEPTION_FILES.items():
		path = os.path.join(output_dir, filename)
		if not os.path.exists(path):
			continue
		with open(path, 'r', newline='', encoding='utf-8') as csvfile:
			for row in csv.DictReader(csvfile):
				key = (row['headword'], row['word'], row['line_number'], row['filename'])
				instances[key] = (exception_type, row)
	return instances


def load_scansion_lookup(output_dir):
	"""Load {key: (word_scansion, oxford_scansion)} for every instance scan_adjs.py processed."""
	lookup = {}
	path = os.path.join(output_dir, SCANSION_LOOKUP_FILE)
	if not os.path.exists(path):
		return lookup
	with open(path, 'r', newline='', encoding='utf-8') as csvfile:
		for row in csv.DictReader(csvfile):
			key = (row['headword'], row['word'], row['line_number'], row['filename'])
			lookup[key] = (row['word_scansion'], row['oxford_scansion'])
	return lookup


def main():
	scan_instances = load_exceptions(SCAN_DIR)
	spell_instances = load_exceptions(SPELL_DIR)
	scansion_lookup = load_scansion_lookup(SCAN_DIR)

	all_keys = set(scan_instances) | set(spell_instances)

	combined_rows = []
	for key in all_keys:
		headword, word, line_number, filename = key
		scan_type, scan_row = scan_instances.get(key, (None, None))
		spell_type, spell_row = spell_instances.get(key, (None, None))

		# Prefer whichever source has the row, for shared fields
		base_row = scan_row or spell_row
		exception_type = scan_type or spell_type

		word_scansion, oxford_scansion = scansion_lookup.get(key, ('', ''))

		combined_rows.append({
			'headword': headword,
			'word': word,
			'line_number': line_number,
			'filename': filename,
			'context': base_row['context'],
			'is_elided': base_row['is_elided'],
			'is_final': base_row['is_final'],
			'exception_type': exception_type,
			'word_scansion': word_scansion,
			'oxford_scansion': oxford_scansion,
			'SCAN_EXCEPTION': 1 if key in scan_instances else 0,
			'SPELL_EXCEPTION': 1 if key in spell_instances else 0,
			'ELIDES': '',
			'REAL_SCAN': '',
			'REAL_EXCEPT': '',
			'EXCEPT_FLAG': '',
			'NOTES': '',
		})

	combined_rows.sort(key=lambda r: (r['filename'], int(r['line_number'].split()[-1]) if r['line_number'].split()[-1].isdigit() else 0, r['headword']))

	fieldnames = [
		'headword', 'word', 'line_number', 'filename', 'context',
		'is_elided', 'is_final', 'exception_type',
		'word_scansion', 'oxford_scansion',
		'SCAN_EXCEPTION', 'SPELL_EXCEPTION',
		'ELIDES', 'REAL_SCAN', 'REAL_EXCEPT', 'EXCEPT_FLAG', 'NOTES',
	]
	with open(OUTPUT_PATH, 'w', newline='', encoding='utf-8') as csvfile:
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
		writer.writeheader()
		for row in combined_rows:
			writer.writerow(row)

	scan_only = sum(1 for r in combined_rows if r['SCAN_EXCEPTION'] and not r['SPELL_EXCEPTION'])
	spell_only = sum(1 for r in combined_rows if r['SPELL_EXCEPTION'] and not r['SCAN_EXCEPTION'])
	both = sum(1 for r in combined_rows if r['SCAN_EXCEPTION'] and r['SPELL_EXCEPTION'])

	print(f"Total combined exception instances: {len(combined_rows)}")
	print(f"  Both scan and spell: {both}")
	print(f"  Scan only: {scan_only}")
	print(f"  Spell only: {spell_only}")
	print(f"Saved to: {OUTPUT_PATH}")


if __name__ == '__main__':
	main()
