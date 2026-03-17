import os
import csv
import re
import pandas as pd

ELISION_FOLLOWERS = ["have", "haven", "haveth", "havest", "had", "hadde",
                    "hadden", "his", "her", "him", "hers", "hide", "hir",
                    "hire", "hires", "hirs", "han"]

def vowel_cluster_count(w):
    return len(re.findall(r'[^aeiouy]*[aeiouy]+(?:[^aeiouy]+(?=[^aeiouy]*[aeiouy])|[^aeiouy]*)?', w)) or 1

def is_elided(word, next_word):
    if word.endswith('e') and (next_word[0] in 'aeiou' or next_word in ELISION_FOLLOWERS):
        return True
    return False

def is_weak_form(prev_tag, prev_word, tag, next_tag):
    for cause in ['demonstrative','def_art','n%gen','pron%gen','pron%pl_gen','interj','pron%fem_gen']:
        if prev_tag.startswith(cause):
            return True
    if next_tag.startswith('n#propn'):
        return True

    return False

def is_plural_form(prev_tag, tag, next_tag):
    if next_tag.startswith('n%pl') or prev_tag.startswith('n%pl'):
        return True
    return False

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

                tag = tag_part
                tag = ''.join([i for i in tag if not i.isdigit()])
                headwords.append(headword)
                tags.append(tag)

    # Handle special demonstrative cases
    for i, (word, tag) in enumerate(zip(words[:len(tags)], tags)):
        if word in ['this', 'that', 'thilke'] and 'gram_adj' in tag:
            tags[i] = 'demonstrative'

    # Ensure all lists are the same length (trim to shortest)
    if len(words)!= len(tags):
        print(oxford_text, oxford_tagging,words,tags)
        print(len(words),len(tags))
    min_len = min(len(words), len(tags), len(headwords))
    return words[:min_len], headwords[:min_len], tags[:min_len]

def search_tag_sequence_csv(df, adjectives, examples):
    """Search for tag sequences in CSV data"""
    spelled, spelled_elided, unspelled, unspelled_elided, final = 0, 0, 0, 0, 0
    unspelled_lines = []
    unspelled_elided_lines = []

    for idx, row in df.iterrows():
        if row["MATCH"] != "DIFF":
            oxford_text = row['OXFORD_TEXT']
            oxford_tagging = row['OXFORD_TAGGING']
            line_number = row['LINE_NUMBER']
            filename = row['OXFORD_FILENAME']

            words, headwords, tags = parse_tagged_text(oxford_text, oxford_tagging)

            prev_tag = 'NA'
            prev_word = 'NA'
            for j in range(len(tags)-1):
                headword = headwords[j]
                tag = tags[j]
                next_tag = tags[j+1]
                word = words[j].lower()
                next_word = words[j+1].lower()

                if tag.startswith('adj'):
                    if not is_elided(word, next_word):
                        weak = is_weak_form(prev_tag,prev_word, tag, next_tag)
                        plural = is_plural_form(prev_tag, tag, next_tag)

                        if headword not in adjectives.keys():
                            adjectives[headword] = [set(),set(), set()]

                        adj_type = 'other'
                        if weak:
                            adjectives[headword][1].add(word)
                            adj_type = 'weak'
                        elif plural:
                            adjectives[headword][2].add(word)
                            adj_type = 'plural'
                        elif tag == 'adj':
                            adjectives[headword][0].add(word)
                            adj_type = 'stem'

                        if (headword,adj_type,word) not in examples.keys():
                            examples[(headword,adj_type,word)] = []

                        examples[(headword,adj_type,word)].append(str(row["LINE_NUMBER"]) + ' ' +  row["OG_OXFORD_TEXT"])

                prev_word = word
                prev_tag = tag

    return adjectives, examples

def process_csv_directory(csv_dir):
    data = {}
    adjectives = {}
    examples = {}

    for root, dirs, files in os.walk(csv_dir):
        for file in files:
            if not file.endswith('_gui.csv'):
                continue

            csv_path = os.path.join(root, file)
            rel_path = os.path.relpath(root, csv_dir)
            file_id = os.path.join(rel_path, file) if rel_path != '.' else file

            try:
                # Read CSV file
                df = pd.read_csv(csv_path, encoding='utf-8')

                # Skip if required columns are missing
                required_columns = ['OXFORD_TAGGING', 'OXFORD_TEXT', 'LINE_NUMBER', 'OXFORD_FILENAME']
                if not all(col in df.columns for col in required_columns):
                    print(f"Skipping {file}: missing required columns")
                    continue
            except:
                print('skipping: ', file)

            adjectives, examples = search_tag_sequence_csv(df, adjectives, examples)

    return adjectives, examples


adjectives, examples = process_csv_directory('data/csvs')#key = headword, val = [stem_forms, weak_forms, plural_forms]
mono_syllabic_adjectives = {}

for adjective in adjectives.keys():
    add_it = True
    if adjectives[adjective][0] == set():
        stem_form = adjective
        if vowel_cluster_count(stem_form.rstrip('e')) >1:
            add_it = False
    else:
        for stem_form in adjectives[adjective][0]:
            if vowel_cluster_count(stem_form.rstrip('e')) >1:
                add_it = False

    if add_it:
        mono_syllabic_adjectives[adjective] = adjectives[adjective]

#print(mono_syllabic_adjectives)

for adj in mono_syllabic_adjectives.keys():
    print(adj, mono_syllabic_adjectives[adj][0])

for mono in ['fair','god','olde','wise','gret','blak','long','yong','fals','fin']:
    print(adjectives[mono])

for french in ['pure','clene','grene','swete','dere']:
    print(adjectives[french])

for example in examples[('fair','stem','faire')]:
    print(example)

for example in examples[('god','stem','goode')]:
    print(example)

for example in examples[('god','stem','goode')]:
    print(example)

for example in examples[('god','stem','gode')]:
    print(example)

for example in examples[('olde','stem','olde')]:
    print(example)

for example in examples[('wise','stem','wise')]:
    print(example)

for example in examples[('gret','stem','grete')]:
    print(example)

for example in examples[('blak','stem','blake')]:
    print(example)

for example in examples[('yong','stem','yonge')]:
    print(example)

for example in examples[('fals','stem','false')]:
    print(example)

for example in examples[('fin','stem','fine')]:
    print(example)

#for example in examples[('gret'
#print(examples[('gret','stem','grete')])
#print(examples[('smal','stem','smale')])


'''
[{'faire', 'feir', 'fair'}, {'faire', 'fairest', 'fair'}, {'faire'}]
[{'good', 'goode', 'gode'}, {'good', 'goode', 'gode'}, {'good', 'goode', 'gode'}]
[{'olde', 'old'}, {'olde', 'old', 'eldest'}, {'olde'}]
[{'wise', 'wis'}, {'wisest', 'wise'}, {'wiser', 'wise'}]
[{'great', 'grete', 'gret', 'greet'}, {'grettest', 'grete', 'gretest', 'gret'}, {'grete'}]
[{'blakke', 'blake', 'blak'}, {'blake'}, {'blake'}]
[set(), {'longe', 'lange'}, set()]
[{'yong', 'yonge'}, {'yonge', 'yongest'}, {'yonge'}]
[{'false', 'fals'}, {'false', 'fals'}, {'false'}]
[{'fin', 'fine'}, {'finest', 'fin'}, {'fine'}]
[{'pure'}, {'pure'}, {'pure'}]
[{'clene'}, set(), set()]
[{'grene'}, {'grene'}, {'grene'}]
[{'swete', 'sweete'}, {'sweet', 'swete', 'sweete'}, set()]
[set(), {'deere', 'dere'}, set()]

'''
