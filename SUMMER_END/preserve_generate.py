import csv
from scansion import *
import re
import string

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
        'comen':'come'
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

def check_naught_not_only_diff(riv_words, ox_words, riv_line, ox_line):
    riv_normalized = normalize(riv_line)
    ox_normalized = normalize(ox_line)
    if riv_normalized == ox_normalized:
        return False
    if len(riv_words) != len(ox_words):
        return False
    differences = []
    for i, (riv_w, ox_w) in enumerate(zip(riv_words, ox_words)):
        riv_clean = riv_w.lower().strip()
        ox_clean = ox_w.lower().strip()
        if normalize(riv_clean) != normalize(ox_clean):
            naught_variants = {'naught', 'nought'}
            not_variants = {'not', 'nat'}
            riv_is_naught = riv_clean in naught_variants
            riv_is_not = riv_clean in not_variants
            ox_is_naught = ox_clean in naught_variants
            ox_is_not = ox_clean in not_variants
            if (riv_is_naught and ox_is_not) or (riv_is_not and ox_is_naught):
                differences.append(i)
            else:
                return False
    return len(differences) > 0

def generate_tc5_lines(riverside_file, oxford_file, target=10):
    """Generate fresh TC5 lines using the original logic"""
    lines_data = []
    
    with open(riverside_file, encoding='utf-8') as riv:
        cat_lines = [line.rstrip('\n') for line in riv if line.strip() != '']

    with open(oxford_file, 'r', encoding='utf-8') as oxford:
        oxford_lines = [line.rstrip('\n') for line in oxford if line.strip()!='']

    for cat_line, ox_line in zip(cat_lines, oxford_lines):
        line_group = []
        
        riv_line, numbering = strip_annotations(cat_line)
        line_num = numbering
        og_riv_line = riv_line
        og_ox_line = ox_line

        # First row: original lines
        row = [og_riv_line, "||", og_ox_line]
        line_group.append(row)

        riv_line = ''.join(ch for ch in riv_line if ch in string.ascii_letters + ' \n')
        ox_line = ''.join(ch for ch in ox_line if ch in string.ascii_letters + ' \n')
        ox_words = ox_line.split()
        riv_words = riv_line.split()
        tags = extract_tags(cat_line)

        if normalize(ox_line) == normalize(riv_line):
            matched = ''
            flag = ''
        else:
            matched = 'DIFF'
            flag = ''
            if check_naught_not_only_diff(riv_words, ox_words, riv_line, ox_line):
                flag = 'green'

        # Riverside words
        stresses, num_sybs = scan(riv_line, target)
        row = [riverside_file, oxford_file, 'TC5', line_num, matched, flag, num_sybs]
        row.extend(riv_words)
        line_group.append(row)

        # Riverside stresses
        row = [riverside_file, oxford_file, 'TC5', line_num, matched, flag, num_sybs]
        row.extend(stresses)
        line_group.append(row)

        # Riverside tags
        row = [riverside_file, oxford_file, 'TC5', line_num, matched, flag, num_sybs]
        row.extend(tag.strip("{}*") for tag in tags)
        line_group.append(row)

        # Oxford words
        stresses, num_sybs = scan(ox_line, target)
        row = [riverside_file, oxford_file, 'TC5', line_num, matched, flag, num_sybs]
        row.extend(ox_words)
        line_group.append(row)

        # Oxford stresses
        row = [riverside_file, oxford_file, 'TC5', line_num, matched, flag, num_sybs]
        row.extend(stresses)
        line_group.append(row)

        # Oxford tags
        row = [riverside_file, oxford_file, 'TC5', line_num, matched, flag, num_sybs]
        flag_for_ox = "yellow"
        if matched == "DIFF":
            riv_tags = list(tags)
            tags = []
            riv_words_normed = []
            for word in riv_words:
                riv_words_normed.append(norm_word(word))

            used_indices = []
            for i, word in enumerate(ox_words):
                ox_clean = word.lower().strip()
                if flag == 'green':
                    riv_word_clean = riv_words[i].lower().strip() if i < len(riv_words) else ''
                    if riv_word_clean in {'naught', 'nought'} and ox_clean in {'not', 'nat'}:
                        tags.append('{*not@adv*}')
                        continue
                    elif riv_word_clean in {'not', 'nat'} and ox_clean in {'naught', 'nought'}:
                        tags.append('{*nought@adv*}')
                        continue
                
                normed_ox_word = norm_word(word)
                if normed_ox_word in riv_words_normed:
                    found_index = None
                    for j, riv_normed in enumerate(riv_words_normed):
                        if riv_normed == normed_ox_word and j not in used_indices:
                            found_index = j
                            used_indices.append(j)
                            break
                    if found_index is not None:
                        tags.append(riv_tags[found_index])
                    else:
                        tags.append('')
                else:
                    flag_for_ox = ""
                    tags.append('')
        if flag_for_ox == "yellow":
            flag_for_ox = "green"

        row.extend(tag.strip("{}*") for tag in tags)
        line_group.append(row)

        # Empty row
        line_group.append([''])
        
        lines_data.append(line_group)
    
    return lines_data

def merge_csv_with_green_flags(existing_csv, riverside_file, oxford_file, output_csv):
    """
    Read existing CSV, preserve green-flagged line groups, replace others with fresh TC5 data
    """
    # Read existing CSV
    with open(existing_csv, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        existing_lines = list(reader)
    
    # Generate fresh TC5 data
    fresh_data = generate_tc5_lines(riverside_file, oxford_file)
    
    # Group existing lines into line groups (8 rows per group)
    existing_groups = []
    current_group = []
    for row in existing_lines:
        current_group.append(row)
        if row == [''] or row == []:  # Empty row signals end of group
            existing_groups.append(current_group)
            current_group = []
    if current_group:  # Add last group if no trailing empty row
        existing_groups.append(current_group)
    
    # Merge: keep green-flagged groups from existing, use fresh for others
    output_lines = []
    
    for i, (existing_group, fresh_group) in enumerate(zip(existing_groups, fresh_data)):
        # Check if this group has a green flag (check row index 5, column index 5)
        has_green_flag = False
        for row in existing_group:
            if len(row) > 5 and row[5] == 'green':
                has_green_flag = True
                break
        
        if has_green_flag:
            # Preserve existing group
            output_lines.extend(existing_group)
        else:
            # Use fresh generated group
            output_lines.extend(fresh_group)
    
    # Write output
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(output_lines)
    
    print(f"Merged CSV written to {output_csv}")
    print(f"Processed {len(existing_groups)} line groups")

# Usage
if __name__ == "__main__":
    merge_csv_with_green_flags(
        existing_csv='for_gui/in_progress/TC5_gui_in_progress.csv',  # Your existing CSV with some green flags
        riverside_file='data/riverside_cats/TC5_riv.cat',
        oxford_file='data/oxford_txts/TC5_oxford.txt',
        output_csv='for_gui/in_progress/TC5_gui_merged.csv'  # Output file
    )
