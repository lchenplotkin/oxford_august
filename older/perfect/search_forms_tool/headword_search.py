import os
import re
import csv
import pandas as pd
from collections import defaultdict

def search_headword_combinations(csv_dir, search_terms, output_file=None):
    """
    Search for specific headword-tag-word combinations across all CSV files.

    Args:
        csv_dir (str): Directory containing CSV files
        search_terms (list): List of dictionaries with search criteria:
            [{'headword': 'been', 'tag': 'v%inf', 'word': 'be'}, ...]
        output_file (str): Optional output CSV file path

    Returns:
        list: List of matching records
    """

    def parse_tagged_text(text, tagging):
        """Extract words from text and tags from tagging"""
        if pd.isna(text) or pd.isna(tagging) or text == '' or tagging == '':
            return [], [], []

        words = re.sub(r'[.,!?°¶]', '', text.lower()).strip().split()
        tag_tokens = tagging.strip().split()
        tags = []
        headwords = []

        for token in tag_tokens:
            if '@' in token:
                parts = token.split('@')
                headword = parts[0].lower()
                tag = parts[1] if len(parts) > 1 else ''
                tag = ''.join([i for i in tag if not i.isdigit()])
                headwords.append(headword)
                tags.append(tag)

        min_len = min(len(words), len(tags), len(headwords))
        return words[:min_len], headwords[:min_len], tags[:min_len]

    def clean_tag(tag):
        """Remove digits before % in a tag"""
        return re.sub(r'\d+(?=%)', '', tag)

    matches = []
    file_count = 0
    match_count = 0

    print(f"Searching for {len(search_terms)} combinations in {csv_dir}...")

    for root, dirs, files in os.walk(csv_dir):
        for file in files:
            if not file.endswith('_gui.csv'):
                continue

            file_path = os.path.join(root, file)
            file_count += 1

            try:
                df = pd.read_csv(file_path, encoding='utf-8')

                # Check which text types are available in this file
                text_types = []
                for text_type in ['OXFORD', 'RIVERSIDE']:
                    if f'{text_type}_TAGGING' in df.columns and f'{text_type}_TEXT' in df.columns:
                        text_types.append(text_type)

                if not text_types:
                    continue

                for idx, row in df.iterrows():
                    for text_type in text_types:
                        text_col = f'{text_type}_TEXT'
                        tagging_col = f'{text_type}_TAGGING'
                        filename_col = f'{text_type}_FILENAME'

                        if pd.isna(row.get(text_col)) or pd.isna(row.get(tagging_col)):
                            continue

                        text = row[text_col]
                        tagging = row[tagging_col]
                        filename = row.get(filename_col, 'unknown')
                        line_number = row.get('LINE_NUMBER', 'unknown')

                        words, headwords, tags = parse_tagged_text(text, tagging)

                        for j in range(len(words)):
                            if j >= len(headwords) or j >= len(tags):
                                continue

                            current_word = words[j].lower()
                            current_headword = headwords[j].lower()
                            current_tag = clean_tag(tags[j])

                            # Check against all search terms
                            for search_term in search_terms:
                                match = True

                                # Check headword (case-insensitive)
                                if 'headword' in search_term and search_term['headword']:
                                    if search_term['headword'].lower() != current_headword:
                                        match = False

                                # Check tag (case-sensitive for tags)
                                if match and 'tag' in search_term and search_term['tag']:
                                    if search_term['tag'] != current_tag:
                                        match = False

                                # Check word (case-insensitive)
                                if match and 'word' in search_term and search_term['word']:
                                    if search_term['word'].lower() != current_word:
                                        match = False

                                if match:
                                    match_count += 1
                                    match_record = {
                                        'file': file,
                                        'file_path': file_path,
                                        'text_type': text_type,
                                        'line_number': line_number,
                                        'filename': filename,
                                        'headword': current_headword,
                                        'tag': current_tag,
                                        'word': current_word,
                                        'context': text,
                                        'full_tagging': tagging
                                    }
                                    matches.append(match_record)

            except Exception as e:
                print(f"Error processing {file}: {e}")
                continue

    print(f"Processed {file_count} files, found {match_count} matches")

    # Write results to CSV if output file specified
    if output_file and matches:
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['file', 'file_path', 'text_type', 'line_number', 'filename',
                         'headword', 'tag', 'word', 'context', 'full_tagging']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for match in matches:
                writer.writerow(match)
        print(f"Results written to {output_file}")

    return matches

def search_interactive():
    """Interactive search tool"""
    print("=== headword Combination Search Tool ===")
    print("Search for specific headword-tag-word combinations across all CSV files")
    print("Enter search terms (press Enter with empty field to skip that criteria)")
    print("-" * 50)

    search_terms = []

    while True:
        print(f"\nSearch term #{len(search_terms) + 1}:")
        headword = input("Headword (e.g., 'been'): ").strip()
        tag = input("Tag (e.g., 'v%inf'): ").strip()
        word = input("Word (e.g., 'be'): ").strip()

        # At least one criteria must be provided
        if not any([headword, tag, word]):
            print("At least one search criteria must be provided!")
            continue

        search_terms.append({
            'headword': headword if headword else None,
            'tag': tag if tag else None,
            'word': word if word else None
        })

        more = input("Add another search term? (y/n): ").strip().lower()
        if more != 'y':
            break

    csv_dir = input("\nEnter CSV directory path (default: 'data/csvs'): ").strip()
    if not csv_dir:
        csv_dir = 'data/csvs'

    output_file = input("Output CSV file (optional, press Enter to skip): ").strip()
    if not output_file:
        output_file = None

    print(f"\nSearching for {len(search_terms)} combinations...")

    results = search_headword_combinations(csv_dir, search_terms, output_file)

    if results:
        print(f"\nFound {len(results)} matches:")
        print("-" * 80)
        for i, result in enumerate(results[:10], 1):  # Show first 10 results
            print(f"{i}. {result['text_type']} - {result['filename']}:{result['line_number']}")
            print(f"   Headword: {result['headword']}, Tag: {result['tag']}, Word: {result['word']}")
            print(f"   Context: {result['context'][:100]}...")
            print()

        if len(results) > 10:
            print(f"... and {len(results) - 10} more matches")
    else:
        print("No matches found.")

def search_from_file(search_file, csv_dir, output_file=None):
    """
    Search using terms from a CSV file.

    Args:
        search_file (str): CSV file with search terms (columns: headword, tag, word)
        csv_dir (str): Directory containing CSV files to search
        output_file (str): Optional output CSV file path
    """
    try:
        df = pd.read_csv(search_file)
        search_terms = []

        for _, row in df.iterrows():
            search_terms.append({
                'headword': str(row.get('headword', '')).strip(),
                'tag': str(row.get('tag', '')).strip(),
                'word': str(row.get('word', '')).strip()
            })

        print(f"Loaded {len(search_terms)} search terms from {search_file}")
        return search_headword_combinations(csv_dir, search_terms, output_file)

    except Exception as e:
        print(f"Error reading search file {search_file}: {e}")
        return []

# Example usage functions
def example_searches():
    """Run some example searches"""
    examples = [
        # Common headword forms to search for
        {'headword': 'been', 'tag': 'v%inf', 'word': 'be'},
        {'headword': 'been', 'tag': 'v%ppl', 'word': 'been'},
        {'headword': 'have', 'tag': 'v%pr_3', 'word': 'hath'},
        {'headword': 'have', 'tag': 'v%pr_pl', 'word': 'han'},
        {'headword': 'been', 'tag': '', 'word': ''},  # All forms of 'been'
        {'headword': '', 'tag': 'v%inf', 'word': ''},  # All infinitives
    ]

    print("Running example searches...")
    results = search_headword_combinations('data/csvs', examples, 'example_search_results.csv')
    return results

if __name__ == "__main__":
    # You can use this tool in several ways:

    # 1. Interactive search
    search_interactive()

    # 2. Programmatic search with specific terms
    # search_terms = [
    #     {'headword': 'been', 'tag': 'v%inf', 'word': 'be'},
    #     {'headword': 'have', 'tag': 'v%pr_3', 'word': 'hath'}
    # ]
    # results = search_headword_combinations('data/csvs', search_terms, 'my_search_results.csv')

    # 3. Search from CSV file
    # results = search_from_file('search_terms.csv', 'data/csvs', 'search_results.csv')

    # 4. Run example searches
    # example_searches()
