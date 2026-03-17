#!/usr/bin/env python3
"""
Headword Combination Search Tool
Search for specific headword-tag-word combinations across all CSV files.
"""

import argparse
import pandas as pd
from headword_search import search_headword_combinations, search_from_file

def main():
    parser = argparse.ArgumentParser(description='Search for headword combinations in CSV files')
    parser.add_argument('--csv-dir', default='data/csvs', help='Directory containing CSV files')
    parser.add_argument('--output', help='Output CSV file for results')
    parser.add_argument('--search-file', help='CSV file with search terms (columns: headword, tag, word)')
    parser.add_argument('--headword', help='Headword to search for')
    parser.add_argument('--tag', help='Tag to search for')
    parser.add_argument('--word', help='Word to search for')

    args = parser.parse_args()

    if args.search_file:
        # Search from file
        results = search_from_file(args.search_file, args.csv_dir, args.output)
    elif args.headword or args.tag or args.word:
        # Single search term from command line
        search_terms = [{
            'headword': args.headword or '',
            'tag': args.tag or '',
            'word': args.word or ''
        }]
        results = search_headword_combinations(args.csv_dir, search_terms, args.output)
    else:
        print("Please provide either --search-file or at least one of --headword/--tag/--word")
        return

    if results:
        print(f"Found {len(results)} matches")
        df = pd.DataFrame(results)
        print(df[['text_type', 'filename', 'line_number', 'headword', 'tag', 'word']].to_string())
    else:
        print("No matches found")

if __name__ == "__main__":
    main()
