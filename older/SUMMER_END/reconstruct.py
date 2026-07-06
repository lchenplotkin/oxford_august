#!/usr/bin/env python3
"""
Program to add a headword column to the verb_forms_gold.csv file.
Extracts headwords from the actual tagged data in the CSV files.
"""

import pandas as pd
import os
import re
from collections import defaultdict

def parse_tagged_text(text, tagging):
    """Extract headwords from tagging data"""
    if pd.isna(text) or pd.isna(tagging) or text == '' or tagging == '':
        return []

    # Parse tags from tagging
    tag_tokens = tagging.strip().split()
    headwords = []

    for token in tag_tokens:
        if '@' in token and token not in ["--@dash", ".@ellipsis"]:
            parts = token.split('@')
            headword = parts[0].lower()
            tag_part = parts[1] if len(parts) > 1 else ''
            
            # Only collect verb-related headwords
            if tag_part.startswith('v%'):
                headwords.append(headword)

    return headwords

def build_infinitive_to_headword_mapping(csv_dir):
    """
    Build a mapping from infinitives to their headwords by scanning all CSV files.
    """
    infinitive_to_headword = {}
    headword_counts = defaultdict(lambda: defaultdict(int))
    
    print("Scanning CSV files to extract headword mappings...")
    
    file_count = 0
    for root, dirs, files in os.walk(csv_dir):
        for file in files:
            if not file.endswith('_gui.csv'):
                continue

            csv_path = os.path.join(root, file)
            file_count += 1
            
            if file_count % 50 == 0:
                print(f"  Processed {file_count} files...")

            try:
                # Read CSV file
                df = pd.read_csv(csv_path, encoding='utf-8')

                # Process Oxford text if columns exist
                oxford_columns = ['OXFORD_TAGGING', 'OXFORD_TEXT']
                if all(col in df.columns for col in oxford_columns):
                    for idx, row in df.iterrows():
                        if row.get("MATCH") != "DIFF":
                            text = row['OXFORD_TEXT']
                            tagging = row['OXFORD_TAGGING']
                            headwords = parse_tagged_text(text, tagging)
                            
                            # Look for infinitive tags
                            tag_tokens = str(tagging).strip().split() if not pd.isna(tagging) else []
                            for token in tag_tokens:
                                if '@v%inf' in token:
                                    parts = token.split('@')
                                    if len(parts) >= 2:
                                        headword = parts[0].lower()
                                        infinitive = headword  # For infinitives, headword = infinitive
                                        headword_counts[infinitive][headword] += 1
                
                # Process Riverside text if columns exist
                riverside_columns = ['RIVERSIDE_TAGGING', 'RIVERSIDE_TEXT']
                if all(col in df.columns for col in riverside_columns):
                    for idx, row in df.iterrows():
                        if row.get("MATCH") != "DIFF":
                            text = row['RIVERSIDE_TEXT']
                            tagging = row['RIVERSIDE_TAGGING']
                            headwords = parse_tagged_text(text, tagging)
                            
                            # Look for infinitive tags
                            tag_tokens = str(tagging).strip().split() if not pd.isna(tagging) else []
                            for token in tag_tokens:
                                if '@v%inf' in token:
                                    parts = token.split('@')
                                    if len(parts) >= 2:
                                        headword = parts[0].lower()
                                        infinitive = headword  # For infinitives, headword = infinitive
                                        headword_counts[infinitive][headword] += 1

            except Exception as e:
                print(f"Error processing {file}: {e}")
                continue

    print(f"Processed {file_count} files")
    
    # Convert to final mapping (choose most common headword for each infinitive)
    for infinitive, headword_dict in headword_counts.items():
        if headword_dict:
            # Get the most common headword for this infinitive
            most_common_headword = max(headword_dict.items(), key=lambda x: x[1])[0]
            infinitive_to_headword[infinitive] = most_common_headword
    
    print(f"Found headword mappings for {len(infinitive_to_headword)} infinitives")
    return infinitive_to_headword

def extract_headword_from_forms(infinitive, preterite, past_participle):
    """
    Extract headword from verb forms if not found in CSV data.
    This is a fallback method.
    """
    # Try to extract from infinitive first
    if infinitive and not pd.isna(infinitive):
        infinitive = str(infinitive).strip().lower()
        if infinitive.endswith('en'):
            headword = infinitive[:-2]
            if len(headword) >= 2:
                return headword
        elif infinitive.endswith('e'):
            headword = infinitive[:-1]
            if len(headword) >= 2:
                return headword
        return infinitive
    
    # If no infinitive, try to extract from other forms
    for form_str in [preterite, past_participle]:
        if form_str and not pd.isna(form_str):
            # Extract first form from the string (remove frequency counts)
            form_str = str(form_str).strip()
            if form_str:
                # Extract first form before comma or parenthesis
                first_form = re.split(r'[,(]', form_str)[0].strip()
                if first_form:
                    return first_form
    
    return ""

def add_headword_column_to_csv(input_file, csv_dir, output_file=None):
    """
    Add headword column to the verb forms CSV file.
    
    Args:
        input_file (str): Path to the input CSV file
        csv_dir (str): Path to the directory containing the tagged CSV files
        output_file (str): Path to the output CSV file (if None, overwrites input)
    """
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        return False
    
    # Check if CSV directory exists
    if not os.path.exists(csv_dir):
        print(f"Error: CSV directory '{csv_dir}' not found.")
        return False
    
    try:
        # Build headword mapping from CSV files
        infinitive_to_headword = build_infinitive_to_headword_mapping(csv_dir)
        
        # Read the verb forms CSV file
        df = pd.read_csv(input_file, encoding='utf-8')
        
        print(f"\nReading {input_file}...")
        print(f"Found {len(df)} rows")
        
        # Check required columns
        required_columns = ['infinitive']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Error: Missing required columns: {missing_columns}")
            print(f"Available columns: {list(df.columns)}")
            return False
        
        # Extract headwords
        print("Extracting headwords...")
        headwords = []
        found_in_csv = 0
        derived_from_forms = 0
        
        for idx, row in df.iterrows():
            infinitive = row['infinitive']
            
            # First try to find in the CSV mapping
            if infinitive in infinitive_to_headword:
                headword = infinitive_to_headword[infinitive]
                found_in_csv += 1
            else:
                # Fallback: derive from verb forms
                preterite = row.get('preterite', '')
                past_participle = row.get('past_participle', '')
                headword = extract_headword_from_forms(infinitive, preterite, past_participle)
                derived_from_forms += 1
            
            headwords.append(headword)
        
        # Add headword column
        df['headword'] = headwords
        
        # Reorder columns to put headword after infinitive
        columns = list(df.columns)
        headword_index = columns.index('headword')
        infinitive_index = columns.index('infinitive')
        
        # Remove headword from its current position
        columns.pop(headword_index)
        # Insert it after infinitive
        columns.insert(infinitive_index + 1, 'headword')
        
        # Reorder dataframe
        df = df[columns]
        
        # Set output file
        if output_file is None:
            output_file = input_file
        
        # Write the updated CSV
        df.to_csv(output_file, index=False, encoding='utf-8')
        
        print(f"\n✓ Successfully added headword column!")
        print(f"✓ Output saved to: {output_file}")
        print(f"✓ Updated {len(df)} rows")
        print(f"✓ Headwords found in CSV data: {found_in_csv}")
        print(f"✓ Headwords derived from forms: {derived_from_forms}")
        
        # Show some examples
        print(f"\nSample headword mappings:")
        print("-" * 50)
        for i, row in df.head(15).iterrows():
            source = "CSV data" if row['infinitive'] in infinitive_to_headword else "derived"
            print(f"{row['infinitive']} → {row['headword']} ({source})")
        
        if len(df) > 15:
            print("...")
        
        return True
        
    except Exception as e:
        print(f"Error processing file: {e}")
        return False

def preview_csv_headwords(csv_dir, limit=20):
    """
    Preview headwords found in CSV files for verification.
    """
    infinitive_to_headword = build_infinitive_to_headword_mapping(csv_dir)
    
    print(f"\nFound {len(infinitive_to_headword)} infinitive-headword mappings in CSV files:")
    print("-" * 60)
    
    count = 0
    for infinitive, headword in sorted(infinitive_to_headword.items()):
        if count >= limit:
            print("...")
            break
        print(f"{infinitive} → {headword}")
        count += 1
    
    return infinitive_to_headword

if __name__ == "__main__":
    input_file = "verb_forms_gold.csv"
    csv_dir = "data/csvs"
    
    print("Headword Column Addition Tool")
    print("=" * 40)
    
    # Check if files exist
    if not os.path.exists(input_file):
        print(f"Error: '{input_file}' not found in current directory.")
        exit(1)
    
    if not os.path.exists(csv_dir):
        print(f"Error: CSV directory '{csv_dir}' not found.")
        exit(1)
    
    print(f"Input file: {input_file}")
    print(f"CSV directory: {csv_dir}")
    
    # Ask user what they want to do
    print("\nOptions:")
    print("1. Preview headwords from CSV files (first 20)")
    print("2. Add headword column to verb_forms_gold.csv (overwrite)")
    print("3. Add headword column and save to new file")
    
    choice = input("\nEnter your choice (1, 2, or 3): ").strip()
    
    if choice == "1":
        preview_csv_headwords(csv_dir)
    elif choice == "2":
        # Overwrite original file
        if add_headword_column_to_csv(input_file, csv_dir):
            print(f"\n🎉 Successfully updated {input_file}")
        else:
            print(f"\n❌ Failed to update {input_file}")
    elif choice == "3":
        # Save to new file
        output_file = input("Enter output filename (e.g., 'verb_forms_gold_with_headwords.csv'): ").strip()
        if not output_file:
            output_file = "verb_forms_gold_with_headwords.csv"
        
        if add_headword_column_to_csv(input_file, csv_dir, output_file):
            print(f"\n🎉 Successfully created {output_file}")
        else:
            print(f"\n❌ Failed to create {output_file}")
    else:
        print("Invalid choice. Please run the script again and enter 1, 2, or 3.")
