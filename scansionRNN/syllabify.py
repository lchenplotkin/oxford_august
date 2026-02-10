#!/usr/bin/env python3
"""
Middle English Syllabification Script
Analyzes OXFORD_TEXT column from CSV files to find words with consecutive vowels.
"""

import csv
import os
import re
from pathlib import Path


def collapse_doubles(text):
    """Collapse doubled characters into single characters."""
    result = []
    prev = None
    for char in text:
        if char != prev:
            result.append(char)
        prev = char
    return ''.join(result)


def is_vowel(char):
    """Check if character is a vowel."""
    return char.lower() in 'aeou'


def is_semivowel(char):
    """Check if character is a semivowel (i or y)."""
    return char.lower() in 'iy'


def is_consonant(char):
    """Check if character is a consonant."""
    return char.isalpha() and not is_vowel(char) and not is_semivowel(char)


def handle_qu(text):
    """Mark 'u' after 'q' as consonant by replacing with placeholder."""
    # Replace 'qu' with 'qC' where C is a marker for consonant-u
    result = re.sub(r'qu', 'qC', text, flags=re.IGNORECASE)
    return result


def is_latin_proper_noun(word):
    """Check if word is a Latin proper noun like 'theseus'."""
    # Simple heuristic: starts with capital letter
    return word and word[0].isupper()


def get_symbol_sequence(word):
    """
    Convert a word into a sequence of V (vowel) and C (consonant) symbols.
    
    Returns: string of V and C characters representing the syllabic structure
    """
    # First, handle doubled characters
    word = word.replace('y','i')
    word = collapse_doubles(word)
    
    # Handle qu -> qC
    word = handle_qu(word)
    
    symbols = ''
    chars = list(word)
    
    for i, char in enumerate(chars):
        if not char.isalpha():
            symbols +=char  # Keep non-alphabetic characters as-is
            continue
            
        # Handle regular vowels (a, e, o)
        if char.lower() in 'aeo':
            symbols += 'V'
        
        # Handle 'u' as a vowel (unless already handled by qu rule)
        elif char.lower() == 'u':
            # Check if this is part of a vowel cluster with preceding vowels
            # and should be grouped (collapsed into previous V)
            if i > 0 and symbols and symbols[i-1] == 'V':
                # Don't add new V, it groups with previous
                # But we need to mark this somehow for the u-group exception
                # For now, use a special marker
                symbols += 'U'  # Special marker for u-group
            else:
                symbols += 'V'
        
        # Handle the placeholder C from qu
        elif char == 'C':
            symbols += 'C'
        
        # Handle semivowels (i, y)
        elif is_semivowel(char):
            # Rule 1: Cy -> CV (consonant before semivowel)
            if i > 0 and is_consonant(chars[i-1]):
                symbols += 'V'
            
            # Rule 2: Vy -> VC (vowel before semivowel)
            elif i > 0 and (is_vowel(chars[i-1]) or chars[i-1].lower() == 'u'):
                symbols += 'C'
            
            # Rule 3: SPACE yV -> CV (space before, vowel after)
            elif i == 0:  # At start of word (after space conceptually)
                if i + 1 < len(chars) and (is_vowel(chars[i+1]) or chars[i+1].lower() == 'u'):
                    symbols += 'C'
                # Rule 4: SPACE yC -> VC (space before, consonant after)
                elif i + 1 < len(chars) and is_consonant(chars[i+1]):
                    symbols += 'V'
                else:
                    # At end of word after space
                    symbols += 'V'
            else:
                symbols += 'V'  # Default to vowel
        
        # Regular consonants
        else:
            symbols += 'C'
    
    return symbols


def has_consecutive_vowels_excluding_u_group(symbol_seq):
    """
    Check if there are consecutive Vs, excluding cases where:
    - There are only two Vs in a row AND
    - The second is a U (u-group marker)
    
    Returns: True if pattern found, False otherwise
    """
    # Look for VV+ patterns
    i = 0
    while i < len(symbol_seq):
        if symbol_seq[i] == 'V':
            # Count consecutive Vs and Us
            j = i
            v_count = 0
            has_u_group = False
            
            while j < len(symbol_seq) and symbol_seq[j] in 'VU':
                if symbol_seq[j] == 'V':
                    v_count += 1
                elif symbol_seq[j] == 'U':
                    has_u_group = True
                j += 1
            
            total_vowels = j - i
            
            # Check if this is a consecutive vowel pattern we care about
            if total_vowels >= 2:
                # Exclude the case: exactly 2 vowel positions and second is U
                if total_vowels == 2 and has_u_group and symbol_seq[i+1] == 'U':
                    # Skip this pattern
                    pass
                else:
                    return True
            
            i = j
        else:
            i += 1
    
    return False


def process_csv_files(directory):
    """
    Process all CSV files in the directory.
    Extract OXFORD_TEXT column and find lines with consecutive vowels.
    """
    directory_path = Path(directory)
    
    if not directory_path.exists():
        print(f"Error: Directory '{directory}' does not exist.")
        return
    
    csv_files = list(directory_path.glob('*.csv'))
    
    if not csv_files:
        print(f"No CSV files found in '{directory}'")
        return
    
    print(f"Processing {len(csv_files)} CSV file(s)...\n")
    
    for csv_file in csv_files:
        print(f"\n{'='*80}")
        print(f"File: {csv_file.name}")
        print(f"{'='*80}\n")
        
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                if 'OXFORD_TEXT' not in reader.fieldnames:
                    print(f"Warning: 'OXFORD_TEXT' column not found in {csv_file.name}")
                    print(f"Available columns: {reader.fieldnames}\n")
                    continue
                
                line_count = 0
                match_count = 0
                
                for row in reader:
                    line_count += 1
                    text = row.get('OXFORD_TEXT', '').strip().lower()
                    
                    if not text:
                        continue
                    
                    # Split into words
                    words = text.split()
                    
                    # Check each word
                    matching_words = []
                    for word in words[:-1]:
                        # Remove punctuation for analysis
                        clean_word = re.sub(r'[^\w]', '', word)
                        if not clean_word:
                            continue
                        
                        symbols = get_symbol_sequence(clean_word)
                        
                        if has_consecutive_vowels_excluding_u_group(symbols):
                            matching_words.append((word, clean_word, symbols))
                    
                    # Print line if it has matching words
                    if matching_words:
                        match_count += 1
                        print(f"Line {line_count}: {text}")
                        for orig_word, clean_word, symbols in matching_words:
                            print(f"  → {clean_word}: {symbols}")
                        print()
                
                print(f"\nSummary for {csv_file.name}:")
                print(f"  Total lines: {line_count}")
                print(f"  Lines with consecutive vowels: {match_count}")
                
        except Exception as e:
            print(f"Error processing {csv_file.name}: {e}")


def main():
    import sys
    
    if len(sys.argv) > 1:
        directory = sys.argv[1]
    else:
        directory = input("Enter directory path containing CSV files: ").strip()
    
    if not directory:
        print("No directory specified. Using current directory.")
        directory = "."
    
    process_csv_files(directory)


if __name__ == "__main__":
    main()
