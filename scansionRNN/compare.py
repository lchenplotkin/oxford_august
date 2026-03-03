#!/usr/bin/env python3
"""
Compare computer-generated scansion with vetted scansion from CSV.

Usage: python compare_scansion.py vetted_scansion.csv output_comparison.csv
"""

import sys
import csv
import re
from scansion_final import scan_line, format_scansion, minimal_clean

def parse_vetted_csv(filename):
    """
    Parse the vetted scansion CSV file.
    Returns list of (line_text, vetted_scansion) tuples.
    """
    lines_data = []
    
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        rows = list(reader)
        
        # Process pairs of rows (text row, scansion row)
        i = 0
        while i < len(rows):
            if i + 1 >= len(rows):
                break
            
            text_row = rows[i]
            scansion_row = rows[i + 1]
            print(scansion_row)
            
            # Skip empty rows
            if not any(text_row) or not any(scansion_row):
                i += 1
                continue
            
            # Reconstruct the line text from words
            words = [w.strip().strip(',').strip('"').strip("'") for w in text_row if w.strip()]
            line_text = ' '.join(words)
            
            # Reconstruct the scansion pattern
            scansion_parts = [s.strip().replace(' ','') for s in scansion_row if s.strip()]
            vetted_scansion = ' '.join(scansion_parts)
            print(vetted_scansion)
            
            lines_data.append((line_text, vetted_scansion))
            
            i += 2  # Move to next pair
    
    return lines_data

def remove_x(scansion):
    """Remove 'x' characters from scansion pattern."""
    return re.sub(r'x', '', scansion)

def normalize_scansion(scansion):
    """
    Normalize scansion for comparison:
    - Remove 'x' (silent markers)
    - Remove spaces
    - Convert to lowercase for consistency
    """
    normalized = remove_x(scansion)
    normalized = normalized.replace(' ', '')
    normalized = normalized.lower()
    return normalized

def compare_scansions(vetted_file, output_file):
    """
    Compare vetted scansion with computer-generated scansion.
    Write results to CSV.
    """
    # Parse vetted scansion
    vetted_data = parse_vetted_csv(vetted_file)
    
    # Prepare output data
    results = []
    
    i = 0
    for line_text, vetted_scansion in vetted_data:
        i+=1
        print('doing line ', i)
        # Generate computer scansion
        result = scan_line(line_text)
        
        if result:
            word_analyses, penalty = result
            formatted = format_scansion(word_analyses)
            print(formatted['marked_words'])
            computer_scansion = formatted['stress_patterns']
            syllable_count = formatted['syllable_count']
        else:
            computer_scansion = "FAILED"
            syllable_count = 0
            penalty = -1
        
        # Normalize both for comparison
        vetted_normalized = normalize_scansion(vetted_scansion)
        computer_normalized = normalize_scansion(computer_scansion)
        
        # Check if they match
        match = (vetted_normalized == computer_normalized)
        
        results.append({
            'line': line_text,
            'vetted_scansion': vetted_scansion,
            'vetted_normalized': vetted_normalized,
            'computer_scansion': computer_scansion,
            'computer_normalized': computer_normalized,
            'match': 'YES' if match else 'NO',
            'syllables': syllable_count,
            'penalty': penalty
        })
    
    # Write to output CSV
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        fieldnames = ['line', 'vetted_scansion', 'vetted_normalized', 
                     'computer_scansion', 'computer_normalized', 'match',
                     'syllables', 'penalty']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        writer.writeheader()
        writer.writerows(results)
    
    # Print summary
    total = len(results)
    matches = sum(1 for r in results if r['match'] == 'YES')
    print(f"\nComparison Summary:")
    print(f"Total lines: {total}")
    print(f"Matches: {matches} ({100*matches/total:.1f}%)")
    print(f"Mismatches: {total - matches} ({100*(total-matches)/total:.1f}%)")
    print(f"\nResults written to: {output_file}")
    
    # Print mismatches
    if total - matches > 0:
        print(f"\nMismatched lines:")
        for r in results:
            if r['match'] == 'NO':
                print(f"\n  Line: {r['line']}")
                print(f"  Vetted:   {r['vetted_scansion']} -> {r['vetted_normalized']}")
                print(f"  Computer: {r['computer_scansion']} -> {r['computer_normalized']}")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python compare_scansion.py vetted_scansion.csv output_comparison.csv")
        sys.exit(1)
    
    vetted_file = sys.argv[1]
    output_file = sys.argv[2]
    
    compare_scansions(vetted_file, output_file)
