import csv
from scansion import *
import os
from glob import glob
import argparse


def rescan_file(input_file, output_file, target=10):
    """
    Rescan a CSV file with the given target syllable count.
    Updates OXFORD_SCANSION, RIVERSIDE_SCANSION, OXFORD_NUM_SYLS, RIVERSIDE_NUM_SYLS.
    """
    rows = []
    
    # Read the file
    with open(input_file, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        
        for row in reader:
            # Rescan Oxford text
            if row['OXFORD_TEXT']:
                try:
                    oxford_scansion, oxford_syls = scan(row['OXFORD_TEXT'], target)
                    row['OXFORD_SCANSION'] = oxford_scansion
                    row['OXFORD_SCANSION'] = ' '.join(oxford_scansion)
                    row['OXFORD_SYLLABLES'] = oxford_syls
                except Exception as e:
                    print(f"Error scanning Oxford text: {row['OXFORD_TEXT'][:50]}... - {e}")
            
            # Rescan Riverside text
            if row['RIVERSIDE_TEXT']:
                try:
                    riverside_scansion, riverside_syls = scan(row['RIVERSIDE_TEXT'], target)
                    row['RIVERSIDE_SCANSION'] = ' '.join(riverside_scansion)
                    row['RIVERSIDE_SYLLABLES'] = riverside_syls
                except Exception as e:
                    print(f"Error scanning Riverside text: {row['RIVERSIDE_TEXT'][:50]}... - {e}")
            
            rows.append(row)
    
    # Write the updated file
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description='Rescan dataset files with target syllable count')
    parser.add_argument('--data_path', default='full_dataset', help='Path to dataset directory')
    parser.add_argument('--target', type=int, default=10, help='Target syllable count for scansion')
    parser.add_argument('--output_path', default=None, help='Output directory (default: overwrite original files)')
    parser.add_argument('--dry_run', action='store_true', help='Print files that would be processed without actually processing them')
    args = parser.parse_args()
    
    # Get all CSV files in the directory
    all_files = glob(os.path.join(args.data_path, "*.csv"))
    
    # Filter out BD, HF, and combined files
    exclude_patterns = ['BD', 'HF', 'combined']
    files_to_process = [
        f for f in all_files 
        if not any(pattern in os.path.basename(f) for pattern in exclude_patterns)
    ]
    
    print(f"Found {len(all_files)} total CSV files")
    print(f"Processing {len(files_to_process)} files (excluding BD, HF, and combined)")
    print(f"Target syllable count: {args.target}")
    print()
    
    if args.dry_run:
        print("DRY RUN - Files that would be processed:")
        for f in files_to_process:
            print(f"  {os.path.basename(f)}")
        return
    
    # Determine output directory
    if args.output_path:
        os.makedirs(args.output_path, exist_ok=True)
        output_dir = args.output_path
    else:
        output_dir = args.data_path
        print("WARNING: Will overwrite original files!")
        response = input("Continue? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return
    
    # Process each file
    total_rows = 0
    for i, input_file in enumerate(files_to_process, 1):
        basename = os.path.basename(input_file)
        output_file = os.path.join(output_dir, basename)
        
        print(f"[{i}/{len(files_to_process)}] Processing {basename}...", end=' ')
        
        try:
            num_rows = rescan_file(input_file, output_file, args.target)
            total_rows += num_rows
            print(f"✓ ({num_rows} rows)")
        except Exception as e:
            print(f"✗ Error: {e}")
    
    print()
    print(f"Complete! Processed {total_rows} total rows across {len(files_to_process)} files.")
    if args.output_path:
        print(f"Output written to: {args.output_path}")


if __name__ == "__main__":
    main()
