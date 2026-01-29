"""
Aggregate Per-File Accuracy Results by Text Groups
Processes the per_file_accuracy_breakdown.csv and creates group-level statistics
"""

import csv
import pandas as pd
import os

# Define the text groupings
TEXT_GROUPS = {
    'BD': ['BD'],
    'PF': ['PF'],
    'HF': ['HF'],
    'TC1': ['TC1'],
    'TC2': ['TC2'],
    'TC3': ['TC3'],
    'TC4': ['TC4'],
    'TC5': ['TC5'],
    'LGW': ['LGW'],
    'LGW_FPro': ['LGW_FPro'],
    'MkT': ['MkT'],
    'SNT': ['SNT'],
    'GP': ['GP'],
    'MilT': ['MilT'],
    'RvT': ['RvT'],
    'PhyT': ['PhyT'],
    'PrT': ['PrT'],
    'MancT': ['MancT'],
    'Thop': ['Thop'],
    'NPT': ['NPT'],
    'PardT': ['PardT'],
    'ClT': ['ClT'],
    'SqT': ['SqT'],
    'MLT': ['MLT'],
    'CYT': ['CYT'],
    'ShipT': ['ShipT'],
    'WBPro': ['WBPro'],
    'WBT': ['WBT'],
    'FrT': ['FrT'],
    'SumT': ['SumT'],
    'MerT': ['MerT'],
    'FranT': ['FranT'],
}

# Higher-level aggregations
AGGREGATE_GROUPS = {
    'BD': ['BD'],
    'PF': ['PF'],
    'HF': ['HF'],
    'TC': ['TC1', 'TC2', 'TC3', 'TC4', 'TC5'],
    'LGW': ['LGW'],
    'LGW_FPro': ['LGW_FPro'],
    'Early_CT': ['MkT', 'SNT'],
    'EarlyMid_CT': ['GP', 'MilT', 'RvT', 'PhyT', 'PrT', 'MancT', 'Thop'],
    'LateMid_CT': ['NPT', 'PardT', 'ClT', 'SqT', 'MLT', 'CYT', 'ShipT'],
    'Late_CT': ['WBPro', 'WBT', 'FrT', 'SumT', 'MerT', 'FranT', 'Links'],
}

def identify_text_group(filename):
    """
    Identify which text group a file belongs to based on its filename.
    Returns the base text identifier (e.g., 'TC1', 'MilT', 'GP', etc.)
    """
    # Remove common suffixes to get base identifier
    filename = filename.replace('_gui_complete.csv', '')
    
    # Check each defined text group
    for group_key in TEXT_GROUPS.keys():
        if filename.startswith(group_key) or group_key in filename:
            return group_key
    
    # Check if it's a Pro file that should go in Links
    if 'Pro' in filename and filename not in ['LGW_FPro', 'WBPro']:
        return 'Links'
    
    # If no match found, return the filename itself
    return None

def aggregate_group_statistics(df, group_files):
    """
    Aggregate statistics for a group of files.
    Returns a dictionary with aggregated counts and calculated accuracy rates.
    """
    # Filter dataframe to only include files in this group
    group_df = df[df['text_group'].isin(group_files)]
    
    if len(group_df) == 0:
        return None
    
    # Aggregate counts
    aggregated = {
        'num_files': len(group_df),
        'total_monosyllabic': group_df['monosyllabic_adjectives_found'].sum(),
        
        # Overall stats
        'overall_total': group_df['total_instances'].sum(),
        'overall_correct': group_df['total_correct'].sum(),
        
        'overall_total_filtered': group_df['total_instances_filtered'].sum(),
        'overall_correct_filtered': group_df['total_correct_filtered'].sum(),
        
        # Weak declension stats
        'weak_total': group_df['weak_total'].sum(),
        'weak_correct': group_df['weak_correct'].sum(),
        
        # Plural stats
        'plural_total': group_df['plural_total'].sum(),
        'plural_correct': group_df['plural_correct'].sum(),
        
        # Strong stats
        'strong_total': group_df['strong_total'].sum(),
        'strong_correct': group_df['strong_correct'].sum(),
        
        # Strong filtered stats
        'strong_total_filtered': group_df['strong_total_filtered'].sum(),
        'strong_correct_filtered': group_df['strong_correct_filtered'].sum(),
    }
    
    # Calculate accuracy rates
    if aggregated['overall_total'] > 0:
        aggregated['overall_accuracy'] = (aggregated['overall_correct'] / aggregated['overall_total']) * 100
    else:
        aggregated['overall_accuracy'] = None
    
    if aggregated['overall_total_filtered'] > 0:
        aggregated['overall_accuracy_filtered'] = (aggregated['overall_correct_filtered'] / aggregated['overall_total_filtered']) * 100
    else:
        aggregated['overall_accuracy_filtered'] = None
    
    if aggregated['weak_total'] > 0:
        aggregated['weak_accuracy'] = (aggregated['weak_correct'] / aggregated['weak_total']) * 100
    else:
        aggregated['weak_accuracy'] = None
    
    if aggregated['plural_total'] > 0:
        aggregated['plural_accuracy'] = (aggregated['plural_correct'] / aggregated['plural_total']) * 100
    else:
        aggregated['plural_accuracy'] = None
    
    if aggregated['strong_total'] > 0:
        aggregated['strong_accuracy'] = (aggregated['strong_correct'] / aggregated['strong_total']) * 100
    else:
        aggregated['strong_accuracy'] = None
    
    if aggregated['strong_total_filtered'] > 0:
        aggregated['strong_accuracy_filtered'] = (aggregated['strong_correct_filtered'] / aggregated['strong_total_filtered']) * 100
    else:
        aggregated['strong_accuracy_filtered'] = None
    
    return aggregated

def process_per_file_data(input_csv, output_dir):
    """
    Process the per-file accuracy breakdown and create group-level aggregations.
    """
    # Read the per-file accuracy data
    df = pd.read_csv(input_csv)
    
    # Convert percentage strings to floats (if they exist as strings)
    for col in df.columns:
        if 'accuracy' in col and df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Identify text group for each file
    df['text_group'] = df['filename'].apply(identify_text_group)
    
    # Handle Links group (files ending in Pro that aren't already grouped)
    links_files = []
    for idx, row in df.iterrows():
        if row['text_group'] == 'Links':
            links_files.append(row['filename'])
    
    if links_files:
        TEXT_GROUPS['Links'] = links_files
        AGGREGATE_GROUPS['Links'] = ['Links']
    
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate individual text group statistics
    individual_results = []
    for group_name, group_files in TEXT_GROUPS.items():
        stats = aggregate_group_statistics(df, group_files)
        if stats:
            stats['group_name'] = group_name
            stats['group_type'] = 'individual'
            individual_results.append(stats)
    
    # Generate aggregate group statistics
    aggregate_results = []
    for group_name, member_groups in AGGREGATE_GROUPS.items():
        stats = aggregate_group_statistics(df, member_groups)
        if stats:
            stats['group_name'] = group_name
            stats['group_type'] = 'aggregate'
            aggregate_results.append(stats)
    
    # Write individual text groups to CSV
    write_group_results(individual_results, 
                       os.path.join(output_dir, 'individual_text_accuracy.csv'))
    
    # Write aggregate groups to CSV
    write_group_results(aggregate_results, 
                       os.path.join(output_dir, 'aggregate_group_accuracy.csv'))
    
    # Write combined results to CSV
    combined_results = individual_results + aggregate_results
    write_group_results(combined_results, 
                       os.path.join(output_dir, 'combined_group_accuracy.csv'))
    
    # Create a detailed mapping of files to groups
    write_file_mapping(df, os.path.join(output_dir, 'file_to_group_mapping.csv'))
    
    # Create summary report
    create_summary_report(individual_results, aggregate_results, 
                         os.path.join(output_dir, 'group_analysis_summary.txt'))
    
    return individual_results, aggregate_results

def write_group_results(results, output_path):
    """Write group results to CSV file."""
    if not results:
        print(f"No results to write to {output_path}")
        return
    
    fieldnames = [
        'group_name',
        'group_type',
        'num_files',
        'total_monosyllabic',
        'overall_accuracy',
        'overall_total',
        'overall_correct',
        'overall_accuracy_filtered',
        'overall_total_filtered',
        'overall_correct_filtered',
        'weak_accuracy',
        'weak_total',
        'weak_correct',
        'plural_accuracy',
        'plural_total',
        'plural_correct',
        'strong_accuracy',
        'strong_total',
        'strong_correct',
        'strong_accuracy_filtered',
        'strong_total_filtered',
        'strong_correct_filtered',
    ]
    
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in results:
            # Format percentages to 2 decimal places
            formatted = {}
            for key, value in result.items():
                if 'accuracy' in key and value is not None:
                    formatted[key] = f"{value:.2f}"
                else:
                    formatted[key] = value
            writer.writerow(formatted)
    
    print(f"Wrote {len(results)} group results to {output_path}")

def write_file_mapping(df, output_path):
    """Write a mapping of files to their assigned groups."""
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['filename', 'text_group', 'aggregate_group'])
        
        for idx, row in df.iterrows():
            text_group = row['text_group']
            
            # Find which aggregate group this belongs to
            aggregate_group = None
            for agg_name, members in AGGREGATE_GROUPS.items():
                if text_group in members:
                    aggregate_group = agg_name
                    break
            
            writer.writerow([row['filename'], text_group, aggregate_group])
    
    print(f"Wrote file mapping to {output_path}")

def create_summary_report(individual_results, aggregate_results, output_path):
    """Create a text summary report of the group analysis."""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("CHAUCER ADJECTIVE DECLENSION - GROUP ANALYSIS SUMMARY\n")
        f.write("="*70 + "\n\n")
        
        # Aggregate groups summary
        f.write("AGGREGATE GROUPS\n")
        f.write("-"*70 + "\n\n")
        
        for result in sorted(aggregate_results, key=lambda x: x['group_name']):
            f.write(f"{result['group_name']}:\n")
            f.write(f"  Files: {result['num_files']}\n")
            f.write(f"  Monosyllabic adjectives: {result['total_monosyllabic']}\n")
            
            if result['overall_accuracy'] is not None:
                f.write(f"  Overall accuracy: {result['overall_accuracy']:.2f}% ")
                f.write(f"({result['overall_correct']}/{result['overall_total']})\n")
            
            if result['overall_accuracy_filtered'] is not None:
                f.write(f"  Overall accuracy (filtered): {result['overall_accuracy_filtered']:.2f}% ")
                f.write(f"({result['overall_correct_filtered']}/{result['overall_total_filtered']})\n")
            
            if result['weak_accuracy'] is not None:
                f.write(f"  Weak declension: {result['weak_accuracy']:.2f}% ")
                f.write(f"({result['weak_correct']}/{result['weak_total']})\n")
            
            if result['plural_accuracy'] is not None:
                f.write(f"  Plural forms: {result['plural_accuracy']:.2f}% ")
                f.write(f"({result['plural_correct']}/{result['plural_total']})\n")
            
            if result['strong_accuracy_filtered'] is not None:
                f.write(f"  Strong forms (filtered): {result['strong_accuracy_filtered']:.2f}% ")
                f.write(f"({result['strong_correct_filtered']}/{result['strong_total_filtered']})\n")
            
            f.write("\n")
        
        # Individual texts summary
        f.write("\n" + "="*70 + "\n")
        f.write("INDIVIDUAL TEXTS\n")
        f.write("-"*70 + "\n\n")
        
        for result in sorted(individual_results, key=lambda x: x['group_name']):
            f.write(f"{result['group_name']}: ")
            if result['overall_accuracy_filtered'] is not None:
                f.write(f"{result['overall_accuracy_filtered']:.2f}% ")
                f.write(f"({result['overall_correct_filtered']}/{result['overall_total_filtered']})")
            else:
                f.write("No data")
            f.write(f" - {result['total_monosyllabic']} monosyllabic adj.\n")
        
        f.write("\n" + "="*70 + "\n")
    
    print(f"Wrote summary report to {output_path}")

if __name__ == "__main__":
    # Configuration
    input_csv = 'adjective_analysis_output/per_file_accuracy_breakdown.csv'
    output_dir = 'group_analysis_output'
    
    print("Processing per-file accuracy data...")
    print(f"Input: {input_csv}")
    print(f"Output directory: {output_dir}\n")
    
    try:
        individual, aggregate = process_per_file_data(input_csv, output_dir)
        
        print("\n" + "="*70)
        print("Analysis complete!")
        print("="*70)
        print(f"\nProcessed {len(individual)} individual texts")
        print(f"Created {len(aggregate)} aggregate groups")
        print(f"\nResults saved to '{output_dir}' directory:")
        print("  - individual_text_accuracy.csv")
        print("  - aggregate_group_accuracy.csv")
        print("  - combined_group_accuracy.csv")
        print("  - file_to_group_mapping.csv")
        print("  - group_analysis_summary.txt")
        
    except FileNotFoundError:
        print(f"\nError: Could not find input file '{input_csv}'")
        print("Please make sure you've run the main analysis script first.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
