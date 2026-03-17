import csv

def merge_with_green_flags(todo_file, in_progress_file, output_file):
    """
    Go through line by line:
    - If in_progress line has green flag, use that line
    - Otherwise, use line from todo file
    """
    # Read both files
    with open(todo_file, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        todo_lines = list(reader)
    
    with open(in_progress_file, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        in_progress_lines = list(reader)
    
    # Make sure they have the same number of lines
    if len(todo_lines) != len(in_progress_lines):
        print(f"WARNING: Files have different lengths!")
        print(f"  todo: {len(todo_lines)} lines")
        print(f"  in_progress: {len(in_progress_lines)} lines")
        print(f"  Will process up to the shorter length")
    
    # Merge line by line
    output_lines = []
    green_count = 0
    
    for i, (todo_line, progress_line) in enumerate(zip(todo_lines, in_progress_lines)):
        # Check if in_progress line has green flag (column index 5)
        has_green = progress_line[-1] == 'green'
        
        if has_green:
            output_lines.append(progress_line)
            green_count += 1
        else:
            output_lines.append(todo_line)
    
    # Write output
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(output_lines)
    
    print(f"Merged CSV written to {output_file}")
    print(f"Total lines: {len(output_lines)}")
    print(f"Lines with green flags (from in_progress): {green_count}")
    print(f"Lines from todo: {len(output_lines) - green_count}")

# Usage
if __name__ == "__main__":
    merge_with_green_flags(
        todo_file='for_gui/to_do/HF_gui.csv',
        in_progress_file='for_gui/done/HF_gui_complete.csv',
        output_file='for_gui/done/HF_gui_corrected_scansion.csv'
    )
