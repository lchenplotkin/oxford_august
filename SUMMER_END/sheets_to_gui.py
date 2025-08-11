import csv

def process_six_lines(block):
    def extract(line):
        return line.split(",", 7)  # split up to num_sybs and rest
    def extract_text(line):
        return line.split("||",7)

    og_riv  = extract_text(block[0])[0].rstrip(',"').lstrip('",')
    og_oxford = extract_text(block[0])[1].rstrip(',"').lstrip('",')
    meta = extract(block[1])[:5]  # filename info + match
    flag = extract(block[1])[5]
    riverside_sybs = extract(block[1])[6]
    oxford_sybs = extract(block[4])[6]

    riverside_text = " ".join(extract(block[1])[7].split(","))
    riverside_scansion = " ".join(extract(block[2])[7].split(","))
    riverside_tags = " ".join(extract(block[3])[7].split(","))
    oxford_text = " ".join(extract(block[4])[7].split(","))
    oxford_scansion = " ".join(extract(block[5])[7].split(","))
    oxford_tags = " ".join(extract(block[6])[7].split(","))

    return [og_riv, og_oxford] + meta + [
        riverside_sybs,
        oxford_sybs,
        riverside_text,
        oxford_text,
        riverside_scansion,
        oxford_scansion,
        riverside_tags,
        oxford_tags,
        flag
    ]

def convert_file(input_filename, output_filename):
    with open(input_filename, 'r') as infile:
        lines = [line.strip() for line in infile if line.strip()]

    if len(lines) % 7 != 0:
        raise ValueError("Input file does not contain a multiple of 6 lines.")

    header = [
        "OG_RIV_TEXT", "OG_OXFORD_TEXT", "RIVERSIDE_FILENAME", "OXFORD_FILENAME", "OUTPUT_FILENAME",
        "LINE_NUMBER", "MATCH",
        "RIVERSIDE_SYLLABLES", "OXFORD_SYLLABLES",
        "RIVERSIDE_TEXT", "OXFORD_TEXT",
        "RIVERSIDE_SCANSION", "OXFORD_SCANSION",
        "RIVERSIDE_TAGGING", "OXFORD_TAGGING",
        "FLAG_COLOR"
    ]

    with open(output_filename, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(header)

        for i in range(0, len(lines), 7):
            block = lines[i:i+7]
            writer.writerow(process_six_lines(block))

# Example usage:
#convert_file('output/GP.csv', 'output/GP_alt_format.csv')

