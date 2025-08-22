import pandas as pd

# Load both CSVs
gold = pd.read_csv("verb_forms_gold.csv")
simple = pd.read_csv("verb_forms_simple.csv")

# Ensure consistent column order
columns = ["infinitive", "preterite", "past_participle", "classification", "notes"]

# Convert to dict keyed by infinitive for fast lookup
gold_dict = gold.set_index("infinitive").to_dict(orient="index")
simple_dict = simple.set_index("infinitive").to_dict(orient="index")

fixed_rows = []

# Case 1 + 2: Iterate through simple file
for inf, row in simple_dict.items():
    if inf in gold_dict:
        # Merge: classification + notes from gold, rest from simple
        merged = {
            "infinitive": inf,
            "preterite": row.get("preterite", ""),
            "past_participle": row.get("past_participle", ""),
            "classification": gold_dict[inf].get("classification", ""),
            "notes": gold_dict[inf].get("notes", ""),
        }
    else:
        # Keep simple row, append (check!) to notes
        notes_val = row.get("notes", "")
        if pd.isna(notes_val):
            notes_val = ""
        merged = {
            "infinitive": inf,
            "preterite": row.get("preterite", ""),
            "past_participle": row.get("past_participle", ""),
            "classification": row.get("classification", ""),
            "notes": notes_val + " (check!)",
        }
    fixed_rows.append(merged)

# Case 3: Any infinitive in gold not in simple
for inf, row in gold_dict.items():
    if inf not in simple_dict:
        print("ERROR", inf, row)

# Save merged file
fixed_df = pd.DataFrame(fixed_rows, columns=columns)
fixed_df.to_csv("fixed_verb_forms.csv", index=False)

