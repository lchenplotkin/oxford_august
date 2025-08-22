import pandas as pd

# load your files
fixed = pd.read_csv("fixed.csv")      # has 'infinitive' but no 'headword'
simple = pd.read_csv("simple.csv")    # has 'infinitive' + 'headword'

# --- normalize infinitives ---
fixed['infinitive_norm'] = fixed['infinitive'].str.strip().str.lower()
simple['infinitive_norm'] = simple['infinitive'].str.strip().str.lower()

# merge on normalized infinitives
merged = fixed.merge(
    simple[['infinitive_norm', 'headword']],
    on='infinitive_norm',
    how='left'
)

# drop helper column
merged = merged.drop(columns=['infinitive_norm'])

# reorder columns: headword first
cols = ['headword'] + [c for c in merged.columns if c != 'headword']
merged = merged[cols]

# check for unmatched infinitives
unmatched = merged[merged['headword'].isna()]
if not unmatched.empty:
    print("⚠️ Warning: some infinitives in fixed did not match any headword:")
    print(unmatched['infinitive'].unique())

# save result
merged.to_csv("fixed_with_headwords.csv", index=False)

