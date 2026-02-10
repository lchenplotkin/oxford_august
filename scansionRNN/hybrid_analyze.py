import csv
import argparse
import torch
import os.path
import re
from typing import List
from torch.nn import Module, Embedding, LSTM, Linear


def vowel_clusters(word: str) -> List[str]:
    """
    Identify vowel clusters in a word with shared consonants between them.
    Returns a list of syllables.
    """
    clusters = re.findall(r"(^|[^aeiou]+)([aeiou]+)([^aeiou]+$)?", word.lower())
    if not clusters:
        return []
    
    processed = []
    prev_end_consonant = ""
    
    for i, (start, vowels, end) in enumerate(clusters):
        if i == 0:
            prev_consonant = start
        else:
            prev_consonant = prev_end_consonant
        
        if i == len(clusters) - 1:
            next_consonant = end if end is not None else ""
        else:
            if clusters[i+1][0]:
                next_consonant = clusters[i+1][0]
                prev_end_consonant = clusters[i+1][0]
            else:
                next_consonant = ""
                prev_end_consonant = ""
        
        processed.append(prev_consonant + vowels + next_consonant)
    
    return processed


class Model(Module):
    def __init__(self, embedding_size, word_vocab, hidden_size, stress_vocab, bidirectional):
        super(Model, self).__init__()
        self.word_vocab = word_vocab
        self.stress_vocab = stress_vocab
        self.embedding = Embedding(len(self.word_vocab), embedding_size)
        self.rnn = LSTM(embedding_size, hidden_size, num_layers=1, batch_first=True,
                       bidirectional=bidirectional, dropout=0.0)
        self.head = Linear(hidden_size * (2 if bidirectional else 1), len(self.stress_vocab))
    
    def forward(self, words):
        out, _ = self.rnn(self.embedding(words))
        return self.head(out)


def predict_scansion(model, text, char_vocab, stress_vocab):
    """Predict scansion for a line of text at character level, then convert to word-level."""
    model.eval()
    
    words = text.split()
    
    # Build character sequence (with spaces)
    chars = []
    for i, word in enumerate(words):
        chars.extend(list(word.lower()))
        if i < len(words) - 1:
            chars.append(' ')
    
    # Convert to indices
    char_indices = []
    for char in chars:
        if char in char_vocab:
            char_indices.append(char_vocab[char])
        else:
            char_indices.append(0)
    
    if not char_indices:
        return ""
    
    # Predict
    char_tensor = torch.tensor([char_indices])
    
    with torch.no_grad():
        output = model(char_tensor)
        predictions = output.argmax(dim=2).squeeze(0)
    
    # Convert to stress labels
    reverse_stress_vocab = {v: k for k, v in stress_vocab.items()}
    char_stresses = [reverse_stress_vocab[idx.item()] for idx in predictions]
    
    # Convert character-level predictions to word-level scansion
    # Group by words and extract only S/u/x (filter out - and |)
    word_stresses = []
    char_idx = 0
    
    for word in words:
        word_len = len(word)
        word_char_stresses = char_stresses[char_idx:char_idx + word_len]
        
        # Get syllables for this word
        syllables = vowel_clusters(word)
        
        # Extract stress for each syllable (find first S/u/x in syllable region)
        syllable_stresses = []
        chars_per_syll = len(word_char_stresses) / len(syllables) if syllables else 1
        
        for syll_idx in range(len(syllables)):
            start = int(syll_idx * chars_per_syll)
            end = int((syll_idx + 1) * chars_per_syll)
            if end > len(word_char_stresses):
                end = len(word_char_stresses)
            
            # Find first S/u/x in this region
            found_stress = None
            for stress in word_char_stresses[start:end]:
                if stress in ['S', 'u', 'x']:
                    found_stress = stress
                    break
            
            if found_stress:
                syllable_stresses.append(found_stress)
            else:
                # Fallback: use most common stress in region
                region_stresses = [s for s in word_char_stresses[start:end] if s in ['S', 'u', 'x']]
                if region_stresses:
                    syllable_stresses.append(region_stresses[0])
                else:
                    syllable_stresses.append('u')  # Default fallback
        
        word_stresses.append(''.join(syllable_stresses))
        
        # Skip space
        char_idx += word_len
        if char_idx < len(char_stresses) and char_stresses[char_idx] == 'SPACE':
            char_idx += 1
    
    return ' '.join(word_stresses)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze model predictions on scansion data")
    parser.add_argument("--model", required=True, help="Path to trained model file")
    parser.add_argument("--output", required=True, help="Output CSV file")
    parser.add_argument("--device", default=None, help="Device to run on")
    args = parser.parse_args()
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    print(f"Loading model and data from {args.model}...")
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    
    # Extract components
    model = checkpoint['model']
    char_vocab = checkpoint['word_vocab']  # It's actually char vocab
    stress_vocab = checkpoint['stress_vocab']
    train_indices = checkpoint['train_indices']
    dev_indices = checkpoint['dev_indices']
    test_indices = checkpoint['test_indices']
    raw_data = checkpoint['raw_data']
    
    model.eval()
    
    print(f"Generating predictions for {len(raw_data)} lines...")
    print(f"  Train: {len(train_indices)}")
    print(f"  Dev: {len(dev_indices)}")
    print(f"  Test: {len(test_indices)}")
    print(f"  Character vocab size: {len(char_vocab)}")
    print(f"  Stress vocab size: {len(stress_vocab)}")
    
    # Create split lookup
    split_map = {}
    for idx in train_indices:
        split_map[idx] = 'train'
    for idx in dev_indices:
        split_map[idx] = 'dev'
    for idx in test_indices:
        split_map[idx] = 'test'
    
    # Generate predictions and write to CSV
    with open(args.output, 'w', newline='') as csvfile:
        fieldnames = ['text', 'original_scansion', 'predicted_scansion', 
                     'dataset_split', 'filename', 'match']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        matches_by_split = {'train': 0, 'dev': 0, 'test': 0}
        totals_by_split = {'train': 0, 'dev': 0, 'test': 0}
        
        for i, item in enumerate(raw_data):
            if (i + 1) % 1000 == 0:
                print(f"Processed {i + 1}/{len(raw_data)} lines...")
            
            predicted = predict_scansion(model, item['text'], char_vocab, stress_vocab)
            match = predicted == item['scansion']
            split = split_map[i]
            
            if match:
                matches_by_split[split] += 1
            totals_by_split[split] += 1
            
            writer.writerow({
                'text': item['text'],
                'original_scansion': item['scansion'],
                'predicted_scansion': predicted,
                'dataset_split': split,
                'filename': os.path.basename(item['filename']),
                'match': 'yes' if match else 'no'
            })
    
    print(f"\nResults written to {args.output}")
    
    # Print summary statistics
    total_matches = sum(matches_by_split.values())
    total = len(raw_data)
    
    print(f"\nSummary:")
    print(f"Total lines: {total}")
    print(f"Perfect matches: {total_matches} ({100*total_matches/total:.2f}%)")
    
    # Stats by split
    for split in ['train', 'dev', 'test']:
        count = totals_by_split[split]
        matches = matches_by_split[split]
        if count > 0:
            print(f"{split}: {matches}/{count} ({100*matches/count:.2f}%)")
