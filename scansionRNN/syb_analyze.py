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
    def __init__(self, embedding_size, syllable_vocab, hidden_size, stress_vocab, bidirectional):
        super(Model, self).__init__()
        self.syllable_vocab = syllable_vocab
        self.stress_vocab = stress_vocab
        self.embedding = Embedding(len(self.syllable_vocab), embedding_size)
        self.rnn = LSTM(embedding_size, hidden_size, num_layers=1, batch_first=True, 
                       bidirectional=bidirectional, dropout=0.0)
        self.head = Linear(hidden_size * (2 if bidirectional else 1), len(self.stress_vocab))
    
    def forward(self, syllables):
        out, _ = self.rnn(self.embedding(syllables))
        return self.head(out)


def predict_scansion(model, text, syllable_vocab, stress_vocab):
    """Predict scansion for a line of text at syllable level."""
    model.eval()
    
    words = text.split()
    all_syllables = []
    syllable_to_word = []  # Track which word each syllable belongs to
    
    # Convert words to syllables
    for word_idx, word in enumerate(words):
        syllables = vowel_clusters(word)
        for syllable in syllables:
            all_syllables.append(syllable)
            syllable_to_word.append(word_idx)
    
    # Convert syllables to indices
    syllable_indices = []
    for syllable in all_syllables:
        if syllable in syllable_vocab:
            syllable_indices.append(syllable_vocab[syllable])
        else:
            # Use index 0 for unknown syllables
            syllable_indices.append(0)
    
    # Create tensor and predict
    if not syllable_indices:
        return ""
    
    syllable_tensor = torch.tensor([syllable_indices])
    
    with torch.no_grad():
        output = model(syllable_tensor)
        predictions = output.argmax(dim=2).squeeze(0)
    
    # Convert predictions back to stress symbols
    reverse_stress_vocab = {v: k for k, v in stress_vocab.items()}
    predicted_syllable_stresses = [reverse_stress_vocab[idx.item()] for idx in predictions]
    
    # Group syllable stresses back into words
    word_stresses = []
    current_word_idx = -1
    current_word_stress = []
    
    for syllable_idx, word_idx in enumerate(syllable_to_word):
        if word_idx != current_word_idx:
            # Starting a new word
            if current_word_stress:
                word_stresses.append(''.join(current_word_stress))
            current_word_idx = word_idx
            current_word_stress = []
        
        if syllable_idx < len(predicted_syllable_stresses):
            current_word_stress.append(predicted_syllable_stresses[syllable_idx])
    
    # Don't forget the last word
    if current_word_stress:
        word_stresses.append(''.join(current_word_stress))
    
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
    syllable_vocab = checkpoint['syllable_vocab']
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
    print(f"  Syllable vocab size: {len(syllable_vocab)}")
    
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
            
            predicted = predict_scansion(model, item['text'], syllable_vocab, stress_vocab)
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
