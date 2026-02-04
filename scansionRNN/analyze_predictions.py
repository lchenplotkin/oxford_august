import csv
import argparse
import torch
import os.path
from torch.nn import Module, Embedding, LSTM, Linear


class Model(Module):
    def __init__(self, embedding_size, word_vocab, hidden_size, stress_vocab, bidirectional):
        super(Model, self).__init__()
        self.word_vocab = word_vocab
        self.stress_vocab = stress_vocab
        self.embedding = Embedding(len(self.word_vocab), embedding_size)
        self.rnn = LSTM(embedding_size, hidden_size, num_layers=1, batch_first=True, bidirectional=bidirectional, dropout=0.0)
        self.head = Linear(hidden_size * (2 if bidirectional else 1), len(self.stress_vocab))

    def forward(self, words):
        out, _ = self.rnn(self.embedding(words))
        return self.head(out)


def predict_scansion(model, words, word_vocab, stress_vocab):
    """Predict scansion for a line of text."""
    model.eval()
    
    # Convert words to indices
    word_indices = []
    for word in words.split():
        if word in word_vocab:
            word_indices.append(word_vocab[word])
        else:
            # Use index 0 for unknown words
            word_indices.append(0)
    
    # Create tensor and predict
    word_tensor = torch.tensor([word_indices])
    
    with torch.no_grad():
        output = model(word_tensor)
        predictions = output.argmax(dim=2).squeeze(0)
    
    # Convert predictions back to stress symbols
    reverse_stress_vocab = {v: k for k, v in stress_vocab.items()}
    predicted_stresses = [reverse_stress_vocab[idx.item()] for idx in predictions]
    
    return ' '.join(predicted_stresses)


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
    word_vocab = checkpoint['word_vocab']
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
            
            predicted = predict_scansion(model, item['text'], word_vocab, stress_vocab)
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
