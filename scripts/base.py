import torch
from torch.nn import Module
from torch.nn.utils.rnn import pad_sequence, unpad_sequence


class BaseModel(Module):
    
    
    def __init__(self, word_vocab, char_vocab, scan_vocab, stress_vocab):
        super(BaseModel, self).__init__()
        self.word_vocab = word_vocab
        self.id2word = {v : k for k, v in self.word_vocab.items()}
        self.char_vocab = char_vocab
        self.id2char = {v : k for k, v in self.char_vocab.items()}
        self.scan_vocab = scan_vocab
        self.id2scan = {v : k for k, v in self.scan_vocab.items()}
        self.stress_vocab = stress_vocab
        self.id2stress = {v : k for k, v in self.stress_vocab.items()}
        self._word2id = {v : k for k, v in self.id2word.items()}
        self._scan2id = {v : k for k, v in self.id2scan.items()}
        self._stress2id = {v : k for k, v in self.id2stress.items()}
        self._char2id = {v : k for k, v in self.id2char.items()}        
