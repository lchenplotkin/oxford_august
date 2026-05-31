from functools import partial
from torch.nn import Module, Embedding, LSTM, Linear
from torch.nn.utils.rnn import pad_sequence, unpad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.nn.functional import log_softmax
import torch
from base import BaseModel


class HierarchicalScanModel(BaseModel):

    def __init__(self, embedding_size, word_vocab, char_vocab, scan_vocab, stress_vocab, hidden_size, bidirectional):
        raise NotImplemented()

    def forward(self, batch):
        raise NotImplemented()
