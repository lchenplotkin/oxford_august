from torch.nn import Module, Embedding, LSTM, Linear
from torch.nn.functional import log_softmax
from torch.nn.utils.rnn import unpad_sequence, pad_packed_sequence, pack_padded_sequence, pad_sequence, pack_sequence
import torch
from base import BaseModel


class HierarchicalStressModel(BaseModel):

    def __init__(self, embedding_size, word_vocab, char_vocab, scan_vocab, stress_vocab, hidden_size, bidirectional):
        raise NotImplemented()

    def forward(self, batch):
        raise NotImplemented()
