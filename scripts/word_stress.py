from functools import partial
from torch.nn import Module, Embedding, LSTM, Linear
from torch.nn.functional import log_softmax
from torch.nn.utils.rnn import pad_sequence, unpad_sequence, pack_padded_sequence, pad_packed_sequence
import torch
from base import BaseModel


class WordStressModel(BaseModel):

    def __init__(self, embedding_size, word_vocab, char_vocab, scan_vocab, stress_vocab, hidden_size, bidirectional):
        super(WordStressModel, self).__init__(word_vocab, char_vocab, scan_vocab, stress_vocab)
        self.word_embedding = Embedding(len(self.word_vocab), embedding_size)
        self.word_rnn = LSTM(embedding_size, hidden_size, num_layers=1, batch_first=True, bidirectional=bidirectional, dropout=0.0)
        self.stress_rnn = LSTM(hidden_size * (2 if bidirectional else 1), hidden_size, num_layers=1, batch_first=True, bidirectional=False, dropout=0.0)
        self.head = Linear(hidden_size, len(self.stress_vocab))

    def forward(self, batch):
        word_rnn_input = self.word_embedding(batch["words"])
        out, _ = self.word_rnn(word_rnn_input)
        # at most, generate up to the length of the longest-observed scan/stress-sequence
        stress_rnn_input = torch.tile(
            out.unsqueeze(2),
            (1, 1, batch["stresses"].shape[2], 1)
        ).reshape(out.shape[0] * out.shape[1], batch["stresses"].shape[2], -1)
        stress_out, _ = self.stress_rnn(stress_rnn_input)
        return log_softmax(self.head(stress_out), dim=2).reshape(out.shape[0], out.shape[1], batch["stresses"].shape[2], -1)
