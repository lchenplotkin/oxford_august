from torch.nn import Embedding, LSTM, Linear
from torch.nn.functional import log_softmax
import torch
from base import BaseModel


class CharScanModel(BaseModel):

    def __init__(self, embedding_size, word_vocab, char_vocab, scan_vocab, stress_vocab, hidden_size, bidirectional):
        super(CharScanModel, self).__init__(word_vocab, char_vocab, scan_vocab, stress_vocab)
        self.char_embedding = Embedding(len(self.char_vocab), embedding_size)
        self.char_rnn = LSTM(embedding_size, hidden_size, num_layers=1, batch_first=True, bidirectional=bidirectional, dropout=0.0)
        self.scan_classifier = Linear(hidden_size * (2 if bidirectional else 1), len(self.scan_vocab))

    def forward(self, batch):
        char_rnn_input = self.char_embedding(batch["characters"])
        out, _ = self.char_rnn(char_rnn_input)
        out = out[batch["word_end_character_mask"]]
        probs = log_softmax(self.scan_classifier(out), dim=1)
        retval = torch.zeros((batch["words"].shape[0], batch["words"].shape[1], probs.shape[-1]), device=probs.device)
        return torch.masked_scatter(
            retval,
            torch.tile(batch["word_mask"].unsqueeze(-1), (1, 1, retval.shape[2])),
            probs
        )
