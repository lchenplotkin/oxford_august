from torch.nn import Embedding, LSTM, Linear
from torch.nn.functional import log_softmax
import torch
from base import BaseModel


class CharStressModel(BaseModel):

    def __init__(self, embedding_size, word_vocab, char_vocab, scan_vocab, stress_vocab, hidden_size, bidirectional):
        super(CharStressModel, self).__init__(word_vocab, char_vocab, scan_vocab, stress_vocab)
        self.char_embedding = Embedding(len(self.char_vocab), embedding_size)
        self.char_rnn = LSTM(embedding_size, hidden_size, num_layers=1, batch_first=True, bidirectional=bidirectional, dropout=0.0)
        self.stress_rnn = LSTM(hidden_size * (2 if bidirectional else 1), hidden_size, num_layers=1, batch_first=True, bidirectional=False, dropout=0.0)
        self.head = Linear(hidden_size, len(self.stress_vocab))

    def forward(self, batch):
        char_rnn_input = self.char_embedding(batch["characters"])
        out, _ = self.char_rnn(char_rnn_input)
        out = out.reshape((out.shape[0] * out.shape[1], out.shape[2]))
        # at most, generate up to the length of the longest-observed scan/stress-sequence
        stress_rnn_input = torch.tile(out.unsqueeze(1), (1, batch["stresses"].shape[2], 1))
        stress_out, _ = self.stress_rnn(stress_rnn_input)
        probs = log_softmax(self.head(stress_out), dim=-1)        
        retval = torch.zeros((batch["words"].shape[0], batch["words"].shape[1], batch["stresses"].shape[2], probs.shape[-1]), device=probs.device)
        probs = probs.reshape((batch["characters"].shape[0], batch["characters"].shape[1], batch["stresses"].shape[2], -1))
        return torch.masked_scatter(
            retval,
            torch.tile(batch["word_mask"].unsqueeze(-1).unsqueeze(-1), (1, 1, retval.shape[2], retval.shape[3])),
            probs[batch["word_end_character_mask"]]
        )
