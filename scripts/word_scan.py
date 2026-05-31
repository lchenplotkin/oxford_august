from torch.nn import Embedding, LSTM, Linear
from torch.nn.functional import log_softmax
from base import BaseModel


class WordScanModel(BaseModel):

    def __init__(self, embedding_size, word_vocab, char_vocab, scan_vocab, stress_vocab, hidden_size, bidirectional):
        super(WordScanModel, self).__init__(word_vocab, char_vocab, scan_vocab, stress_vocab)
        self.word_embedding = Embedding(len(self.word_vocab), embedding_size)
        self.word_rnn = LSTM(embedding_size, hidden_size, num_layers=1, batch_first=True, bidirectional=bidirectional, dropout=0.0)
        self.scan_classifier = Linear(hidden_size * (2 if bidirectional else 1), len(self.scan_vocab))

    def forward(self, batch):
        embs = self.word_embedding(batch["words"])
        out, _ = self.word_rnn(embs)
        return log_softmax(self.scan_classifier(out), dim=2)
