from torch.nn import Module, Embedding, LSTM, Linear


class RNNModel(Module):
    def __init__(self, embedding_size, word_vocab, hidden_size, stress_vocab, bidirectional):
        super(RNNModel, self).__init__()
        self.word_vocab = word_vocab
        self.stress_vocab = stress_vocab
        self.embedding = Embedding(len(self.word_vocab), embedding_size)
        self.rnn = LSTM(embedding_size, hidden_size, num_layers=1, batch_first=True, bidirectional=bidirectional, dropout=0.0)
        self.head = Linear(hidden_size * (2 if bidirectional else 1), len(self.stress_vocab))

    def forward(self, words):
        out, _ = self.rnn(self.embedding(words))
        return self.head(out)
