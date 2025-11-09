from . import _native

class Tokenizer:
    def __init__(self, vocab=None):
        if vocab is None:
            vocab = {}
        self._tokenizer = _native.Tokenizer(vocab)

    def train_streaming(self, iterator, vocab_size, shrinking_factor=0.75):
        self._tokenizer.train(list(iterator), vocab_size, shrinking_factor)

    def encode(self, s: str) -> list[int]:
        return self._tokenizer.encode(s)

    def decode(self, ids: list[int]) -> str:
        return self._tokenizer.decode(ids)

    def save(self, path: str):
        model = _native.Model(self._tokenizer)
        model.save(path)

    @classmethod
    def from_file(cls, path: str):
        model = _native.Model.load(path)
        tokenizer = cls()
        tokenizer._tokenizer = model.tokenizer
        return tokenizer
