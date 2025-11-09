from . import _native


class Tokenizer:
    def __init__(self, vocab=None):
        if vocab is None:
            vocab = {}
        self._tokenizer = _native.Tokenizer(vocab)

    def train_streaming(self, iterator, vocab_size, shrinking_factor=0.75, subword_regularization=False):
        self._tokenizer.train(list(iterator), vocab_size, shrinking_factor, subword_regularization)

    def encode(
        self, s: str, mode: str = None
    ) -> tuple[list[int], list[tuple[int, int]]]:
        return self._tokenizer.encode(s, mode)

    def decode(self, ids: list[int], mode: str = None) -> str:
        return self._tokenizer.decode(ids, mode)

    def segment_probabilities(self, text: str, top_k: int = 5) -> list[tuple[list[str], float]]:
        return self._tokenizer.segment_probabilities(text, top_k)

    def save(self, path: str):
        model = _native.Model(self._tokenizer)
        model.save(path)

    @classmethod
    def from_file(cls, path: str):
        model = _native.Model.load(path)
        tokenizer = cls()
        tokenizer._tokenizer = model.tokenizer
        return tokenizer
