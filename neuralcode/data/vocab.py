import collections
from typing import Dict, Iterable, List, Optional


class Vocab:
    def __init__(self, unk_token: Optional[str]):
        assert isinstance(unk_token, str) or unk_token is None
        self._stoi = dict()
        self._itos = []
        self.unk_token = unk_token
        if unk_token is not None:
            self.add(unk_token)

    def add(self, token: str) -> int:
        if token not in self._stoi:
            self._itos.append(token)
            self._stoi[token] = len(self._itos) - 1
        return self._stoi[token]

    def stoi(self, token: str) -> int:
        if token in self._stoi:
            return self._stoi[token]
        elif self.unk_token is not None:
            return self._stoi[self.unk_token]
        else:
            raise ValueError(f'unk_token is not set but got unknown token {token}')

    def itos(self, idx: int) -> str:
        return self._itos[idx]

    def __len__(self) -> int:
        return len(self._itos)


def get_vocab_from_iterator(iterator: Iterable[List[str]], unk_token: Optional[str], min_freq: int = 1,
                            max_size: Optional[int] = None, special_tokens: List[str] = []) -> Vocab:

    if max_size is None:
        max_size = 100_000_000

    counter = collections.Counter()

    for i in iterator:
        counter.update(i)

    vocab = Vocab(unk_token)

    for token in special_tokens:
        vocab.add(token)

    for token, count in counter.most_common():
        if len(vocab) <= max_size and count >= min_freq:
            vocab.add(token)
        else:
            break

    return vocab


def get_vocab_from_dict(dictionary: Dict, unk_token: Optional[str]) -> Vocab:

    vocab = Vocab(unk_token=None)

    for token, _ in dictionary.items():
        vocab.add(token)

    vocab.unk_token = unk_token

    return vocab
