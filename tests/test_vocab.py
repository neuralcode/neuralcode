from neuralcode.data import Vocab
from neuralcode.data import get_vocab_from_dict
from neuralcode.data import get_vocab_from_iterator


def test_get_vocab_from_dict():
    dictionary = {'hello': 0, 'world': 1, 'how': 2, 'are': 3, 'you': 4, '?': 5}
    vocab = get_vocab_from_dict(dictionary, unk_token='<unk>')
    assert isinstance(vocab, Vocab)


def test_get_vocab_from_iterator():
    iterator = [['one', 'two', 'three'], ['four', 'five', 'six'], ['seven', 'eight', 'nine']]
    vocab = get_vocab_from_iterator(iterator, unk_token='<unk>')
    assert isinstance(vocab, Vocab)
