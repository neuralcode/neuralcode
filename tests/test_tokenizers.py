from neuralcode.tokenizers import Tokenizer
from neuralcode.tokenizers import TransformerTokenizer
from neuralcode.tokenizers import PygmentsTokenizer


test_code_str = """def quick_sort(collection: list) -> list:
    if len(collection) < 2:
        return collection
    pivot = collection.pop()  # Use the last element as the first pivot
    greater: List[int] = []  # All elements greater than pivot
    lesser: List[int] = []  # All elements less than or equal to pivot
    for element in collection:
        (greater if element > pivot else lesser).append(element)
    return quick_sort(lesser) + [pivot] + quick_sort(greater)"""


def test_tokenizer():
    def tokenize_fn(s):
        return s.split()
    tokenizer = Tokenizer(tokenize_fn)
    tokens = tokenizer.tokenize(test_code_str)
    assert isinstance(tokens, list)
    assert all([isinstance(t, str) for t in tokens])


def test_transformer_tokenizer():
    tokenizer = TransformerTokenizer()
    tokens = tokenizer.tokenize(test_code_str)
    assert isinstance(tokens, list)
    assert all([isinstance(t, str) for t in tokens])


def test_pygments_tokenizer():
    tokenizer = PygmentsTokenizer('.py')
    tokens = tokenizer.tokenize(test_code_str)
    assert isinstance(tokens, list)
    assert all([isinstance(t, str) for t in tokens])
