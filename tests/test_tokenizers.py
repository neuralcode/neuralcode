from neuralcode.tokenizers import TransformerTokenizer

test_code_str = """def quick_sort(collection: list) -> list:
    if len(collection) < 2:
        return collection
    pivot = collection.pop()  # Use the last element as the first pivot
    greater: List[int] = []  # All elements greater than pivot
    lesser: List[int] = []  # All elements less than or equal to pivot
    for element in collection:
        (greater if element > pivot else lesser).append(element)
    return quick_sort(lesser) + [pivot] + quick_sort(greater)"""


def test_transformer_tokenizer():
    tokenizer = TransformerTokenizer()
    tokens = tokenizer.tokenize(test_code_str)
    assert isinstance(tokens, list)
    assert all([isinstance(t, str) for t in tokens]), tokens
    encoded_tokens = tokenizer.encode(test_code_str)
    assert isinstance(encoded_tokens, list)
    assert all([isinstance(t, int) for t in encoded_tokens]), encoded_tokens
    code_str = tokenizer.decode(encoded_tokens, skip_special_tokens=True)
    assert isinstance(code_str, str)
    assert code_str == test_code_str
