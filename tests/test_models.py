from neuralcode.models import TransformerModel
from neuralcode.models import TransformerModelForMaskedLM
from neuralcode.tokenizers import TransformerTokenizer
from neuralcode.tokenizers import TransformerTokenizerForMaskedLM
import torch

test_code_str = """def quick_sort(collection: list) -> list:
    if len(collection) < 2:
        return collection
    pivot = collection.pop()  # Use the last element as the first pivot
    greater: List[int] = []  # All elements greater than pivot
    lesser: List[int] = []  # All elements less than or equal to pivot
    for element in collection:
        (greater if element > pivot else lesser).append(element)
    return quick_sort(lesser) + [pivot] + quick_sort(greater)"""


def test_transformer_model():
    tokenizer = TransformerTokenizer()
    encoded_tokens = tokenizer.encode(test_code_str, return_tensors='pt')
    batch_size, seq_len = encoded_tokens.shape
    model = TransformerModel()
    hid_dim = model.model.config.hidden_size
    output = model(encoded_tokens)
    assert isinstance(output.last_hidden_state, torch.FloatTensor)
    assert output.last_hidden_state.shape == (batch_size, seq_len, hid_dim)
    assert isinstance(output.pooler_output, torch.FloatTensor)
    assert output.pooler_output.shape == (batch_size, hid_dim)


def test_transformer_model_for_masked_lm():
    tokenizer = TransformerTokenizerForMaskedLM()
    encoded_tokens = tokenizer.encode(test_code_str, return_tensors='pt')
    batch_size, seq_len = encoded_tokens.shape
    model = TransformerModelForMaskedLM()
    hid_dim = model.model.config.hidden_size
    output = model(encoded_tokens)
    assert isinstance(output.last_hidden_state, torch.FloatTensor)
    assert output.last_hidden_state.shape == (batch_size, seq_len, hid_dim)
    assert isinstance(output.pooler_output, torch.FloatTensor)
    assert output.pooler_output.shape == (batch_size, hid_dim)
