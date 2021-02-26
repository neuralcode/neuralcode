from typing import Callable, List

import pygments.lexers
import transformers


class Tokenizer:
    def __init__(self, tokenize_fn: Callable):
        self.tokenize_fn = tokenize_fn
        example_tokens = self.tokenize_fn('example string')
        assert isinstance(example_tokens, list), 'tokenize_fn must return a List[str]'
        assert all([isinstance(t, str) for t in example_tokens]), 'tokenize_fn must return a List[str]'

    def tokenize(self, s: str) -> List[str]:
        tokens = self.tokenize_fn(s)
        return tokens


class TransformerTokenizer:
    def __init__(self, name: str = 'microsoft/codebert-base'):
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(name)
        self._vocab = self._tokenizer.vocab

    def tokenize(self, s: str, **kwargs) -> List[str]:
        idxs = self._tokenizer.encode(s, truncation=True)
        tokens = self._tokenizer.convert_ids_to_tokens(idxs)
        return tokens


class PygmentsTokenizer:
    def __init__(self, extension: str):
        self._lexer = pygments.lexers.get_lexer_for_filename(extension)

    def tokenize(self, s: str) -> List[str]:
        lexer_output = self._lexer.get_tokens(s)
        tokens = [t for _, t in lexer_output if t.strip() != '']
        return tokens
