import transformers
from typing import List


class TransformerTokenizer:
    def __init__(self, name='microsoft/codebert-base'):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained('microsoft/codebert-base')

    def tokenize(self, code_str: str) -> List[str]:
        """Tokenize code string."""
        tokens = self.tokenizer.tokenize(code_str)
        return tokens

    def encode(self, code_str: str) -> List[int]:
        """Tokenize and numericalize code string."""
        encoded_tokens = self.tokenizer.encode(code_str)
        return encoded_tokens

    def decode(self, encoded_tokens: List[int], skip_special_tokens: bool = False) -> str:
        """Converts encoded tokens back to code string."""
        code_str = self.tokenizer.decode(encoded_tokens, skip_special_tokens)
        return code_str
