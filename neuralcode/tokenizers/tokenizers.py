import transformers
from typing import List


class TransformerTokenizer:
    def __init__(self, name: str = 'microsoft/codebert-base'):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained('microsoft/codebert-base')

    def tokenize(self, code_str: str, **kwargs) -> List[str]:
        """Tokenize code string."""
        tokens = self.tokenizer.tokenize(code_str, **kwargs)
        return tokens

    def encode(self, code_str: str, **kwargs) -> List[int]:
        """Tokenize and numericalize code string."""
        encoded_tokens = self.tokenizer.encode(code_str, **kwargs)
        return encoded_tokens

    def decode(self, encoded_tokens: List[int], **kwargs) -> str:
        """Converts encoded tokens back to code string."""
        code_str = self.tokenizer.decode(encoded_tokens, **kwargs)
        return code_str


class TransformerTokenizerForMaskedLM(TransformerTokenizer):
    def __init__(self, name: str = 'microsoft/codebert-base-mlm'):
        self.tokenizer = transformers.RobertaTokenizer.from_pretrained('microsoft/codebert-base-mlm')
