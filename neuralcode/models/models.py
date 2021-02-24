import transformers


class TransformerModel:
    def __init__(self, name: str = 'microsoft/codebert-base'):
        self.model = transformers.AutoModel.from_pretrained(name)

    def __call__(self, x, **kwargs):
        return self.model(x, **kwargs)


class TransformerModelForMaskedLM(TransformerModel):
    def __init__(self, name: str = 'microsoft/codebert-base-mlm'):
        self.model = transformers.RobertaForMaskedLM.from_pretrained(name)
