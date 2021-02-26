# MILESTONES.md

## Milestone One

* [x] CodeSearchNet dataset
* Tokenizer wrapper:
  * [x] Wrapper for provided callable `tokenize_fn`
  * [x] Wrapper for HuggingFace Tokenizers
  * [x] Wrapper for Pygments
  * Improve tokenizer tests
* Method to strip tokens from a file
* ~~`Token` class~~
* [x] Vocab class
  * Improve vocab tests
* BaseModel
  * Wrapper around PyTorch nn.Module
  * Wrapper around HuggingFace module
  * CausalLM
  * MaskedLM
* Training code
  * CausalLM
  * MaskedLM
