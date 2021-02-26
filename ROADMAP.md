# ROADMAP.md

## General

* The library should provide methods for analyzing source code with machine learning.
  * Narrowing down the scope, we should use PyTorch for the machine learning side.
  * Therefore, this library should provide helper functions and wrappers around PyTorch modules.
  * HuggingFace Transformers, Tokenizers and Datasets are also worth using as they are high quality repositories.
  * Look into using PyTorch-lightning for easier training.
* The following tasks should be considered:
  * Causal language modeling (i.e. unidirectional language modeling) on source code
  * Masked language modeling on source code
  * Predicting variable names
  * Predicting method names
  * Code to natural language
  * Natural language to code
  * Semantic code search
* Can predicting variable and method names be done with masked language modeling or do they need to be their own task?
* The last three require parsing natural languages

## Datasets

* Should be wrappers around HuggingFace Datasets
* CodeSearchNet is the most popular
* Find other code datasets and add functions to convert to HuggingFace Datasets
* Should be functions and not classes?
* Ability to give the URL of a github repo and NeuralCode automatically downloads it and converts to HuggingFace dataset
  * Requires the ability to:
    * Specify a tokenizer
    * Specify which files to get via extensions

## Tokenizers

* Wrapper for general provided callable tokenize functions
* Should be wrappers around HuggingFace Tokenizers
* Also look into the Pygments library
* Wrappers around ANTLR
* Wrappers around parsing function as AST
* Need to see if HuggingFace Tokenizers can do non-BPE tokenization?
* Should create a `Token` object that carries extra information such as `is_variable`, `is_function`, `is_punct`. `is_keyword`.

## Vocabulary

* Need to build this from scratch

## Models

* There should be a base model class which takes in a sequence of tokens and outputs hidden states.
  * These should be `Token` objects.
  * This sequence should already be tokenized and numericalized.
* Should take in PyTorch nn.Modules
* Also be able to wrap around HuggingFace models
* Then there should be a task specific head for:
  * CasualLM
  * MaskedLM
  * PredVariable
  * PredFunction
  * CodeToNaturalLanguage
  * NaturalLanguageToCode
  * SemanticCodeSearch

## Other

* Connect a VSCode extension to use models
