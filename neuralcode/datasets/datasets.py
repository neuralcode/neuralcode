import datasets
from typing import List, Optional, Union


def get_code_search_net_dataset(split: Optional[Union[str, List[str]]] = None, lang: str = 'all'):
    dataset = datasets.load_dataset('code_search_net', split=split, name=lang)
    return dataset
