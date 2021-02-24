import datasets


def get_code_search_net_dataset():
    dataset = datasets.load_dataset('code_search_net')
    return dataset


def get_dataset_from_github_repo():
