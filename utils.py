from datasets import DatasetDict
from sanitizer import Sanitizer


def extract_sentences_from_label(dataset: DatasetDict, label: str):
    extracted_sentences = []

    for i in range(len(dataset["text"])):
        if dataset["label"][i] == label:
            extracted_sentences.append(dataset["text"][i])

    return extracted_sentences

def get_max_value_key(d: dict):
    max_value = max(d.values())

    for key in d.keys():
        if d[key] == max_value:
            return key


def flatten(matrix: list) -> list:
    """flattens a matrix into a list

    Args:
        lst (list): 2d list

    Returns:
        list: 1d list
    """
    return [item for row in matrix for item in row]
