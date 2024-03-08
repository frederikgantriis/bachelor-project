from copy import deepcopy
import os
import sys
from typing import Callable
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict
from dotenv import load_dotenv
import spacy
from spacy.language import Language
from spacy.tokens import Token

from data_storage import DataStorage


class Datasets(object):
    def __init__(self, dataset_type: str) -> None:
        if dataset_type != "train" and dataset_type != "test":
            raise ValueError(
                "ERROR in Datasets(dataset_type: str): dataset_type argument must be either 'train' or 'test'")
        # Load environment variables from .env file
        load_dotenv()

        # Retrieve data from huggingface and store in ./data folder only if ./data
        # folder doesn't exist or -r flag was used
        dataset_path = './data/dk-hate-dataset'

        if not os.path.isdir(dataset_path) or len(sys.argv) > 1 and sys.argv[1] == "-r":
            load_dataset(
                "DDSC/dkhate", token=os.getenv('HUGGING_FACE_ACCESS_TOKEN')).save_to_disk(dataset_path)

        datasets: DatasetDict = load_from_disk(dataset_path)

        # Unsanitized version of the dataset in a dict of lists of strings format.
        # MUST be sanitized to result in the correct format which is:
        # DICT of LISTS of LISTS of words
        nlp = spacy.load("da_core_news_sm")
        storage = DataStorage()
        folder_path = "data/spacy/"
        disk_path = folder_path + dataset_type + ".pkl"

        try:
            self.dataset = storage.load_from_disk(disk_path)
            print("Found tokenized dataset on disk!")
        except FileNotFoundError:
            print(
                "No tokenized dataset on disk...\nTokenizing dataset using spacy and saving to disk...")
            self.dataset = convert_dataset(nlp, datasets[dataset_type])
            storage.save_to_disk(self.dataset, folder_path, disk_path)
            print("Succesfully tokenized dataset and saved to disk!")

    def remove_dots(self):
        method: Callable[[list[Token]], list[Token]] = lambda lst: [
            x for x in lst if not x.pos_ == "PUNCT"]
        self.dataset = sanitize_dict(self.dataset, method)
        return self

    def remove_stop_words(self):
        method: Callable[[list[Token]], list[Token]] = lambda lst: [
            x for x in lst if not x.is_stop]

        self.dataset = sanitize_dict(self.dataset, method)
        return self

    def to_dict(self) -> dict[str, list]:
        return self.dataset


def convert_dataset(nlp: Language, dataset: Dataset):
    """converts a hugginface dataset into the dataset type accepted by our models

    Args:
        dataset (Dataset): train or test dataset from huggingface

    Returns:
        dict[str, list]: keys = label/classification, values = sentences in classification
    """
    offensive_sentences = []
    not_offensive_sentences = []

    for item in list(zip(dataset["text"], dataset["label"])):
        doc = nlp(item[0])

        if item[1] == "OFF":
            offensive_sentences.append(doc)
        else:
            not_offensive_sentences.append(doc)

    return {"OFF": offensive_sentences, "NOT": not_offensive_sentences}


def sanitize_dict(d, func):
    new_dict = {}
    for key in d.keys():
        lst = []
        for value in d[key]:
            lst.append(func(value))
        new_dict[key] = lst
    return new_dict
