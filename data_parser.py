import os
import sys
import spacy

from typing import Callable
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict
from dotenv import load_dotenv
from spacy.language import Language
from spacy.tokens import Token
from data_storage import DataStorage
from utils import flatten
from constants import OFF, NOT, TEST, TRAIN


class Datasets(object):
    def __init__(self, dataset_type: str) -> None:
        if dataset_type not in [TRAIN, TEST]:
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

        self.datasets: DatasetDict = load_from_disk(dataset_path)

        # Unsanitized version of the dataset in a dict of lists of strings format.
        # MUST be sanitized to result in the correct format which is:
        # DICT of LISTS of LISTS of words
        self.nlp = spacy.load("da_core_news_sm")
        self.dataset_type = dataset_type
        self.storage = DataStorage()
        self.folder_path = "data/spacy/"
        self._initial_sync_with_disk()

    def to_dict(self) -> dict[str, list]:
        """Returns:
            dict[str, list]: a dataset in the format that our models accept
        """
        return self.dataset

    def to_list(self) -> list[list]:
        return flatten([[x for x in lst] for lst in self.dataset.values()])

    def remove_dots(self):
        """remove all punctuation"""

        # try getting the dataset variation from cache
        self.dataset_type = self.dataset_type + "_remove-dots"
        if self._try_load_from_disk():
            return self

        method: Callable[[list[Token]], list[Token]] = lambda lst: [
            x for x in lst if not x.pos_ == "PUNCT"]
        self.dataset = self._sanitize_dataset(self.dataset, method)

        # save to disk for quicker execution next time
        self._save_to_disk()
        return self

    def remove_stop_words(self):
        """remove the most common words in the danish language"""
        self.dataset_type = self.dataset_type + "_remove-stop-words"
        if self._try_load_from_disk():
            return self

        method: Callable[[list[Token]], list[Token]] = lambda lst: [
            x for x in lst if not x.is_stop]
        self.dataset = self._sanitize_dataset(self.dataset, method)

        self._save_to_disk()
        return self

    def lemmatize(self):
        """group words together and convert to simplest form (see: https://en.wikipedia.org/wiki/Lemmatization)"""
        self.dataset_type = self.dataset_type + "_lemmatize"
        if self._try_load_from_disk():
            return self

        method: Callable[[list[Token]], list[Token]] = lambda lst: [x for x in self.nlp(" ".join([
            x.lemma_ for x in lst]))]
        self.dataset = self._sanitize_dataset(self.dataset, method)
        self._save_to_disk()
        return self

    def lowercase(self):
        """lowercase wuhu"""
        self.dataset_type = self.dataset_type + "_lowercase"
        if self._try_load_from_disk():
            return self

        method: Callable[[list[Token]], list[Token]] = lambda lst: [x for x in self.nlp(" ".join([
            x.lower_ for x in lst]))]
        self.dataset = self._sanitize_dataset(self.dataset, method)
        self._save_to_disk()
        return self

    def shuffle(self):
        """extracts unique words from the dataset"""
        self.dataset_type = self.dataset_type + "_shuffle"
        if self._try_load_from_disk():
            return self

        method: Callable[[list[Token]], list[Token]
                         ] = lambda lst: list(set(lst))
        self.dataset = self._sanitize_dataset(self.dataset, method)
        self._save_to_disk()
        return self

    def remove_duplicates(self):
        """extracts unique words from the dataset"""
        self.dataset_type = self.dataset_type + "_remove_duplicates"
        if self._try_load_from_disk():
            return self

        method: Callable[[list[Token]], list[Token]
                         ] = lambda lst: list(dict([(i.text, i) for i in lst]).values())
        self.dataset = self._sanitize_dataset(self.dataset, method)
        self._save_to_disk()
        return self

    def _convert_dataset(self, nlp: Language, dataset: Dataset):
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

            if item[1] == OFF:
                offensive_sentences.append(doc)
            else:
                not_offensive_sentences.append(doc)

        return {OFF: offensive_sentences, NOT: not_offensive_sentences}

    def _sanitize_dataset(self, dataset, sanitize_func):
        new_dict = {}
        for key in dataset.keys():
            lst = []
            for value in dataset[key]:
                lst.append(
                    self.nlp(" ".join([x.text for x in sanitize_func(value)])))
            new_dict[key] = lst
        return new_dict

    def _initial_sync_with_disk(self):
        disk_path = self.folder_path + self.dataset_type + ".pkl"

        try:
            self.dataset = self.storage.load_from_disk(disk_path)
        except FileNotFoundError:
            print(
                "No tokenized dataset on disk...\nTokenizing dataset using spacy and saving to disk...")
            self.dataset = self._convert_dataset(
                self.nlp, self.datasets[self.dataset_type])
            self.storage.save_to_disk(
                self.dataset, self.folder_path, disk_path)
            print("Succesfully tokenized dataset and saved to disk!")

    def _try_load_from_disk(self):
        disk_path = self.folder_path + self.dataset_type + ".pkl"

        try:
            self.dataset = self.storage.load_from_disk(disk_path)
            print("Found tokenized dataset on disk of type:", self.dataset_type)
            return True
        except FileNotFoundError:
            print("No tokenized dataset on disk of type:", self.dataset_type)
            return False

    def _save_to_disk(self):
        disk_path = self.folder_path + self.dataset_type + ".pkl"
        self.storage.save_to_disk(self.dataset, self.folder_path, disk_path)

    def get_all_attributes(self):
        return ["remove_dots", "remove_stop_words", "lowercase", "lemmatize", "remove_duplicates"]
