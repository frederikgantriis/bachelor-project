import math

import utils

from data_parser import Dataset
from models.ml_algorithm import MLAlgorithm
from data_storage import TrainData
from constants import OFF, NOT
from spacy.tokens import Token, Doc


class NaiveBayes(MLAlgorithm):  # pragma: no cover
    def __init__(
        self,
        dataset: Dataset,
        model_name="naive-bayes",
        variation_name=None,
        k_factor: float = 1,
    ) -> None:  # pragma: no cover
        if k_factor != 1:
            if variation_name is None:
                variation_name = f"add-k-{k_factor}"
            else:
                variation_name = f"add-k-{k_factor}_" + variation_name

        super().__init__(dataset, model_name, variation_name)  # type: ignore
        # base chance based on the split in classes in the dataset
        self.logprior = {}
        # Chance for each word to belong to each class
        self.loglikelihood = {}

        # amount of instances aka. comments/sentences in the dataset
        self.n_instances = len(self.dataset[OFF]) + len(self.dataset[NOT])

        # creates a set of words in the dataset
        self.vocabulary: set[Token] = set()
        for comment in dataset.to_list():
            self.vocabulary.update(comment)

        self.train_data = TrainData(self.name)

        self.k_factor = k_factor

    def train(self):  # pragma: no cover
        for c in self.classes:  # type: ignore
            # amount of instances with this class
            n_classes = len(self.dataset[c])
            # it gives a base chance for it being NOT or OFF based on the split in the dataset
            self.logprior[c] = math.log10(n_classes / self.n_instances)

            words_in_class = utils.extract_words_from_comments(self.dataset[c])
            n_words = self._count_words(words_in_class, self.vocabulary)

            for word in self.vocabulary:
                count = words_in_class[word.text] if word.text in words_in_class else 0

                # compute the likelihood of this word being generated from this class based on
                # the amount of the word used in the class compared to the total amount of
                # words used in the class.
                self.loglikelihood[(word.text, c)] = math.log10(
                    (count + self.k_factor) / (n_words)
                )

            # update the train data parameters
            self.train_data.parameters = self.logprior, self.loglikelihood, self.classes
            # save the train data to disk
            self.train_data.save_to_disk()

    def test(self, test_dataset_text):
        result = []

        for i in range(2):  # pragma: no cover
            try:
                self.logprior, self.loglikelihood, _ = self.train_data.load_from_disk()
                break
            except FileNotFoundError:
                # exit the program if FileNotFoundError happens more than once
                if i == 1:
                    print("ERROR: terminating...")
                    exit()
                print("Naive Bayes training data not found:\nInitializing training...")
                self.train()

        for comment in test_dataset_text:
            result.append(
                self._find_class(
                    comment, list(self.classes), self.logprior, self.loglikelihood
                )
            )
        return result

    def _find_class(
        self, comment: Doc, classes: list, logprior: dict, loglikelihood: dict
    ):  # pragma: no cover
        sum = {}
        for c in classes:
            sum[c] = logprior[c]
            for word in comment:
                try:
                    sum[c] += loglikelihood[(word.text, c)]
                except KeyError:
                    continue
        return utils.get_max_value_key(sum)

    def _count_words(self, words: dict, vocabulary: list):  # pragma: no cover
        sum = 0
        for word in vocabulary:
            if word.text in words.keys():
                sum += words[word.text] + self.k_factor
            else:
                sum += self.k_factor
        return sum
