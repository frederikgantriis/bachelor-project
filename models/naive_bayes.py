import math
import utils

from data_parser import Dataset
from models.ml_algorithm import MLAlgorithm
from data_storage import TrainData
from constants import OFF, NOT
from spacy.tokens import Token, Doc


class NaiveBayes(MLAlgorithm):  # pragma: no cover
    def __init__(self, dataset: Dataset) -> None:  # pragma: no cover
        super().__init__(dataset)  # type: ignore
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

        self.train_data = TrainData("naive-bayes")

    def __str__(self) -> str:
        return "naive-bayes"

    def train(self):  # pragma: no cover
        for c in self.classes:  # type: ignore
            # amount of instances with this class
            n_classes = len(self.dataset)
            # it gives a base chance for it being NOT or OFF based on the split in the dataset
            self.logprior[c] = math.log10(n_classes / self.n_instances)

            words_in_class = utils.extract_words_from_comments(self.dataset[c])
            n_words = count_words(words_in_class, self.vocabulary)

            for word in self.vocabulary:
                count = words_in_class[word] if word in words_in_class else 0

                # compute the likelihood of this word being generated from this class based on
                # the amount of the word used in the class compared to the total amount of
                # words used in the class.
                self.loglikelihood[(word.text, c)] = math.log10(
                    (count + 1) / (n_words - count)
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
                print("Found Naive Bayes training data!")
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
                find_class(
                    comment,
                    list(self.classes),
                    self.logprior,
                    self.loglikelihood
                )
            )
        return result


def find_class(
    comment: Doc, classes: list, logprior: dict, loglikelihood: dict
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

def count_words(words: dict, vocabulary: list):  # pragma: no cover
    sum = 0
    for word in vocabulary:
        if word in words:
            sum += words[word] + 1
        else:
            sum += 1
    return sum
