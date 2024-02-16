import math
import utils

from datasets import DatasetDict
from models.ml_algorithm import MLAlgorithm
from storage_manager import StorageManager
from data import TrainData


class NaiveBayes(MLAlgorithm):
    def __init__(self, dataset: DatasetDict) -> None:
        super().__init__(dataset)
        # base chance based on the split in classes in the dataset
        self.logprior = {}
        # Chance for each word to belong to each class
        self.loglikelihood = {}

        # amount of instances aka. comments/sentences in the dataset
        self.n_instances = len(dataset["text"])

        # creates a list of all words in the dataset after sentences have been sanitized
        self.vocabulary = utils.flatten([utils.sanitize(comment)
                                         for comment in self.dataset["text"]])

        self.sm = StorageManager(
            TrainData(str(self), (self.logprior, self.loglikelihood, self.vocabulary)))

    def train(self):
        for c in self.classes:
            # amount of instances with this class
            n_classes = self.dataset["label"].count(c)
            # it gives a base chance for it being NOT or OFF based on the split in the dataset
            self.logprior[c] = math.log10(n_classes / self.n_instances)

            words_in_class = utils.extract_words_from_label(self.dataset, c)
            n_words = count_words(words_in_class, self.vocabulary)

            for word in self.vocabulary:
                count = words_in_class[word] if word in words_in_class else 0

                # compute the likelihood of this word being generated from this class based on
                # the amount of the word used in the class compared to the total amount of
                # words used in the class.
                self.loglikelihood[(word, c)] = math.log10(
                    (count + 1) / (n_words - count))
        self.sm.store_data()

    def test(self, test_dataset_text):
        result = []
        for i in range(2):
            try:
                logprior, loglikelihood, _ = self.sm.load_data()
                print("Found Naive Bayes training data!")
                break
            except FileNotFoundError:
                # exit the program if FileNotFoundError happens more than once
                if i == 1:
                    print("ERROR: terminating...")
                    exit()
                print("Naive Bayes training data not found:\nInitializing training...")
                self.train()

        for test in test_dataset_text:
            result.append(find_class(test, self.classes,
                          logprior=logprior, loglikelihood=loglikelihood))
        return result

    def __str__(self) -> str:
        return "naive-bayes"


def find_class(test_instance: str, classes: list, logprior: dict, loglikelihood: dict):
    sum = {}
    test_instance = utils.sanitize(test_instance)
    for c in classes:
        sum[c] = logprior[c]
        for word in test_instance:
            try:
                sum[c] += loglikelihood[(word, c)]
            except KeyError:
                continue
    return utils.get_max_value_key(sum)


def count_words(words: dict, vocabulary: list):
    sum = 0
    for word in vocabulary:
        if word in words:
            sum += (words[word] + 1)
        else:
            sum += 1
    return sum
