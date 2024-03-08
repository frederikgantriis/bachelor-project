import math
import utils

from datasets import DatasetDict
from models.ml_algorithm import MLAlgorithm
from data import TrainData
from sanitizer import Sanitizer


class NaiveBayes(MLAlgorithm):  # pragma: no cover
    def __init__(self, dataset: DatasetDict) -> None:  # pragma: no cover
        super().__init__(dataset)  # type: ignore
        # base chance based on the split in classes in the dataset
        self.logprior = {}
        # Chance for each word to belong to each class
        self.loglikelihood = {}

        # amount of instances aka. comments/sentences in the dataset
        self.n_instances = len(dataset["text"])

        # creates a list of all words in the dataset after sentences have been sanitized
        self.vocabulary = utils.flatten(
            [Sanitizer(comment).sanitize_simple()
             for comment in self.dataset["text"]]
        )

        self.train_data = TrainData("naive-bayes")

    def __str__(self) -> str:
        return "naive-bayes"

    def train(self):  # pragma: no cover
        c: str
        for c in self.classes:  # type: ignore
            # amount of instances with this class
            n_classes = self.dataset["label"].count(c)
            # it gives a base chance for it being NOT or OFF based on the split in the dataset
            self.logprior[c] = math.log10(n_classes / self.n_instances)

            words_in_class = self.extract_words_from_label(self.dataset, c)
            n_words = count_words(words_in_class, self.vocabulary)

            for word in self.vocabulary:
                count = words_in_class[word] if word in words_in_class else 0

                # compute the likelihood of this word being generated from this class based on
                # the amount of the word used in the class compared to the total amount of
                # words used in the class.
                self.loglikelihood[(word, c)] = math.log10(
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

        for test in test_dataset_text:
            result.append(
                self.find_class(
                    test,
                    list(self.classes),
                )
            )
        return result

    def find_class(self, test_instance: str, classes: list):  # pragma: no cover
        sum = {}
        test_instance_list = Sanitizer(test_instance).sanitize_simple()
        for c in classes:
            sum[c] = self.logprior[c]
            for word in test_instance_list:
                try:
                    sum[c] += self.loglikelihood[(word, c)]
                except KeyError:
                    continue
        return utils.get_max_value_key(sum)

    def extract_words_from_label(dataset: DatasetDict, label: str):
        extracted_words = {}
        sentences = utils.extract_sentences_from_label(dataset, label)

        for s in sentences:
            s = Sanitizer(s).sanitize_simple()
            for word in s:
                if word not in extracted_words:
                    extracted_words[word] = 1
                else:
                    extracted_words[word] += 1

        return extracted_words


def count_words(words: dict, vocabulary: list):  # pragma: no cover
    sum = 0
    for word in vocabulary:
        if word in words:
            sum += words[word] + 1
        else:
            sum += 1
    return sum
