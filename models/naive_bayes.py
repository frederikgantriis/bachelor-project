import math
import utils

from datasets import DatasetDict
from models.ml_algoritmh import MlAlgorithm
from storage_manager import StorageManager


class NaiveBayes(MlAlgorithm):
    def __init__(self, dataset: DatasetDict) -> None:
        super().__init__(dataset)
        self.logprior = {}
        self.loglikelihood = {}

        # amount of documents aka. comments/sentences in the dataset
        self.n_documents = len(dataset["text"])

        # creates a list of all words in the dataset after sentences have been sanitized
        self.vocabulary = utils.flatten([utils.sanitize(comment)
                                         for comment in self.dataset["text"]])

        self.sm = StorageManager(
            "nb_data", (self.logprior, self.loglikelihood, self.vocabulary))

    def train(self):
        for c in self.classes:
            # amount of instances with this class
            n_classes = self.dataset["label"].count(c)

            self.logprior[c] = math.log10(n_classes / self.n_documents)

            words_in_class = utils.extract_words_from_label(self.dataset, c)
            n_words = count_words(words_in_class, self.vocabulary)

            for word in self.vocabulary:
                count = words_in_class[word] if word in words_in_class else 0

                # compute the likelihood of this word being generated from this class
                self.loglikelihood[(word, c)] = math.log10(
                    (count + 1) / (n_words - count))
        self.sm.store_train_data()

    def test(self, testdoc: str):
        sum = {}
        testdoc = utils.sanitize(testdoc)

        for i in range(2):
            try:
                logprior, loglikelihood, _ = self.sm.load_train_data()
                print("Found Naive Bayes training data!")
                break
            except FileNotFoundError:
                if i == 1:
                    print("ERROR: terminating...")
                    exit()
                print("Naive Bayes training data not found:\nInitializing training...")
                self.train()

        for c in self.classes:
            sum[c] = logprior[c]
            for word in testdoc:
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
