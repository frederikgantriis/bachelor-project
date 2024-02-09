import math
import utils
import pickle

from datasets import DatasetDict
from models.ml_algoritmh import MlAlgorithm


class NaiveBayes(MlAlgorithm):
    def __init__(self, dataset: DatasetDict) -> None:
        super().__init__(dataset)
        self.logprior = {}
        self.loglikelihood = {}

        # set of unique classes in the dataset (i.e in our case "OFF" & "NOT")
        self.classes = set(dataset["label"])

        # amount of documents aka. comments/sentences in the dataset
        self.n_documents = len(dataset["text"])

        # creates a list of all words in the dataset after sentences have been sanitized
        self.vocabulary = utils.flatten([utils.sanitize(comment)
                                         for comment in self.dataset["text"]])

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

    def test(self, testdoc: str):
        sum = {}
        testdoc = utils.sanitize(testdoc)
        for c in self.classes:
            sum[c] = self.logprior[c]
            for word in testdoc:
                try:
                    sum[c] += self.loglikelihood[(word, c)]
                except KeyError:
                    continue
        return utils.get_max_value_key(sum)

    def store_train_data(self):
        with open('computed_data/nb_lp.pkl', 'wb') as f:
            pickle.dump(self.logprior, f)
        with open('computed_data/nb_ll.pkl', 'wb') as f:
            pickle.dump(self.loglikelihood, f)
        with open('computed_data/nb_v.pkl', 'wb') as f:
            pickle.dump(self.vocabulary, f)

    def load_train_data(self):
        with open('computed_data/nb_lp.pkl', 'rb') as f:
            self.logprior = pickle.load(f)
        with open('computed_data/nb_ll.pkl', 'rb') as f:
            self.loglikelihood = pickle.load(f)
        with open('computed_data/nb_v.pkl', 'rb') as f:
            self.vocabulary = pickle.load(f)


def count_words(words: dict, vocabulary: list):
    sum = 0
    for word in vocabulary:
        if word in words:
            sum += (words[word] + 1)
        else:
            sum += 1

    return sum
