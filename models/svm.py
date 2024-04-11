import array
import numpy
import pandas as pd

from numpy.random import permutation
from datasets import DatasetDict
from models.ml_algorithm import MLAlgorithm
from constants import OFF, NOT
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report, accuracy_score


class SVM(MLAlgorithm):
    def __init__(self, dataset: DatasetDict, variation_name=None) -> None:
        super().__init__(dataset, "svm", variation_name)

        self.hateful_words: set = set(
            pd.read_csv("./hurtlex_DA.tsv", sep="\t")["lemma"]
        )

        self.data_length = len(self.dataset[OFF]) + len(self.dataset[NOT])
        self.variation_name = ""
        self.bias_term = 0
        self.weights = [0, 0, 0]
        self.svm_model = LinearSVC(dual="auto")

        self.positive_words: set = set()
        pos_words = open("data/sentiment-lexicons/positive_words_da.txt", "r")
        while True:
            word = pos_words.readline()
            if not word:
                break
            self.positive_words.add(word[:-1])
        pos_words.close()

    def sigmoid(self, x):
        z = numpy.exp(-x)
        return float(1 / (1 + z))

    def is_hateful(self, word: str) -> bool:
        return word.lower() in self.hateful_words

    def is_positive(self, word: str) -> bool:
        return word.lower() in self.positive_words

    def train(self):
        # Assuming the structure is ['sentence', 'label'], where label is "OFF" or "NOT"
        for i in permutation(self.data_length):
            if i < len(self.dataset[OFF]):
                expected = 1
                comment = self.dataset[OFF][i]
            else:
                expected = 0
                comment = self.dataset[NOT][i - len(self.dataset[OFF])]

            features = self.calculate_feature_amount(comment)

        X = features.reshape(-1, 1)
        y = self.classes

        # Step 4: Train the SVM model
        self.svm_model.fit(X, y)

    def test(self, test_dataset_text):
        if self.svm_model is None:
            self.train()

        results = []
        self.train()

        for test in test_dataset_text:
            features = self.calculate_feature_amount(test)

            result = self.svm_model.predict(
                features.reshape(-1, 1))

            results.append("OFF" if "OFF" in result else "NOT")

        return results

    def calculate_feature_amount(self, comment):
        """Assigns value to each feature based on comment then asignes their weight. Then normalises the output.

        Args:
            comment (list[str]): list of words in a specific comment

        Returns:
            list[int]: List of features amount ex. amount of hate words
        """
        features = numpy.array([0, 0])

        for word in comment:
            if self.is_hateful(word.text):
                features[0] += 1
            elif self.is_positive(word.text):
                features[1] += 1

        return features
