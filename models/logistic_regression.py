import math
import pandas as pd
import utils
import decimal

from numpy.random import permutation
from datasets import DatasetDict, Dataset
from models.ml_algorithm import MLAlgorithm
from analytics.constants import OFF, NOT


class LogisticRegression(MLAlgorithm):
    def __init__(self, dataset: DatasetDict) -> None:
        super().__init__(dataset)
        self.hateful_words: set = set(pd.read_csv(
            "./hurtlex_DA.tsv", sep="\t")["lemma"])
        self.comments = self.dataset["text"]
        self.expected = self.dataset["label"]

        self.comments = [utils.sanitize(comment)
                         for comment in self.dataset["text"]]

        self.expected = [0 if label == OFF else 1 for label in self.expected]

    def sigmoid(self, x):

        x = decimal.Decimal(x)
        y = decimal.Decimal(math.e)
        z = y ** (-x)
        return float(1 / (1 + z))

    def is_hateful(self, word: str) -> int:
        return int(word.lower() in self.hateful_words)

    def crossentropy_loss(self, guess, expected):
        return -(expected * math.log(guess) + (1-expected) * math.log(1-guess))

    def gradident_descent(self, features, loss, trainingspeed):

        new_weights = [(loss*feature)*(-trainingspeed)
                       for feature in features]

        result = [x + y for x, y in zip(self.weights, new_weights)]

        self.weights = result
        self.bias_term += loss * (-trainingspeed)

    def train(self):
        """Resets weights and bias term, then train model on all comments in a random order
        """
        self.weights = [0, 0]
        self.bias_term = 0
        for i in permutation(len(self.comments)):
            (guess, features) = self.guess(self.comments[i])
            self.gradident_descent(features, guess - self.expected[i], 0.1)

    def guess(self, comment):
        """Assigns value to each feature based on comment then asignes their weight. Then normalises the output.

        Args:
            comment (list[str]): list of words in a specific comment

        Returns:
            float, list[int]: returns a number between 0 and 1, 0 indicating a hateful comment and 1 the opposite
        """
        features = [0, 0]

        for word in comment:
            if self.is_hateful(word):
                features[0] += 1
            else:
                features[1] += 1

        predict = [x * y for x, y in zip(self.weights, features)]

        return (self.sigmoid(sum(predict)+self.bias_term), features)

    def test(self, test_dataset_text: list):
        return super().test(test_dataset_text)
