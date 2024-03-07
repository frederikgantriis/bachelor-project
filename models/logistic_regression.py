import math
import pandas as pd
import utils
import decimal

from numpy.random import permutation
from datasets import DatasetDict, Dataset
from models.ml_algorithm import MLAlgorithm
from constants import OFF, NOT
from sanitizer import Sanitizer


class LogisticRegression(MLAlgorithm):
    def __init__(self, dataset: DatasetDict) -> None:
        super().__init__(dataset)
        self.hateful_words: set = set(
            pd.read_csv("./hurtlex_DA.tsv", sep="\t")["lemma"]
        )
        self.comments = self.dataset["text"]
        self.expected = self.dataset["label"]

        self.comments = [
            Sanitizer(comment).sanitize_simple() for comment in self.dataset["text"]
        ]

        self.expected = [0 if label == OFF else 1 for label in self.expected]

    def sigmoid(self, x):

        x = decimal.Decimal(x)
        y = decimal.Decimal(math.e)
        z = y ** (-x)
        return float(1 / (1 + z))

    def is_hateful(self, word: str) -> int:
        return int(word.lower() in self.hateful_words)

    def crossentropy_loss(self, guess, expected):
        return -(expected * math.log(guess) + (1 - expected) * math.log(1 - guess))

    def gradient_descent(self, features, loss, trainingspeed):
        """Finds gradient vector and moves the opposite way

        Args:
            features (list[int]): List of features for the comment
            loss (float): A number giving value to how far the guess is from the right answer
            trainingspeed (float): Dictates how fast the weights change
        """
        new_weights = [(loss * feature) * (-trainingspeed) for feature in features]

        result = [x + y for x, y in zip(self.weights, new_weights)]

        self.weights = result
        self.bias_term += loss * (-trainingspeed)

    def train(self):
        """Resets weights and bias term, then train model on all comments in a random order"""
        self.weights = [0, 0]
        self.bias_term = 0
        for i in permutation(len(self.comments)):
            features = self.calculate_feature_amount(self.comments[i])
            vector_product = [x * y for x, y in zip(self.weights, features)]
            guess = self.sigmoid(sum(vector_product) + self.bias_term)
            self.gradient_descent(features, guess - self.expected[i], 0.1)

    def calculate_feature_amount(self, comment):
        """Assigns value to each feature based on comment then asignes their weight. Then normalises the output.

        Args:
            comment (list[str]): list of words in a specific comment

        Returns:
            list[int]: List of features amount ex. amount of hate words
        """
        features = [0, 0]

        for word in comment:
            if self.is_hateful(word):
                features[0] += 1
            else:
                features[1] += 1

        return features

    def test(self, test_dataset_text: list):
        result = []
        self.train()

        test_comments = [
            Sanitizer(comment).sanitize_simple() for comment in test_dataset_text
        ]
        for test in test_comments:
            features = self.calculate_feature_amount(test)
            vector_product = [x * y for x, y in zip(self.weights, features)]
            guess = self.sigmoid(sum(vector_product) + self.bias_term)

            result.append(OFF) if guess > 0.5 else result.append(NOT)

        return result

    def __str__(self) -> str:
        return "logistic-regression"
