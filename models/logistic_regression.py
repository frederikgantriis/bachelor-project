import imp
import math
from operator import le
import numpy
import pandas as pd
import utils

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

    def sigmoid(self, x: float):
        return 1 / (1 + math.e ** (-x))

    def is_hateful(self, word: str) -> int:
        return int(word.lower() in self.hateful_words)

    def crossentropy_loss(self, guess, expected):
        return -(expected * math.log(guess) + (1-expected) * math.log(1-guess))

    def gradident_descent(self, features, weights, loss, trainingspeed):
        print(features)
        new_weights = [(loss*feature)*(-trainingspeed)
                       for feature in features]
     
        result = [x + y for x, y in zip(weights, new_weights)]

        return result

    def train(self):
        # b is the last weight
        weights = [0, 0, 0]
        
        # len(self.comments)
        for i in permutation(200):
            # print(weights)
            # b is the last feature
            features = [0, 0, 1]
            x = self.comments[i]

            for word in x:
                if self.is_hateful(word):
                    features[0] += 1
                else:

                    features[1] += 1
            predict = [x + y for x, y in zip(weights, features)]
            predict[-1] = 0
            weights = self.gradident_descent(features, weights, self.sigmoid(
                sum(predict)+weights[-1])-self.expected[i], 0.1)

    def test(self, test_dataset_text: list):
        return super().test(test_dataset_text)
