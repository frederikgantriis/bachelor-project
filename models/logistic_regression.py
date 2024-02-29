import math
import trace
import pandas as pd
import numpy as np

from datasets import DatasetDict, Dataset
from models.ml_algorithm import MLAlgorithm


class LogisticRegression(MLAlgorithm):
    def __init__(self, dataset: DatasetDict) -> None:
        super().__init__(dataset)
        self.hateful_words: set = set(pd.read_csv(
            "./hurtlex_DA.tsv", sep="\t")["lemma"])

    def sigmoid(self, x: float):
        return 1 / (1 + math.e ** (-x))

    def is_hateful(self, word: str) -> int:
        return int(word.lower() in self.hateful_words)

    def crossentropy_loss(self, guess, expected):
        return -(expected * math.log(guess) + (1-expected) * math.log(1-guess))

    def gradident_descent(self, features, weights, guess, expected, trainingspeed):
        new_weights = [((guess-expected)*feature)*(-trainingspeed)
                       for feature in features]

        new_weights.append((guess-expected)*(-trainingspeed))

        result = [x + y for x, y in zip(weights, new_weights)]

        return result
