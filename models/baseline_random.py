import random

from datasets import DatasetDict, Dataset
import pytest
from models.ml_algorithm import MLAlgorithm


class BaselineRandom(MLAlgorithm):  # pragma: no cover
    def __init__(self, dataset) -> None:
        super().__init__(dataset)
        self.classes: list = self.dataset.keys()

    def train(self):
        pass

    def test(self, test_dataset_text: list | None):
        answer = []

        for _ in range(len(test_dataset_text)):
            answer.append(random.choice(self.classes))

        return answer

    def __str__(self):
        return "Baseline Random"
