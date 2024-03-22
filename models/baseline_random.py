import random
from typing import Literal

from models.ml_algorithm import MLAlgorithm
from datasets import DatasetDict
from constants import OFF, NOT


class BaselineRandom(MLAlgorithm):  # pragma: no cover
    def __init__(self, dataset: DatasetDict) -> None:
        super().__init__(dataset, "baseline-random")

    def train(self):
        pass

    def test(self, test_dataset_text: list | None):
        answer = []

        for _ in range(len(test_dataset_text)):
            answer.append(random.choice([OFF, NOT]))

        return answer
