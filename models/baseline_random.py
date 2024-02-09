import random

from datasets import DatasetDict
from models.ml_algoritmh import MLAlgorithm


class BaselineRandom(MLAlgorithm):
    def __init__(self, dataset: DatasetDict) -> None:
        super().__init__(dataset)
        self.classes = list(set(self.dataset["label"]))
        self.comments = dataset["text"]

    def test(self):
        answer = []

        for _ in range(len(self.comments)):
            answer.append(random.choice(self.classes))

        return answer
