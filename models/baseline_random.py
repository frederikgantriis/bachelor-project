import random

from datasets import DatasetDict
from models.ml_algoritmh import MlAlgorithm


class BaselineRandom(MlAlgorithm):
    def __init__(self, dataset: DatasetDict) -> None:
        super().__init__(dataset)
        self.classes = set(self.dataset["label"])
        self.comments = dataset["text"]

    def test(self):
        answer = []

        for _ in range(len(self.comments)):
            answer.append(random.choice(self.classes))

        return answer
