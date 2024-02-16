import random

from datasets import DatasetDict, Dataset
from models.ml_algorithm import MLAlgorithm


class BaselineRandom(MLAlgorithm):
    def __init__(self, dataset: DatasetDict) -> None:
        super().__init__(dataset)
        self.classes: list = list(set(self.dataset["label"]))
        self.comments: Dataset = dataset["text"]

    def train(self):
        pass

    def test(self, test_dataset_text: list | None):
        answer = []

        for _ in range(len(self.comments)):
            answer.append(random.choice(self.classes))

        return answer

    def __str__(self):
        return "Baseline Random"
