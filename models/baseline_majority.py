from constants import NOT, OFF
from models.ml_algorithm import MLAlgorithm


class BaselineMajority(MLAlgorithm):
    def __init__(self, dataset):
        super().__init__(dataset, "baseline-majority")
        self.majority = None
        self.dataset = dataset

    def train(self):

        self.majority = (
            NOT
            if len(self.dataset.to_dict()[NOT]) > len(self.dataset.to_dict()[OFF])
            else OFF
        )

    def test(self, test_dataset_text):
        if self.majority is None:
            self.train()

        return [self.majority for _ in range(len(test_dataset_text))]
