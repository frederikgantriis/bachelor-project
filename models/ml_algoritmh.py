from datasets import DatasetDict


class MlAlgorithm(object):
    def __init__(self, dataset: DatasetDict) -> None:
        self.dataset = dataset

    def train(self):
        """train the model and store the results"""
        raise NotImplementedError("This has not been implemented yet")

    def test(self):
        """run the test and return the result"""
        raise NotImplementedError("This has not been implemented yet")
