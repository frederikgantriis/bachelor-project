from datasets import DatasetDict


class MLAlgorithm(object):
    def __init__(self, dataset: DatasetDict) -> None:
        self.dataset = dataset
        # set of unique classes in the dataset (i.e in our case "OFF" & "NOT")
        self.classes = set(dataset["label"])

    def train(self):
        """train the model and store the results"""
        raise NotImplementedError("This has not been implemented yet")

    def test(self, test_dataset_text: list):
        """run the test and return the result"""
        raise NotImplementedError("This has not been implemented yet")

    def to_string(self):
        """return the name of the model"""
        raise NotImplementedError("This has not been implemented yet")
