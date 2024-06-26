from data_parser import Dataset
from constants import OFF, NOT


class MLAlgorithm(object):  # pragma: no cover
    def __init__(self, dataset: Dataset, model_name=None, variation_name=None) -> None:
        self.dataset = dataset.to_dict()
        # set of unique classes in the dataset (i.e in our case "OFF" & "NOT")
        self.classes = [OFF, NOT]

        self.name = (
            model_name if variation_name is None else model_name + "_" + variation_name
        )

    def train(self):
        """train the model and store the results"""
        raise NotImplementedError("This has not been implemented yet")

    def test(self, test_dataset_text: list):
        """run the test and return the result"""
        raise NotImplementedError("This has not been implemented yet")

    def __str__(self) -> str:
        return self.name
