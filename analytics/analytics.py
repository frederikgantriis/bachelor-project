from datasets import DatasetDict

class Analyzer(object):
    def __init__(self, result_labels: list, dataset: DatasetDict) -> None:
        self.result_labels = result_labels
        self.dataset = dataset
        self.dataset_labels = dataset["label"]
        

    def f1_score(self) -> float:
        """Get the f1_score based labels from model and labels from the test dataset

        f1 formula: 2tp / 2tp + fp + fn

        Parameters
        ----------
        result_labels : list
            A list of labels given by a model

        Returns
        -------
        int
            the f1_score
        """

        true_positives = self.calculate_true_positives()
        false_positives = self.calculate_false_positives()
        false_negatives = self.calculate_false_negatives()

        return (2 * true_positives) / ((2 * true_positives) + false_positives + false_negatives)


    def calculate_precision(self):

        true_positives = self.calculate_true_positives()

        return true_positives / (true_positives + self.calculate_true_negatives())


    def calculate_recall(self):
        true_positives = self.calculate_true_positives()

        return true_positives / (true_positives + self.calculate_false_negatives())


    def calculate_false_positives(self):
        return self.count_true_labels("OFF", "NOT")


    def calculate_false_negatives(self):
        return self.count_true_labels("NOT", "OFF")


    def calculate_true_positives(self):
        return self.count_true_labels("OFF", "OFF")


    def calculate_true_negatives(self):
        return self.count_true_labels("NOT", "NOT")


    def count_true_labels(self, compare_result_label: str, compare_test_label: str) -> int:
        """counts amount of labels from model and test dataset, with a given compare_label

        Parameters
        ----------
        compare_result_label : str
            A label (either NOT or OFF) to compare result_labels with
        compare_test_label : str
            A label (either NOT or OFF) to compare dataset_labels with

        Returns
        -------
        int
            A amount where both labels where true (used for true_positives, false_negatives and so on)
        """
        counter = 0

        for i in range(len(self.result_labels)):
            if self.result_labels[i] == compare_result_label and self.dataset_labels[i] == compare_test_label:
                counter += 1

        return counter