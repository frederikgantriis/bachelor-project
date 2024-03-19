from pandas import concat
from data_storage import StatsData
from constants import OFF, NOT
from models.ml_algorithm import MLAlgorithm
from utils import clear
from constants import OFF, NOT


class Benchmarker(object):
    def __init__(self, models: list[MLAlgorithm], dataset) -> None:
        self.dataset = dataset
        self.dataset_labels = [OFF] * len(self.dataset.to_dict()[OFF]) + [NOT] * len(
            self.dataset.to_dict()[NOT]
        )
        self.models = models

    def f1_score(self, result_labels: list[str]) -> float:
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

        true_positives = self.calculate_true_positives(result_labels)
        false_positives = self.calculate_false_positives(result_labels)
        false_negatives = self.calculate_false_negatives(result_labels)

        return (2 * true_positives) / (
            (2 * true_positives) + false_positives + false_negatives
        )

    def calculate_precision(self, result_labels: list[str]):

        true_positives = self.calculate_true_positives(result_labels)

        return true_positives / (
            true_positives + self.calculate_true_negatives(result_labels)
        )

    def calculate_recall(self, result_labels: list[str]):
        true_positives = self.calculate_true_positives(result_labels)

        return true_positives / (
            true_positives + self.calculate_false_negatives(result_labels)
        )

    def calculate_accuracy(self, result_labels: list[str]):
        return (
            self.calculate_true_positives(result_labels)
            + self.calculate_true_negatives(result_labels)
        ) / len(self.dataset_labels)

    def calculate_false_positives(self, result_labels: list[str]):
        return self.count_true_labels(OFF, NOT, result_labels)

    def calculate_false_negatives(self, result_labels: list[str]):
        return self.count_true_labels(NOT, OFF, result_labels)

    def calculate_true_positives(self, result_labels: list[str]):
        return self.count_true_labels(OFF, OFF, result_labels)

    def calculate_true_negatives(self, result_labels: list[str]):
        return self.count_true_labels(NOT, NOT, result_labels)

    def count_true_labels(
        self,
        compare_result_label: str,
        compare_test_label: str,
        result_labels: list[str],
    ) -> int:
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

        for i in range(len(result_labels)):
            if (
                result_labels[i] == compare_result_label
                and self.dataset_labels[i] == compare_test_label
            ):
                counter += 1

        return counter

    def benchmark_models(self, repetitions: int):  # pragma: no cover
        """tests each model, saves the result to data/models/stats and returns a data-frame containing all test-results

        Args:
            models (list): list of all the models

        Returns:
            DataFrame: A DataFrame containing all the test-results from the different models
        """

        data_frame = None

        for model in self.models:

            stats_average = {
                "f1": 0,
                "accuracy": 0,
                "precision": 0,
                "recall": 0,
                "true_positives": 0,
                "false_positives": 0,
                "true_negatives": 0,
                "false_negatives": 0,
            }

            print(f"Testing model: {model}")
            for _ in range(repetitions):
                print(f"Repetition: {_ + 1}/{repetitions}", end="\r")

                result_labels = model.test(self.dataset.to_list())

                stats_average["f1"] += self.f1_score(result_labels)
                stats_average["accuracy"] += self.calculate_accuracy(result_labels)
                stats_average["precision"] += self.calculate_precision(result_labels)
                stats_average["recall"] += self.calculate_recall(result_labels)
                stats_average["true_positives"] += self.calculate_true_positives(
                    result_labels
                )
                stats_average["false_positives"] += self.calculate_false_positives(
                    result_labels
                )
                stats_average["true_negatives"] += self.calculate_true_negatives(
                    result_labels
                )
                stats_average["false_negatives"] += self.calculate_false_negatives(
                    result_labels
                )

            for key in stats_average:
                stats_average[key] = stats_average[key] / repetitions

            model_data = StatsData(
                str(model),
                f1=stats_average["f1"],
                accuracy=stats_average["accuracy"],
                precision=stats_average["precision"],
                recall=stats_average["recall"],
                true_positives=stats_average["true_positives"],
                false_positives=stats_average["false_positives"],
                true_negatives=stats_average["true_negatives"],
                false_negatives=stats_average["false_negatives"],
            )

            if data_frame is None:
                data_frame = model_data.as_data_frame()
            else:
                data_frame = concat(
                    [data_frame, model_data.as_data_frame()], ignore_index=True
                )

            model_data.save_to_disk()

            # Clear terminal
            clear()

        return data_frame
