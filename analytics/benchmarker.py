from pyexpat import model
from pandas import DataFrame, concat
from data import StatsData
from data_parser import get_test_dataset
from constants import OFF, NOT
from models.ml_algorithm import MLAlgorithm
from utils import clear, makedir
import matplotlib.pyplot as plt


class Benchmarker(object):
    def __init__(self, models: list[MLAlgorithm]) -> None:
        self.dataset = get_test_dataset()
        self.dataset_labels = self.dataset["label"]
        self.models = models
        self.benchmark = None

    def create_all_charts(self, repetitions: int):
        """Create all the charts for the models

        Args:
            models (list): list of all the models
        """

        self.create_bar_chart_f1(repetitions)
        self.create_bar_chart_accuracy(repetitions)
        self.create_bar_chart_precision(repetitions)
        self.create_bar_chart_recall(repetitions)
        self.create_diagram_repetition(repetitions)
        self.create_pie_chart(repetitions)


    def _get_benchmark(self, repetitions: int):
        if self.benchmark is None:
            self.benchmark = self.benchmark_models(repetitions)
        
        return self.benchmark

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
        return self.benchmark_models_with_index(repetitions, None)

    def benchmark_models_with_index(self, repetitions: int, model_index: int):  # pragma: no cover
        """tests each model, saves the result to data/models/stats and returns a data-frame containing all test-results

        Args:
            models (list): list of all the models

        Returns:
            DataFrame: A DataFrame containing all the test-results from the different models
        """

        if model_index is not None:
            models = [self.models[model_index]]
        else:
            models = self.models


        data_frame = None

        for model in models:

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

                result_labels = model.test(self.dataset["text"])

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

    def create_pie_chart(self, repetitions: int):
        """Create a pie chart of the average f1 score for each model

        Args:
            models (list): list of all the models
        """

        f1_scores = []
        model_names = []

        model_benchmark = self._get_benchmark(repetitions)

        for i in range(len(self.models)):
            true_positives = model_benchmark["true_positives"].values[i]
            false_positives = model_benchmark["false_positives"].values[i]
            true_negatives = model_benchmark["true_negatives"].values[i]
            false_negatives = model_benchmark["false_negatives"].values[i]

            df = DataFrame(
                {
                    "Amount": [
                        true_positives,
                        false_positives,
                        true_negatives,
                        false_negatives,
                    ],

                },
                index=[
                    "True Positives",
                    "False Positives",
                    "True Negatives",
                    "False Negatives",
                ],
            )

            df.plot.pie(
                subplots=True,
                figsize=(20, 10),
                autopct="%1.1f%%",
                legend=False,
                title=f"{model_benchmark['model_name'].values[i]}",
            )

            makedir(f"img/{model_benchmark['model_name'].values[i]}")
            plt.savefig(f"img/{model_benchmark['model_name'].values[i]}/pie_chart.png")

    def create_bar_chart_f1(self, repetitions: int):
        """Create a bar chart of the average f1 score for each model

        Args:
            models (list): list of all the models
        """

        f1_scores = []
        model_names = []
        model_benchmark = self._get_benchmark(repetitions)

        for i in range(len(self.models)):

            f1_scores.append(model_benchmark["f1"].values[i])
            model_names.append(model_benchmark["model_name"].values[i])

        df = DataFrame({"F1 Score": f1_scores}, index=model_names)

        df.plot(kind="bar", figsize=(20, 10), legend=False, title="F1 Score")

        makedir("img")
        plt.savefig("img/f1_score.png")

    def create_bar_chart_accuracy(self, repetitions: int):
        """Create a bar chart of the average accuracy for each model

        Args:
            models (list): list of all the models
        """

        accuracies = []
        model_names = []
        model_benchmark = self._get_benchmark(repetitions)

        for i in range(len(self.models)):

            accuracies.append(model_benchmark["accuracy"].values[i])
            model_names.append(model_benchmark["model_name"].values[i])

        df = DataFrame({"Accuracy": accuracies}, index=model_names)

        df.plot(kind="bar", figsize=(20, 10), legend=False, title="Accuracy")

        makedir("img")
        plt.savefig("img/accuracy.png")

    def create_bar_chart_precision(self, repetitions: int):
        """Create a bar chart of the average precision for each model

        Args:
            models (list): list of all the models
        """

        precisions = []
        model_names = []
        model_benchmark = self._get_benchmark(repetitions)

        for i in range(len(self.models)):

            precisions.append(model_benchmark["precision"].values[i])
            model_names.append(model_benchmark["model_name"].values[i])

        df = DataFrame({"Precision": precisions}, index=model_names)

        df.plot(kind="bar", figsize=(20, 10), legend=False, title="Precision")

        makedir("img")
        plt.savefig("img/precision.png")

    def create_bar_chart_recall(self, repetitions: int):
        """Create a bar chart of the average recall for each model

        Args:
            models (list): list of all the models
        """

        recalls = []
        model_names = []
        model_benchmark = self._get_benchmark(repetitions)

        for i in range(len(self.models)):

            recalls.append(model_benchmark["recall"].values[i])
            model_names.append(model_benchmark["model_name"].values[i])

        df = DataFrame({"Recall": recalls}, index=model_names)

        df.plot(kind="bar", figsize=(20, 10), legend=False, title="Recall")

        makedir("img")
        plt.savefig("img/recall.png")

    def create_diagram_repetition(self, repetitions: int):
        """Create a diagram of the average f1 score for each model

        Args:
            models (list): list of all the models
        """


        for i in range(len(self.models)):
            f1_scores = []
            accuracy = []
            precision = []
            recall = []


            for _ in range(repetitions):
                model_benchmark = self.benchmark_models_with_index(1, i)
                f1_scores.append(model_benchmark["f1"].values[0])
                accuracy.append(model_benchmark["accuracy"].values[0])
                precision.append(model_benchmark["precision"].values[0])
                recall.append(model_benchmark["recall"].values[0])


            df = DataFrame(
                {
                    "F1 Score": f1_scores,
                    "Accuracy": accuracy,
                    "Precision": precision,
                    "Recall": recall,
                },
                index=[i for i in range(repetitions)],
            )

            df.plot(
                kind="line",
                figsize=(20, 10),
                legend=True,
                title=f"{model_benchmark['model_name'].values[0]}",
            )

            makedir(f"img/{model_benchmark['model_name'].values[0]}")
            plt.savefig(f"img/{model_benchmark['model_name'].values[0]}/repetition.png")

