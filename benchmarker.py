from pandas import DataFrame, concat
from data_storage import StatsData
from data_parser import Datasets
from constants import *
from models.ml_algorithm import MLAlgorithm
from utils import clear, makedir
import matplotlib.pyplot as plt


class Benchmarker:
    def __init__(self, models: list[(MLAlgorithm, Datasets)], repetitions: int) -> None:
        test_dataset = Datasets(TEST)
        self.dataset_labels = [OFF] * len(test_dataset.to_dict()[OFF]) + [NOT] * len(test_dataset.to_dict()[NOT])
        self.models = models
        self.benchmark = None
        self.repetitions = repetitions
        self.metrics = [F1, PRECISION, RECALL, ACCURACY, TRUE_POSITIVES, FALSE_POSITIVES, TRUE_NEGATIVES, FALSE_NEGATIVES]

    def _get_benchmark(self):
        if self.benchmark is None:
            self.benchmark = self.benchmark_models(None)
        return self.benchmark

    def calculate_metric(self, result_labels: list[str], metric: str):
        tp = self.count_labels(OFF, OFF, result_labels)
        tn = self.count_labels(NOT, NOT, result_labels)
        fp = self.count_labels(OFF, NOT, result_labels)
        fn = self.count_labels(NOT, OFF, result_labels)

        if metric == F1:
            return (2 * tp) / ((2 * tp) + fp + fn)
        elif metric == PRECISION: 
            return tp / (tp + tn)
        elif metric == RECALL:
            return tp / (tp + fn)
        elif metric == ACCURACY:
            return (tp + tn) / len(self.dataset_labels)
        elif metric == TRUE_POSITIVES:
            return tp
        elif metric == FALSE_POSITIVES:
            return fp
        elif metric == TRUE_NEGATIVES:
            return tn
        elif metric == FALSE_NEGATIVES:
            return fn

    def count_labels(self, compare_result_label: str, compare_test_label: str, result_labels: list[str]) -> int:
        return sum(1 for i in range(len(result_labels)) if result_labels[i] == compare_result_label and self.dataset_labels[i] == compare_test_label)

    def benchmark_models(self, model_index=None, repetitions=None):
        if repetitions is None:
            repetitions = self.repetitions
        if self.benchmark is not None:
            return self.benchmark
        models = [self.models[model_index]] if model_index is not None else self.models
        data_frame = None

        for model in models:
            stats_average = {metric: 0 for metric in self.metrics}
            # print "Running benchmark for model_name" but only update model_name
            print(f"Running benchmark for {model[0].name}", end="\r")

            for _ in range(repetitions):
                print(f"Repetition: {_ + 1}/{repetitions}", end="\r")

                result_labels = model[0].test(model[1].to_list())
                for metric in stats_average.keys():
                    stats_average[metric] += self.calculate_metric(result_labels, metric)

            stats_average = {key: value / self.repetitions for key, value in stats_average.items()}

            model_data = StatsData(str(model[0]), **stats_average)

            data_frame = concat([data_frame, model_data.as_data_frame()], ignore_index=True) if data_frame is not None else model_data.as_data_frame()

            model_data.save_to_disk()
            clear()

        data_frame.to_csv("data/models/stats/latest_benchmark.csv")
        return data_frame
    
    def create_all_charts(self):
        # Only create the first 4 metrics (F1, Precision, Recall, Accuracy)
        for metric in self.metrics[:4]:
            self.create_bar_chart(metric)

        self.create_diagram_repetition()
        self.create_confusion_matrix()

    def create_bar_chart(self, metric: str):
        metrics = []
        model_names = []
        latest_benchmark = self._get_benchmark()

        for i in range(len(self.models)):
            metrics.append(latest_benchmark[metric].values[i])
            model_names.append(latest_benchmark["model_name"].values[i])

        df = DataFrame({metric.capitalize(): metrics}, index=model_names)

        df.plot(kind="bar", figsize=(20, 10), legend=False, title=metric.capitalize())

        makedir("img")
        plt.savefig(f"img/{metric}.png")
        plt.close()

    def create_diagram_repetition(self):
        """Create a diagram of the average f1 score for each model

        Args:
            models (list): list of all the models
        """

        for i in range(len(self.models)):
            # Only create the first 4 metrics (F1, Precision, Recall, Accuracy)
            metrics = {metric: [] for metric in self.metrics[:4]}

            for _ in range(self.repetitions):
                model_benchmark = self.benchmark_models(1, i)

                for metric in metrics.keys():
                    metrics[metric].append(model_benchmark[metric].values[0])

            df = DataFrame(metrics, index=[i for i in range(self.repetitions)])

            df.plot(
                kind="line",
                figsize=(20, 10),
                legend=True,
                title=f"{model_benchmark['model_name'].values[0]}",
            )

            makedir(f"img/{model_benchmark['model_name'].values[0]}")
            plt.savefig(f"img/{model_benchmark['model_name'].values[0]}/repetition.png")
            plt.close()

    def create_confusion_matrix(self):
        for i in range(len(self.models)):
            model_benchmark = self._get_benchmark()
            result_labels = self.models[i][0].test(self.models[i][1].to_list())
            self.create_confusion_matrix_for_model(result_labels, model_benchmark["model_name"].values[i])

    def create_confusion_matrix_for_model(self, result_labels: list[str], model_name: str):
        tp = self.count_labels(OFF, OFF, result_labels)
        tn = self.count_labels(NOT, NOT, result_labels)
        fp = self.count_labels(OFF, NOT, result_labels)
        fn = self.count_labels(NOT, OFF, result_labels)

        df = DataFrame(
            {
                "Predicted Offensive": [tp, fp],
                "Predicted Not Offensive": [fn, tn],
            },
            index=["Actual Offensive", "Actual Not Offensive"],
        )

        # plot the confusion matrix in a matrix form
        plt.figure(figsize=(10, 7))
        plt.title(f"{model_name} Confusion Matrix")
        plt.imshow(df, cmap="Blues", interpolation="nearest")
        plt.colorbar()

        for i in range(2):
            for j in range(2):
                plt.text(j, i, df.values[i, j], ha="center", va="center", color="black")

        plt.xticks(range(2), df.columns)
        plt.yticks(range(2), df.index)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")

        makedir(f"img/{model_name}")
        plt.savefig(f"img/{model_name}/confusion_matrix.png")
        plt.close()
