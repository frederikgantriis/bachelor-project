import datetime
import fcntl
from pandas import DataFrame, read_csv
from data_storage import StatsData
from data_parser import Datasets
from constants import *
from models.ml_algorithm import MLAlgorithm
from utils import makedir
import threading
import matplotlib.pyplot as plt


class Benchmarker:
    def __init__(self, models: list[(MLAlgorithm, Datasets)], repetitions: int) -> None:
        self.test_dataset = Datasets(TEST)
        self.dataset_labels = [OFF] * len(self.test_dataset.to_dict()[OFF]) + [
            NOT
        ] * len(self.test_dataset.to_dict()[NOT])
        self.models = models
        self.benchmark = None
        self.repetitions = repetitions
        self.metrics = [
            F1,
            ACCURACY,
            PRECISION,
            RECALL,
            TRUE_POSITIVES,
            FALSE_POSITIVES,
            TRUE_NEGATIVES,
            FALSE_NEGATIVES,
        ]
        self.data_frame = None

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
            return tp / (tp + fp) if tp + fp != 0 else 0
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

    def count_labels(
        self,
        compare_result_label: str,
        compare_test_label: str,
        result_labels: list[str],
    ) -> int:
        return sum(
            1
            for i in range(len(result_labels))
            if result_labels[i] == compare_result_label
            and self.dataset_labels[i] == compare_test_label
        )

    def benchmark_models(self, model_index=None, repetitions=None):
        if repetitions is None:
            repetitions = self.repetitions
        if self.benchmark is not None:
            return self.benchmark
        models = [self.models[model_index]
                  ] if model_index is not None else self.models

        data_frame = DataFrame()
        makedir("data/models/stats/latest_benchmark")

        filename = f"data/models/stats/latest_benchmark/{
            datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.csv"

        data_frame = DataFrame(
            {
                "model_name": [],
                F1: [],
                ACCURACY: [],
                PRECISION: [],
                RECALL: [],
                TRUE_POSITIVES: [],
                FALSE_POSITIVES: [],
                TRUE_NEGATIVES: [],
                FALSE_NEGATIVES: [],
                "timestamp": [],
            }
        )

        data_frame.to_csv(
            filename, header=True, index=False
        )  # Save the initial empty dataframe with headers

        def run_model(model, repetitions, filename):
            print(f"Running benchmark for {model[0].name}")
            stats_average = {metric: 0 for metric in self.metrics}

            for _ in range(repetitions):
                print(f"Repetition: {_ + 1}/{repetitions}")

                result_labels = model[0].test(model[1].to_list())
                for metric in stats_average.keys():
                    stats_average[metric] += self.calculate_metric(
                        result_labels, metric
                    )

            stats_average = {
                key: value / self.repetitions for key, value in stats_average.items()
            }

            model_data = StatsData(str(model[0]), **stats_average)

            data_frame = model_data.as_data_frame()

            model_data.save_to_disk()

            # append to latest_benchmark file
            with open(filename, "a") as f:
                # Lock the file before writing to it
                fcntl.flock(f, fcntl.LOCK_EX)
                if f.tell() == 0:  # Check if file is empty
                    f.write(
                        data_frame.to_csv(header=True, index=False)
                    )  # Write headers when file is empty
                else:
                    f.write("\n")
                    f.write(
                        data_frame.to_csv(header=False, index=False)
                    )  # Do not write headers when appending
                # Unlock the file after writing
                fcntl.flock(f, fcntl.LOCK_UN)

        threads = []
        for model in models:
            thread = threading.Thread(
                target=run_model, args=(model, repetitions, filename)
            )
            thread.start()
            threads.append(thread)

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Read the latest benchmark file
        data_frame = read_csv(filename)

        data_frame = data_frame.sort_values(by=[F1], ascending=True)

        print(data_frame)

        return data_frame

    def create_all_charts(self):
        # Only create the first 4 metrics (F1, Precision, Recall, Accuracy)
        for metric in self.metrics[:4]:
            self.create_bar_chart(metric)

        # self.create_diagram_repetition()
        # self.create_confusion_matrix()

    def create_bar_chart(self, metric: str):
        metrics = []
        model_names = []
        latest_benchmark = self._get_benchmark()

        for i in range(len(self.models)):
            metrics.append(latest_benchmark[metric].values[i])
            model_names.append(latest_benchmark["model_name"].values[i])

        df = DataFrame({metric.capitalize(): metrics}, index=model_names)

        ax = df.plot(
            kind="bar", figsize=(20, 10), legend=False, title=metric.capitalize()
        )

        # Rotate the x-axis labels for better readability
        plt.xticks(rotation=45, ha="right")

        makedir("img")
        plt.savefig(f"img/{metric}.png", bbox_inches="tight")
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
            plt.savefig(
                f"img/{model_benchmark['model_name'].values[0]}/repetition.png")
            plt.close()

    def create_confusion_matrix(self):
        for i in range(len(self.models)):
            model_benchmark = self._get_benchmark()
            result_labels = self.models[i][0].test(self.models[i][1].to_list())
            self.create_confusion_matrix_for_model(
                result_labels, model_benchmark["model_name"].values[i]
            )

    def create_confusion_matrix_for_model(
        self, result_labels: list[str], model_name: str
    ):
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
                plt.text(j, i, df.values[i, j],
                         ha="center", va="center", color="black")

        plt.xticks(range(2), df.columns)
        plt.yticks(range(2), df.index)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")

        makedir(f"img/{model_name}")
        plt.savefig(f"img/{model_name}/confusion_matrix.png")
        plt.close()

    def get_wrongly_classified(self):
        makedir("data/models/stats/wrongly-classified")

        for model, dataset in self.models:
            print(f"Getting wrongly classified for {model.name}", end="\r")

            result_labels = model.test(dataset.to_list())
            self.get_wrongly_classified_for_model(result_labels, model.name)

        self.get_common_wrongly_classified()

    def get_common_wrongly_classified(self):
        wrongly_classified = {}
        basic_train_dataset = Datasets(TRAIN).to_dict()
        words_set = set(
            word.text
            for data in basic_train_dataset[OFF] + basic_train_dataset[NOT]
            for word in data
        )

        for model, dataset in self.models:
            result_labels = model.test(dataset.to_list())

            for result_label, dataset_label, comment in zip(
                result_labels, self.dataset_labels, self.test_dataset.to_list()
            ):
                if result_label != dataset_label:
                    comment_words = [word.text for word in comment]
                    common_words = [
                        word for word in comment_words if word not in words_set
                    ]
                    if comment not in wrongly_classified:
                        wrongly_classified[comment] = [
                            dataset_label,
                            result_label,
                            1,
                            common_words,
                            [model.name],
                        ]
                    else:
                        wrongly_classified[comment][2] += 1
                        wrongly_classified[comment][4].append(model.name)

        wrongly_classified_list = [
            [k, *v] for k, v in wrongly_classified.items() if v[2] > 1
        ]
        df = DataFrame(
            wrongly_classified_list,
            columns=[
                "Comment",
                "Actual",
                "Predicted",
                "Amount of models",
                "Words not in train dataset",
                "Models",
            ],
        )
        df.to_csv(
            "data/models/stats/wrongly-classified/common-wrongly-classified.csv")

    def get_wrongly_classified_for_model(
        self, result_labels: list[str], model_name: str
    ):
        dataset = next(
            (d for m, d in self.models if m.name == model_name), None)
        if dataset:
            wrongly_classified = [
                [dataset_label, result_label, comment]
                for dataset_label, result_label, comment in zip(
                    self.dataset_labels, result_labels, self.test_dataset.to_list()
                )
                if result_label != dataset_label
            ]

            df = DataFrame(
                wrongly_classified, columns=["Actual", "Predicted", "Comment"]
            )
            df.to_csv(f"data/models/stats/wrongly-classified/{model_name}.csv")
