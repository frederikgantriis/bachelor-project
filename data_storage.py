from datetime import datetime
import os
import pickle
from pandas import DataFrame
import pandas


class DataStorage(object):
    def __init__(self) -> None:
        self.timestamp = datetime.now()

    def save_to_disk(self, data, folder_path, disk_path):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        with open(disk_path, "wb") as f:
            pickle.dump(data, f)

    def load_from_disk(self, disk_path):
        with open(disk_path, "rb") as f:
            return pickle.load(f)


class StatsData(DataStorage):
    def __init__(self, model_name: str, **kwargs):
        """keyword args is optional

        Args:
            model_name (str): model name

        Keyword Args:
            f1 (float): f1 score
            accuracy (float): accuracy
            precision (float): precision
            recall (float): recall
            true_positives (float): true positives
            false_positives (float): false positives
            true_negatives (float): true negatives
            false_negatives (float): false negatives
        """
        super().__init__()
        self.model_name = model_name
        self.folder_path = "data/models/stats/"
        self.disk_path = self.folder_path + self.model_name + ".csv"

        self.f1 = kwargs.get("f1", None)
        self.accuracy = kwargs.get("accuracy", None)
        self.precision = kwargs.get("precision", None)
        self.recall = kwargs.get("recall", None)
        self.true_positives = kwargs.get("true_positives", None)
        self.false_positives = kwargs.get("false_positives", None)
        self.true_negatives = kwargs.get("true_negatives", None)
        self.false_negatives = kwargs.get("false_negatives", None)

    def as_data_frame(self):
        return DataFrame({
            "model_name": [self.model_name],
            "f1": [self.f1],
            "accuracy": [self.accuracy],
            "precision": [self.precision],
            "recall": [self.recall],
            "true_positives": [self.true_positives],
            "false_positives": [self.false_positives],
            "true_negatives": [self.true_negatives],
            "false_negatives": [self.false_negatives],
            "timestamp": [self.timestamp],
        })

    def save_to_disk(self):
        df = self.as_data_frame()
        if os.path.exists(self.disk_path):
            # append to existing file
            df.to_csv(self.disk_path, mode="a", header=False)
        else:
            if not os.path.exists(self.folder_path):
                os.makedirs(self.folder_path)
            df.to_csv(self.disk_path)

    def load_from_disk(self):
        if not os.path.exists(self.disk_path):
            raise FileNotFoundError(f"File not found: {self.disk_path}")
        return pandas.read_csv(self.disk_path)


class TrainData(DataStorage):
    def __init__(self, model_name: str, *parameters) -> None:
        super().__init__()
        self.model_name = model_name
        self.folder_path = "data/models/train/"
        self.disk_path = self.folder_path + self.model_name + ".pkl"

        self.parameters = parameters

    def save_to_disk(self):
        super().save_to_disk(self.parameters, self.folder_path, self.disk_path)

    def load_from_disk(self):
        return super().load_from_disk(self.disk_path)
