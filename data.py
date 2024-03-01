from datetime import datetime
import os
import pickle
from pandas import DataFrame
import pandas


class Data(object):
    def __init__(self) -> None:
        self.timestamp = datetime.now()

    def save_to_disk(self):
        raise NotImplementedError("not implemented")

    def load_from_disk(self):
        raise NotImplementedError("not implemented")


class StatsData(Data):
    def __init__(self, model_name: str, **kwargs):
        """_summary_

        Args:
            model_name (str): _description_

        Keyword Args:
            f1 (float): _description_
            accuracy (float): _description_
            precision (float): _description_
            recall (float): _description_
            true_positives (float): _description_
            false_positives (float): _description_
            true_negatives (float): _description_
            false_negatives (float): _description_
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


class TrainData(Data):
    def __init__(self, model_name: str, *parameters) -> None:
        super().__init__()
        self.model_name = model_name
        self.folder_path = "data/models/train/"
        self.disk_path = self.folder_path + self.model_name + ".pkl"

        self.parameters = parameters

    def save_to_disk(self):
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)
        with open(self.disk_path, "wb") as f:
            pickle.dump(self.parameters, f)

    def load_from_disk(self):
        with open(self.disk_path, "rb") as f:
            return pickle.load(f)


class SanitizedTextData(Data):
    def __init__(self, method: str, text_data: list[str]) -> None:
        super().__init__()
        self.folder_path = "data/text/"
        self.disk_path = self.folder_path + method + ".pkl"

        self.method = method
        self.text_data = text_data

    def save_to_disk(self):
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)
        with open(self.disk_path, "wb") as f:
            pickle.dump(self.text_data, f)

    def load_from_disk(self):
        with open(self.disk_path, "rb") as f:
            return pickle.load(f)
