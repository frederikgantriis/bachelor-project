from datetime import datetime
from pandas import DataFrame


class Data(object):
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.timestamp = datetime.now()


class StatsData(Data):
    def __init__(self, model_name: str, f1: float, accuracy: int, precision: float, recall: float, true_positives: int, false_positives: int, false_negatives: int, true_negatives: int) -> None:
        super().__init__(model_name)
        self.f1 = f1
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
        self.true_positives = true_positives
        self.false_positives = false_positives
        self.true_negatives = true_negatives
        self.false_negatives = false_negatives

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


class TrainData(Data):
    def __init__(self, model_name: str, data: tuple) -> None:
        super().__init__(model_name)
        self.data = data
