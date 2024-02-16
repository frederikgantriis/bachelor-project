import pickle
import os
from data import Data, TrainData, StatsData
import pandas


class StorageManager(object):
    """Stores and loads your train and stats data."""

    def __init__(self) -> None:
        pass

    def store_data(self, data: Data):
        """Stores data in a file

        Args:
            data (Data): data to store
        """
        store_train_data(data) if isinstance(
            data, TrainData) else store_test_data(data)

    def load_data(self, model_name: str, data_type: str):
        """loads data from file

        Args:
            model_name (str): name of model to load data from
            data_type (str): must be either 'train' or 'stats'
        """
        if data_type != "train" and data_type != "stats":
            raise ValueError(
                "data_type must be either 'train' or 'stats', but was " + data_type)

        return load_train_data(model_name) if data_type == "train" else load_stats_data(model_name)


def store_train_data(data: TrainData):
    folder_path = "data/models/train"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    with open(folder_path + "/" + data.model_name + ".pkl", "wb") as f:
        pickle.dump(data.data, f)


def store_test_data(data: StatsData):
    folder_path = "data/models/stats"
    data_frame: pandas.DataFrame = data.as_data_frame()
    if os.path.exists(folder_path + "/" + data.model_name + ".csv"):
        # append to existing file
        data_frame.to_csv(folder_path + "/" + data.model_name +
                          ".csv", mode="a", header=False)
    else:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        data_frame.to_csv(folder_path + "/" + data.model_name + ".csv")


def load_train_data(file_name: str):
    folder_path = "data/models/train"
    with open(folder_path + "/" + file_name + ".pkl", "rb") as f:
        return pickle.load(f)


def load_stats_data(file_name: str):
    folder_path = "data/models/stats"
    return pandas.read_csv(folder_path + "/" + file_name + ".csv")
