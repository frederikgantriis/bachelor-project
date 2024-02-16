import pickle
import os
from data import Data, TrainData, TestData
import pandas


class StorageManager(object):
    """Stores and loads your train and test data.
    data_type: must be either 'train' or 'test
    """

    def __init__(self, data: Data) -> None:
        if not isinstance(data, Data):
            raise ValueError("data must be either TrainData or TestData")

        self.data = data
        self.data_type = "train" if isinstance(data, TrainData) else "test"

    def store_data(self):
        """store data in storage_manager_data/<data_type>/<key>.<file_extension>"""
        if isinstance(self.data, TrainData):
            store_train_data(self.data)
            return

        store_test_data(self.data)

    def load_data(self):
        """loads and returns data

        raises FileNotFoundError
        """
        # with open(
        #     "storage_manager_data/" + self.data_type + "/" + self.key + ".pkl", "rb"
        # ) as f:
        #     return pickle.load(f)
        return load_test_data(self.data)


def store_train_data(data: TrainData):
    if not os.path.exists("storage_manager_data/" + self.data_type):
        os.makedirs("storage_manager_data/" + self.data_type)

    with open(
        "storage_manager_data/" + self.data_type + "/" + self.key + ".pkl", "wb"
    ) as f:
        pickle.dump(self.data, f)


def store_test_data(data: TestData):
    data_frame: pandas.DataFrame = data.as_data_frame()
    if os.path.exists("data/" + data.model_name + ".csv"):
        # append to existing file
        data_frame.to_csv("data/" + data.model_name +
                          ".csv", mode="a", header=False)
        return

    data_frame.to_csv("data/" + data.model_name + ".csv")


def load_test_data(data: TestData):
    return pandas.read_csv("data/" + data.model_name + ".csv")
