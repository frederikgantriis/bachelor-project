import pickle
import os


class StorageManager(object):
    def __init__(self, key: str, data) -> None:
        self.data = data
        self.key = key

    def store_train_data(self):
        """store data in storage_manager_data/<key>.pkl"""
        if not os.path.exists("storage_manager_data"):
            os.makedirs("storage_manager_data")

        with open('storage_manager_data/' + self.key + '.pkl', 'wb') as f:
            pickle.dump(self.data, f)

    def load_train_data(self):
        """loads and returns data

        raises FileNotFoundError
        """
        with open('storage_manager_data/' + self.key + '.pkl', 'rb') as f:
            return pickle.load(f)
