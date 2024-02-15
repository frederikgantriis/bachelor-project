import pickle
import os


class StorageManager(object):
    """Stores and loads your train and test data.
    data_type: must be either 'train' or 'test
    """

    def __init__(self, key: str, data_type: str, data) -> None:
        self.data = data
        self.key = key
        self.data_type = data_type

        if data_type != 'train' and data_type != 'test':
            raise ValueError(
                'data_type argument must be either "train" or "test')

        # TODO: change test data extension to correct type
        self.ext = '.pkl' if data_type == 'train' else '.json'

    def store_data(self):
        """store data in storage_manager_data/<data_type>/<key>.<file_extension>"""
        if not os.path.exists("storage_manager_data/" + self.data_type):
            os.makedirs("storage_manager_data/" + self.data_type)

        with open('storage_manager_data/' + self.data_type + '/' + self.key + self.ext, 'wb') as f:
            pickle.dump(self.data, f)

    def load_data(self):
        """loads and returns data

        raises FileNotFoundError
        """
        with open('storage_manager_data/' + self.data_type + '/' + self.key + self.ext, 'rb') as f:
            return pickle.load(f)
