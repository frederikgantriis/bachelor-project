import os
import sys
from datasets import load_dataset, load_from_disk
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve data from huggingface and store in ./data folder only if ./data
# folder doesn't exist or -r flag was used

dataset_path = './data/dk-hate-dataset'

if not os.path.isdir(dataset_path) or len(sys.argv) > 1 and sys.argv[1] == "-r":
    load_dataset(
        "DDSC/dkhate", token=os.getenv('HUGGING_FACE_ACCESS_TOKEN')).save_to_disk(dataset_path)

dataset = load_from_disk(dataset_path)


def get_test_dataset():
    return dataset["test"]


def get_train_dataset():
    return dataset["train"]
