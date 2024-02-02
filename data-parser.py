import os
import sys
from datasets import load_dataset, load_from_disk, DatasetDict
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve data from huggingface and store in ./data folder only if ./data
# folder doesn't exist or -r flag was used
if not os.path.isdir("./data") or len(sys.argv) > 1 and sys.argv[1] == "-r":
    load_dataset(
        "DDSC/dkhate", token=os.getenv('HUGGING_FACE_ACCESS_TOKEN')).save_to_disk("./data")

dataset = load_from_disk("./data")

dataset_train = dataset["train"]
dataset_test = dataset["test"]


def get_test():
    return dataset_test


def get_test_comments():
    return dataset_test["text"]


def get_train():
    return dataset_train


def majority_offensive(dataset: DatasetDict):
    labels = dataset["label"]

    off_label_amount = 0

    for label in labels:
        if label == "OFF":
            off_label_amount += 1

    return 100 * (off_label_amount/len(labels))
