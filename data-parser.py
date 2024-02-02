import os
import sys
from datasets import load_dataset, load_from_disk
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