import os
from datasets import load_dataset, load_from_disk
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

if not os.path.isdir("./data"):
    load_dataset(
        "DDSC/dkhate", token=os.getenv('HUGGING_FACE_ACCESS_TOKEN')).save_to_disk("./data")

dataset = load_from_disk("./data")

print(dataset)
