import os
from datasets import load_dataset
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

dataset = load_dataset(
    "DDSC/dkhate", token=os.getenv('HUGGING_FACE_ACCESS_TOKEN'))

print(dataset)
