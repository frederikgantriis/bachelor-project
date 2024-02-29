import math

from datasets import DatasetDict, Dataset
from models.ml_algorithm import MLAlgorithm

class LogisticRegression(MLAlgorithm):
    def __init__(self, dataset: DatasetDict) -> None:
        super().__init__(dataset)
        

    def sigmoid(self, x:float):
        return 1/(1+math.e**(-x))
    
    def crossentropy_loss(self, guess, expected):
        return -(expected * math.log(guess) + (1-expected) * math.log(1-guess))