import array
import numpy
import pandas as pd

from numpy.random import permutation
from datasets import DatasetDict
from sklearn.pipeline import FeatureUnion, make_pipeline
from models.ml_algorithm import MLAlgorithm
from constants import OFF, NOT
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report, accuracy_score


class SVM(MLAlgorithm):
    def __init__(self, dataset: DatasetDict, variation_name=None) -> None:
        super().__init__(dataset, "svm", variation_name)

        self.data = {}
        off = [(x, "OFF") for x in self.dataset["OFF"]]
        not_off = [(x, "NOT") for x in self.dataset["NOT"]]

        self.X = [x[0].text for x in off + not_off]
        self.y = [x[1] for x in off + not_off]

        self.df = pd.DataFrame(self.data)

        self.variation_name = ""
        self.svm_model = make_pipeline(FeatureUnion([
            ('word_tfidf', TfidfVectorizer(analyzer='word', ngram_range=(1, 2))),
            ('char_tfidf', TfidfVectorizer(analyzer='char', ngram_range=(2, 4)))
        ]), SVC(kernel='linear', C=10))

    def train(self):
        self.svm_model.fit(self.X, self.y)

    def test(self, test_dataset_text):
        if self.svm_model is None:
            self.train()

        results = []

        for test in test_dataset_text:
            y_pred = self.svm_model.predict([test.text])

            results.append("NOT" if "NOT" in y_pred else "OFF")

        return results
