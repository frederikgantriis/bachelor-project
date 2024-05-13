from random import shuffle
import pandas as pd

from datasets import DatasetDict
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.svm import SVC
from models.ml_algorithm import MLAlgorithm
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from models.ml_algorithm import MLAlgorithm
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.svm import SVC


class SVM(MLAlgorithm):
    def __init__(self, dataset: DatasetDict, variation_name=None) -> None:
        super().__init__(dataset, "svm", variation_name)

        self.data = {}
        off = [(x, "OFF") for x in self.dataset["OFF"]]
        not_off = [(x, "NOT") for x in self.dataset["NOT"]]
        self.list = off + not_off

        shuffle(self.list)

        self.X = [x[0].text for x in self.list]
        self.y = [x[1] for x in self.list]

        self.df = pd.DataFrame(self.data)

        self.variation_name = ""
        self.svm_model = make_pipeline(
            FeatureUnion(
                [
                    (
                        "word_tfidf",
                        TfidfVectorizer(analyzer="word", ngram_range=(1, 2)),
                    ),
                    (
                        "char_tfidf",
                        TfidfVectorizer(analyzer="char", ngram_range=(2, 4)),
                    ),
                ]
            ),
            SVC(kernel="linear", C=10),
        )
        self.is_trained = False

    def train(self):
        self.svm_model.fit(self.X, self.y)
        self.is_trained = True

    def test(self, test_dataset_text):
        if not self.is_trained:
            self.train()

        results = []

        for test in test_dataset_text:
            y_pred = self.svm_model.predict([test.text])

            results.append("NOT" if "NOT" in y_pred else "OFF")

        return results
