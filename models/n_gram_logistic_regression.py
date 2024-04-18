
from numpy.random import permutation
from datasets import DatasetDict
from data_parser import Dataset
from models.logistic_regression import LogisticRegression
from constants import OFF, NOT
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict


class NGramLogisticRegression(LogisticRegression):
    def __init__(self, dataset: Dataset, variation_name="") -> None:
        super().__init__(dataset, "n_grams_"+ variation_name)
        vocabolary = []
        
        self.nCharMin, self.nCharMax = 1,4
        self.nWordMin, self.nWordMax = 1,1


        for comment in dataset.to_list():
            vocabolary.append(comment.text)

        vectorizer = CountVectorizer(analyzer='char', ngram_range=(self.nCharMin, self.nCharMax))
        vectorizer.fit_transform(vocabolary)
        self.weights = dict().fromkeys(vectorizer.get_feature_names_out(), 0)

        vectorizer = CountVectorizer(analyzer='word', ngram_range=(self.nWordMin, self.nWordMax))
        vectorizer.fit_transform(vocabolary)
        x = dict().fromkeys(vectorizer.get_feature_names_out(), 0)
        self.weights.update(x)

    def train(self):
        """Resets weights and bias term, then train model on all comments in a random order"""

        for i in permutation(self.data_length):
            if i < len(self.dataset[OFF]):
                expected = 1
                comment = self.dataset[OFF][i]
            else:
                expected = 0
                comment = self.dataset[NOT][i - len(self.dataset[OFF])]

            features = self.calculate_features(comment)
            vector_product = 0
            for feat in features.keys():
                vector_product += self.weights[feat] * features[feat]
            guess = self.sigmoid(vector_product + self.bias_term)
            self.gradient_descent(
                features, self.crossentropy_loss(
                    guess, expected), 0.1 if expected == 1 else -0.1
            )

    def gradient_descent(self, features, loss, trainingspeed):
        """Finds gradient vector and moves the opposite way

        Args:
            features (list[int]): List of features for the comment
            loss (float): A number giving value to how far the guess is from the right answer
            trainingspeed (float): Dictates how fast the weights change
        """
        for feat in features.keys():
            self.weights[feat] += (loss * features[feat]) * (trainingspeed)

        self.bias_term += loss * trainingspeed

    def calculate_features(self, comment):
        features = defaultdict(int)
        try:
            vectorizer = CountVectorizer(analyzer='char', ngram_range=(self.nCharMin, self.nCharMax))
            vectorizer.fit_transform([comment.text])

            for feat in vectorizer.get_feature_names_out():
                if feat in self.weights:
                    features[feat] += 1

            vectorizer = CountVectorizer(analyzer='word', ngram_range=(self.nWordMin, self.nWordMax))
            vectorizer.fit_transform([comment.text])

            for feat in vectorizer.get_feature_names_out():
                if feat in self.weights:
                    features[feat] += 1
        except ValueError:
            None
        return features

    def test(self, test_dataset_text: list):
        result = []
        self.train()

        for test in test_dataset_text:
            features = self.calculate_features(test)
            vector_product = 0
            for feat in features:
                vector_product += self.weights[feat]
            guess = self.sigmoid(vector_product + self.bias_term)

            result.append(OFF) if guess > 0.5 else result.append(NOT)

        return result
