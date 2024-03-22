from data_parser import Dataset
from data_storage import TrainData
from models.naive_bayes import NaiveBayes
import math
import utils


class AddKNaiveBayes(NaiveBayes):
    def __init__(self, dataset: Dataset) -> None:
        super().__init__(dataset)
        self.train_data = TrainData("add-k-naive-bayes")

    def __str__(self) -> str:
        return "Add-k-naive-bayes"

    def train(self):  # pragma: no cover
        for c in self.classes:  # type: ignore
            # amount of instances with this class
            n_classes = len(self.dataset)
            # it gives a base chance for it being NOT or OFF based on the split in the dataset
            self.logprior[c] = math.log10(n_classes / self.n_instances)

            words_in_class = utils.extract_words_from_comments(self.dataset[c])
            n_words = self.count_words(words_in_class, self.vocabulary)

            for word in self.vocabulary:
                count = words_in_class[word] if word in words_in_class else 0

                # compute the likelihood of this word being generated from this class based on
                # the amount of the word used in the class compared to the total amount of
                # words used in the class.
                self.loglikelihood[(word.text, c)] = math.log10(
                    # Changed from base: Smothing factor changed from 1 to 0.5
                    (count + 0.5) / (n_words - count + 0.5)
                )

            # update the train data parameters
            self.train_data.parameters = self.logprior, self.loglikelihood, self.classes
            # save the train data to disk
            self.train_data.save_to_disk()
