
from data_parser import Dataset
from models.naive_bayes import NaiveBayes
from spacy.tokens import Token, Doc
from data_storage import TrainData
import math
import utils


class BinaryNaiveBayes(NaiveBayes):
    def __init__(self, dataset: Dataset) -> None:
        super().__init__(dataset)
        self.train_data = TrainData("binary-naive-bayes")

    def train(self):  # pragma: no cover
        for c in self.classes:  # type: ignore
            # amount of instances with this class
            n_classes = len(self.dataset)
            # it gives a base chance for it being NOT or OFF based on the split in the dataset
            self.logprior[c] = math.log10(n_classes / self.n_instances)

            # Changed from base: Only counts words ones pr comment
            words_in_class = utils.extract_unique_words_from_comments(
                self.dataset[c])

            n_words = self._count_words(words_in_class, self.vocabulary)

            for word in self.vocabulary:
                count = words_in_class[word] if word in words_in_class else 0

                # compute the likelihood of this word being generated from this class based on
                # the amount of the word used in the class compared to the total amount of
                # words used in the class.
                self.loglikelihood[(word.text, c)] = math.log10(
                    (count + 1) / (n_words - count)
                )

            # update the train data parameters
            self.train_data.parameters = self.logprior, self.loglikelihood, self.classes
            # save the train data to disk
            self.train_data.save_to_disk()

    def _find_class(
        self, comment: Doc, classes: list, logprior: dict, loglikelihood: dict
    ):  # pragma: no cover
        sum = {}
        for c in classes:
            sum[c] = logprior[c]
            seen_words = []
            for word in comment:
                # Changed from base: Only counts words ones pr comment
                if word not in seen_words:
                    seen_words.append(word)
                    try:
                        sum[c] += loglikelihood[(word.text, c)]
                    except KeyError:
                        continue
        return utils.get_max_value_key(sum)

    def __str__(self) -> str:
        return "binary-naive-bayes"
