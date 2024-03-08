
import math
from datasets import DatasetDict
from models.naive_bayes import NaiveBayes
from sanitizer import Sanitizer
import utils


class Binary_naive_bayes(NaiveBayes):
    def train(self):  # pragma: no cover
        c: str
        for c in self.classes:  # type: ignore
            # amount of instances with this class
            n_classes = self.dataset["label"].count(c)
            # it gives a base chance for it being NOT or OFF based on the split in the dataset
            self.logprior[c] = math.log10(n_classes / self.n_instances)

            words_in_class = utils.extract_words_from_label(self.dataset, c)
            n_words = super.count_words(words_in_class, self.vocabulary)

            for word in self.vocabulary:
                count = words_in_class[word] if word in words_in_class else 0

                # compute the likelihood of this word being generated from this class based on
                # the amount of the word used in the class compared to the total amount of
                # words used in the class.
                self.loglikelihood[(word, c)] = math.log10(
                    (count + 1) / (n_words - count)
                )

            # update the train data parameters
            self.train_data.parameters = self.logprior, self.loglikelihood, self.classes
            # save the train data to disk
            self.train_data.save_to_disk()

    def extract_binary_words_from_label(dataset, label):
        extracted_words = {}
        sentences = utils.extract_sentences_from_label(dataset, label)

        for s in sentences:
            s = Sanitizer(s).sanitize_simple()
            for word in s:
                if word not in extracted_words:
                    extracted_words[word] = 1                                                       
                                                            #Changed from naive
        return extracted_words
    
    def test(self, test_dataset_text):
        result = []
        loglikelihood = {}
        logprior = {}

        for i in range(2):  # pragma: no cover
            try:
                logprior, loglikelihood, _ = self.train_data.load_from_disk()
                print("Found Naive Bayes training data!")
                break
            except FileNotFoundError:
                # exit the program if FileNotFoundError happens more than once
                if i == 1:
                    print("ERROR: terminating...")
                    exit()
                print("Naive Bayes training data not found:\nInitializing training...")
                self.train()

        for test in test_dataset_text:
            result.append(
                find_class(
                    test,
                    list(self.classes),
                    logprior=logprior,
                    loglikelihood=loglikelihood,
                )
            )
        return result
    
def find_class(
    test_instance: str, classes: list, logprior: dict, loglikelihood: dict
):  # pragma: no cover
    sum = {}
    test_instance_list = Sanitizer(test_instance).sanitize_simple()
    for c in classes:
        seen_words = []
        sum[c] = logprior[c]
        for word in test_instance_list:
            if word not in seen_words:                                          #Changed from naive
                seen_words.append(word)
                try:
                    sum[c] += loglikelihood[(word, c)]
                except KeyError:
                    continue
    return utils.get_max_value_key(sum)