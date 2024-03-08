
from models.naive_bayes import NaiveBayes
from sanitizer import Sanitizer
import utils


class Binary_naive_bayes(NaiveBayes):
    # Changed from naive, only count uniqe words in each sentence
    def extract_words_from_label(dataset, label):
        extracted_words = {}
        sentences = utils.extract_sentences_from_label(dataset, label)

        for s in sentences:
            s = Sanitizer(s).sanitize_simple()
            for word in s:
                if word not in extracted_words:
                    extracted_words[word] = 1
        return extracted_words

    def find_class(self, test_instance: str, classes: list):  # pragma: no cover
        sum = {}
        test_instance_list = Sanitizer(test_instance).sanitize_simple()
        for c in classes:
            seen_words = []
            sum[c] = self.logprior[c]
            for word in test_instance_list:
                if word not in seen_words:  # Changed from naive, only counts uniqe words from comment
                    seen_words.append(word)
                    try:
                        sum[c] += self.loglikelihood[(word, c)]
                    except KeyError:
                        continue
        return utils.get_max_value_key(sum)
