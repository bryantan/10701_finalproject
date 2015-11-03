import os
import math

# Constants
POSITIVE = True
NEGATIVE = False
POSITIVE_REVIEWS_FOLDER = "txt_sentoken/pos"
NEGATIVE_REVIEWS_FOLDER = "txt_sentoken/neg"
POSITIVE_WORDS = "vocabulary/positive-words.txt"
NEGATIVE_WORDS = "vocabulary/negative-words.txt"
TOTAL_DATA_SIZE = 1000
TRAINING_DATA_SIZE = 800
LAPLACE_SMOOTHING = 0.1
POSITIVE_PROBS_FILE = "pos_probs.txt"
NEGATIVE_PROBS_FILE = "neg_probs.txt"
ADVERBS = ["extremely", "quite", "just", "almost", "very", "too", "enough"]


class Review:

    def __init__(self, filename=None):
        self.filename = filename
        self.features = set()

    def __repr__(self):
        return "%r" % self.filename

    def add_features(self, vocabulary):
        with open(self.filename) as file:
            text = file.read()
            words = text.replace("\n", "").split(" ")
            i = 0
            while i < len(words):
                # Negation handling
                if ((words[i] == "not" or words[i].endswith("n't"))
                        and words[i + 1] in vocabulary):
                    self.features.add("not_" + words[i + 1])
                    i += 2
                # Including bi-grams
                elif words[i] in ADVERBS and words[i + 1] in vocabulary:
                    self.features.add(words[i] + "_" + words[i + 1])
                    i += 1
                elif words[i] in vocabulary:
                    self.features.add(words[i])
                    i += 1
                else:
                    i += 1


class NaiveBayes:

    def __init__(self):
        self.vocabulary = set()
        self.volume = 0
        self.reviews = {POSITIVE: [], NEGATIVE: []}
        self.probabilities = {POSITIVE: {}, NEGATIVE: {}}

    def create_vocabulary(self):
        self._add_words_to_vocabulary(POSITIVE_WORDS)
        self._add_words_to_vocabulary(NEGATIVE_WORDS)

    def create_reviews(self, files=[], positive=False):
        for filename in files:
            review = Review(filename=filename)
            review.add_features(self.vocabulary)
            if positive:
                self.reviews[POSITIVE].append(review)
            else:
                self.reviews[NEGATIVE].append(review)

    def learn_parameters(self):
        for word in self.vocabulary:
            if word not in self.probabilities[POSITIVE]:
                self.probabilities[POSITIVE][word] = self._find_probability(
                    word, self.reviews[POSITIVE])
            if word not in self.probabilities[NEGATIVE]:
                self.probabilities[NEGATIVE][word] = self._find_probability(
                    word, self.reviews[NEGATIVE])

    def output_probabilities(self):
        given_positive = sorted(self.probabilities[POSITIVE],
                                key=lambda x: self.probabilities[
            POSITIVE][x],
            reverse=True)
        given_negative = sorted(self.probabilities[NEGATIVE],
                                key=lambda x: self.probabilities[
            NEGATIVE][x],
            reverse=True)
        with open(POSITIVE_PROBS_FILE, "w") as file:
            for word in given_positive:
                file.write("%s -> %f\n" %
                           (word, self.probabilities[POSITIVE][word]))
        with open(NEGATIVE_PROBS_FILE, "w") as file:
            for word in given_negative:
                file.write("%s -> %f\n" %
                           (word, self.probabilities[NEGATIVE][word]))

    def predict_reviews(self, files):
        negative, positive = 0, 0
        for filename in files:
            if self._predict_review(filename) == NEGATIVE:
                negative += 1
            else:
                positive += 1
        return (negative, positive)

    def _predict_review(self, filename):
        review = Review(filename=filename)
        review.add_features(self.vocabulary)
        total_pos_prob, total_neg_prob = 1, 1
        for word in self.vocabulary:
            if word in review.features:
                # P(word = 1 | POSITIVE)
                total_pos_prob *= self.probabilities[POSITIVE][word]
                # P(word = 1 | NEGATIVE)
                total_neg_prob *= self.probabilities[NEGATIVE][word]
            else:
                # P(word = 0 | POSITIVE)
                total_pos_prob *= (1 - self.probabilities[POSITIVE][word])
                # P(word = 0 | NEGATIVE)
                total_neg_prob *= (1 - self.probabilities[NEGATIVE][word])
        return POSITIVE if total_pos_prob >= total_neg_prob else NEGATIVE

    def _add_words_to_vocabulary(self, filename):
        with open(filename) as file:
            for line in file:
                word = line.strip()
                if word and not word.startswith(";"):
                    self.vocabulary.add(word)
                    self.volume += 1
                    # Negation handling
                    self.vocabulary.add("not_" + word)
                    # Including bi-grams
                    for adverb in ADVERBS:
                        self.vocabulary.add(adverb + "_" + word)

    def _find_probability(self, word, reviews):
        total_size = len(reviews)
        occurences = 0
        for review in reviews:
            if word in review.features:
                occurences += 1
            if "_" not in word:
                for adverb in ADVERBS:
                    if adverb + "_" + word in review.features:
                        occurences += 1
        return ((float(occurences) + LAPLACE_SMOOTHING) /
                (total_size + LAPLACE_SMOOTHING * self.volume))


if __name__ == "__main__":
    # Retrieve training and testing split
    positive_data_points = map(lambda filename: os.path.join(
        POSITIVE_REVIEWS_FOLDER, filename), os.listdir(POSITIVE_REVIEWS_FOLDER))
    negative_data_points = map(lambda filename: os.path.join(
        NEGATIVE_REVIEWS_FOLDER, filename), os.listdir(NEGATIVE_REVIEWS_FOLDER))
    positive_testing_split = sorted(
        positive_data_points)[TRAINING_DATA_SIZE - TOTAL_DATA_SIZE:]
    negative_testing_split = sorted(
        negative_data_points)[TRAINING_DATA_SIZE - TOTAL_DATA_SIZE:]
    positive_training_split = sorted(positive_data_points)[:TRAINING_DATA_SIZE]
    negative_training_split = sorted(negative_data_points)[:TRAINING_DATA_SIZE]

    # Create Naive Bayes client
    naive_bayes = NaiveBayes()
    naive_bayes.create_vocabulary()
    naive_bayes.create_reviews(files=positive_training_split, positive=True)
    naive_bayes.create_reviews(files=negative_training_split, positive=False)

    # Learn the parameters of the training split
    naive_bayes.learn_parameters()

    # Classify the testing split and report accuracy
    print "Training data accuracy:"
    _, positive = naive_bayes.predict_reviews(positive_training_split)
    print "%.2f %%" % (float(positive) * 100 / len(positive_training_split))
    negative, _ = naive_bayes.predict_reviews(negative_training_split)
    print "%.2f %%" % (float(negative) * 100 / len(negative_training_split))

    print "Testing data accuracy:"
    _, positive = naive_bayes.predict_reviews(positive_testing_split)
    print "%.2f %%" % (float(positive) * 100 / len(positive_testing_split))
    negative, _ = naive_bayes.predict_reviews(negative_testing_split)
    print "%.2f %%" % (float(negative) * 100 / len(negative_testing_split))

    # Output top 10 positive and negative words
    print "Top 10 positive words:", sorted(naive_bayes.probabilities[POSITIVE],
                                           key=lambda x: naive_bayes.probabilities[
                                               POSITIVE][x],
                                           reverse=True)[:10]
    print "Top 10 negative words:", sorted(naive_bayes.probabilities[NEGATIVE],
                                           key=lambda x: naive_bayes.probabilities[
                                               NEGATIVE][x],
                                           reverse=True)[:10]
