import os
import re
import json
import math

# Constants
KNOWLEDGE = True
GENERAL = False
SAMPLES_FOLDER = os.getcwd() + "/samples"
NUM_TRAINING_FILES = 7
LAPLACE_SMOOTHING = 0.1
DICT_FILE = os.getcwd() + "/textbook_dict"
GENERAL_PROBS_FILE = "general.probs"
KNOWLEDGE_PROBS_FILE = "knowledge.probs"


class Sample:

    def __init__(self, features=None):
        self.features = features
        self.main_set = set()
        self.prediction = GENERAL

    def add_features(self, vocabulary):
        words = self.features["main"].split(" ")
        words = map(lambda word: self._remove_formatting(word), words)
        i = 0
        while i < len(words):
            if words[i] in vocabulary:
                self.main_set.add(words[i])
            i += 1

    def _remove_formatting(self, word):
        word = re.sub('<.*?>', '', re.sub('</.*?>', '', word))
        punc_to_remove = ['.', ',', '!', '?', '(', ')', ';']
        for punc in punc_to_remove:
            word = word.replace(punc, '')
        return word.lower()


class NaiveBayes:

    def __init__(self):
        self.vocabulary = set()
        self.volume = 0
        self.samples = {KNOWLEDGE: [], GENERAL: []}
        self.probabilities = {KNOWLEDGE: {}, GENERAL: {}}

    def create_vocabulary(self):
        with open(DICT_FILE, "r") as dict_file:
            dict_raw = dict_file.read()
            self.vocabulary = set(json.loads(dict_raw))
        self.volume = len(self.vocabulary)

    def add_samples(self, files=[]):
        for filename in files:
            with open(filename, "r") as sample_file:
                samples_raw = sample_file.read()
                samples_json = json.loads(samples_raw)
                for sample_json in samples_json:
                    sample = Sample(features=sample_json)
                    sample.add_features(self.vocabulary)
                    if sample_json['type'] != "none":
                        self.samples[KNOWLEDGE].append(sample)
                    else:
                        self.samples[GENERAL].append(sample)

    def learn_parameters(self):
        for word in self.vocabulary:
            if word not in self.probabilities[KNOWLEDGE]:
                self.probabilities[KNOWLEDGE][word] = self._find_probability(
                    word, self.samples[KNOWLEDGE])
            if word not in self.probabilities[GENERAL]:
                self.probabilities[GENERAL][word] = self._find_probability(
                    word, self.samples[GENERAL])

    def output_probabilities(self):
        given_knowledge = sorted(self.probabilities[KNOWLEDGE],
                                key=lambda x: self.probabilities[KNOWLEDGE][x],
                                reverse=True)
        given_general = sorted(self.probabilities[GENERAL],
                                key=lambda x: self.probabilities[GENERAL][x],
                                reverse=True)
        with open(KNOWLEDGE_PROBS_FILE, "w") as file:
            for word in given_knowledge:
                file.write("%s -> %f\n" %
                           (word, self.probabilities[KNOWLEDGE][word]))
        with open(GENERAL_PROBS_FILE, "w") as file:
            for word in given_general:
                file.write("%s -> %f\n" %
                           (word, self.probabilities[GENERAL][word]))

    def predict_samples(self, files):
        knowledge_samples, general_samples = [], []
        for filename in files:
            with open(filename, "r") as sample_file:
                samples_raw = sample_file.read()
                samples_json = json.loads(samples_raw)
                for sample_json in samples_json:
                    sample = Sample(features=sample_json)
                    sample.add_features(self.vocabulary)
                    if self._predict_sample(sample) == GENERAL:
                        general_samples.append(sample)
                    else:
                        sample.prediction = KNOWLEDGE
                        knowledge_samples.append(sample)
        return (knowledge_samples, general_samples)

    def _predict_sample(self, sample):
        total_knowledge_prob, total_general_prob = 1, 1
        for word in self.vocabulary:
            if word in sample.main_set:
                # P(word = 1 | KNOWLEDGE)
                total_knowledge_prob *= self.probabilities[KNOWLEDGE][word]
                # P(word = 1 | GENERAL)
                total_general_prob *= self.probabilities[GENERAL][word]
            else:
                # P(word = 0 | KNOWLEDGE)
                total_knowledge_prob *= (1 - self.probabilities[KNOWLEDGE][word])
                # P(word = 0 | GENERAL)
                total_general_prob *= (1 - self.probabilities[GENERAL][word])
        return KNOWLEDGE if total_knowledge_prob >= total_general_prob else GENERAL

    def _find_probability(self, word, samples):
        total_size = len(samples)
        occurences = 0
        for sample in samples:
            if word in sample.main_set:
                occurences += 1
        return ((float(occurences) + LAPLACE_SMOOTHING) /
                (total_size + LAPLACE_SMOOTHING * self.volume))


if __name__ == "__main__":
    # Retrieve training and testing splits
    training_data_files = map(lambda filename: os.path.join(
        SAMPLES_FOLDER, filename), sorted(os.listdir(SAMPLES_FOLDER))[:NUM_TRAINING_FILES])
    testing_data_files = map(lambda filename: os.path.join(
        SAMPLES_FOLDER, filename), sorted(os.listdir(SAMPLES_FOLDER))[NUM_TRAINING_FILES:])

    # Create Naive Bayes client
    naive_bayes = NaiveBayes()
    naive_bayes.create_vocabulary()
    naive_bayes.add_samples(files=training_data_files)

    # Learn the parameters of the training split
    naive_bayes.learn_parameters()
    naive_bayes.output_probabilities()

    # Classify the testing split and report accuracy
    knowledge_samples, general_samples = naive_bayes.predict_samples(training_data_files)
    knowledge_correct_count, general_correct_count = 0, 0
    total_count = len(knowledge_samples) + len(general_samples)
    for sample in knowledge_samples:
        if sample.features['type'] != "none":
            knowledge_correct_count = knowledge_correct_count + 1
    for sample in general_samples:
        if sample.features['type'] == "none":
            general_correct_count = general_correct_count + 1

    correct_count = knowledge_correct_count + general_correct_count
    print "Training data accuracy:"
    print "Total: %.2f %%" % (float(correct_count) * 100 / total_count)
    print "Knowledge: %.2f %%" % (float(knowledge_correct_count) * 100 / len(knowledge_samples))
    print "General: %.2f %%" % (float(general_correct_count) * 100 / len(general_samples))
    print "Correct count: %d, total count: %d" % (correct_count, total_count)


    knowledge_samples, general_samples = naive_bayes.predict_samples(testing_data_files)
    knowledge_correct_count, general_correct_count = 0, 0
    total_count = len(knowledge_samples) + len(general_samples)
    for sample in knowledge_samples:
        if sample.features['type'] != "none":
            knowledge_correct_count = knowledge_correct_count + 1
    for sample in general_samples:
        if sample.features['type'] == "none":
            general_correct_count = general_correct_count + 1

    correct_count = knowledge_correct_count + general_correct_count
    print "Testing data accuracy:"
    print "Total: %.2f %%" % (float(correct_count) * 100 / total_count)
    print "Knowledge: %.2f %%" % (float(knowledge_correct_count) * 100 / len(knowledge_samples))
    print "General: %.2f %%" % (float(general_correct_count) * 100 / len(general_samples))
    print "Correct count: %d, total count: %d" % (correct_count, total_count)


    # # Output top 10 positive and negative words
    # print "Top 10 positive words:", sorted(naive_bayes.probabilities[POSITIVE],
    #                                        key=lambda x: naive_bayes.probabilities[
    #                                            POSITIVE][x],
    #                                        reverse=True)[:10]
    # print "Top 10 negative words:", sorted(naive_bayes.probabilities[NEGATIVE],
    #                                        key=lambda x: naive_bayes.probabilities[
    #                                            NEGATIVE][x],
    #                                        reverse=True)[:10]
