import os
import re
import json
import math
from nltk.tag import StanfordPOSTagger

# Constants
KNOWLEDGE = True
GENERAL = False
SAMPLES_FOLDER = os.getcwd() + "/samples"
TRAINING_SAMPLES_FOLDER = SAMPLES_FOLDER + "/training"
TESTING_SAMPLES_FOLDER = SAMPLES_FOLDER + "/testing"
LAPLACE_SMOOTHING = 0.1
DICT_FILE = os.getcwd() + "/textbook_dict"
GENERAL_PROBS_FILE = "general.probs"
KNOWLEDGE_PROBS_FILE = "knowledge.probs"
HAS_BOLD = "_has_bold_"

# Stanford POS Tagger
TAGGER_PATH = "/media/bolat/DATA/10-701/project/crf++/stanford-postagger" + \
    "-2015-04-20/models/english-bidirectional-distsim.tagger"
MODEL_PATH = "/home/bolat/data/10-701/project/crf++/stanford-postagger" + \
    "-2015-04-20/stanford-postagger.jar"
OUTPUT_FILE = "train.txt"


# Features
# - contains a key phrase
# - contains a bold word
KEY_PHRASES = ["is defined", "are defined", "is called", "are called"]


class Sample:

    def __init__(self, features=None):
        self.features = features
        self.main_set = set()
        self.prediction = GENERAL
        self.is_bold = False

    def add_features(self):
        words = self.features["main"].split(" ")
        words = map(lambda word: self._remove_formatting(word), words)
        self.sentence = " ".join(words)
        for phrase in KEY_PHRASES:
            if phrase in self.sentence:
                self.main_set.add(phrase)
        if re.search("<b>[a-z, A-Z]{2,}</b>", self.features["main"]) is not None:
            self.is_bold = True

    def _remove_formatting(self, word):
        word = re.sub('<.*?>', '', re.sub('</.*?>', '', word))
        punc_to_remove = ['.', ',', '!', '?', '(', ')', ';']
        for punc in punc_to_remove:
            word = word.replace(punc, '')
        return word.lower()


class NaiveBayes:

    def __init__(self):
        self.vocabulary = set()
        self.samples = {KNOWLEDGE: [], GENERAL: []}
        self.probabilities = {KNOWLEDGE: {}, GENERAL: {}}

    def add_samples(self, files=[]):
        for filename in files:
            with open(filename, "r") as sample_file:
                samples_raw = sample_file.read()
                samples_json = json.loads(samples_raw)
                for sample_json in samples_json:
                    sample = Sample(features=sample_json)
                    sample.add_features()
                    if sample_json['type'] != "none":
                        self.samples[KNOWLEDGE].append(sample)
                    else:
                        self.samples[GENERAL].append(sample)

    def learn_parameters(self):
        self.probabilities[KNOWLEDGE][HAS_BOLD] = self._find_probability(
            HAS_BOLD, self.samples[KNOWLEDGE])
        self.probabilities[GENERAL][HAS_BOLD] = self._find_probability(
            HAS_BOLD, self.samples[GENERAL])
        for phrase in KEY_PHRASES:
            if phrase not in self.probabilities[KNOWLEDGE]:
                self.probabilities[KNOWLEDGE][phrase] = self._find_probability(
                    phrase, self.samples[KNOWLEDGE])
            if phrase not in self.probabilities[GENERAL]:
                self.probabilities[GENERAL][phrase] = self._find_probability(
                    phrase, self.samples[GENERAL])

    def output_probabilities(self):
        given_knowledge = sorted(self.probabilities[KNOWLEDGE],
                                 key=lambda x: self.probabilities[
                                     KNOWLEDGE][x],
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
                    sample.add_features()
                    if self._predict_sample(sample) == GENERAL:
                        general_samples.append(sample)
                    else:
                        sample.prediction = KNOWLEDGE
                        knowledge_samples.append(sample)
        return (knowledge_samples, general_samples)

    def _predict_sample(self, sample):
        total_knowledge_prob, total_general_prob = 1, 1
        for phrase in KEY_PHRASES:
            if phrase in sample.main_set:
                # P(phrase = 1 | KNOWLEDGE)
                total_knowledge_prob *= self.probabilities[KNOWLEDGE][phrase]
                # P(phrase = 1 | GENERAL)
                total_general_prob *= self.probabilities[GENERAL][phrase]
            else:
                # P(phrase = 0 | KNOWLEDGE)
                total_knowledge_prob *= (1 -
                                         self.probabilities[KNOWLEDGE][phrase])
                # P(phrase = 0 | GENERAL)
                total_general_prob *= (1 - self.probabilities[GENERAL][phrase])
        return KNOWLEDGE if total_knowledge_prob >= total_general_prob else GENERAL

    def _find_probability(self, phrase, samples):
        total_size = len(samples)
        occurences = 0
        for sample in samples:
            if phrase == HAS_BOLD and sample.is_bold:
                occurences += 1
            elif phrase in sample.main_set:
                occurences += 1
        return float(occurences) / total_size


class POSTagger:

    def __init__(self, tagger_path, model_path, output_filename):
        self.st = StanfordPOSTagger(tagger_path, model_path)
        self.output_filename = output_filename
        try:
            os.remove(self.output_filename)
        except OSError:
            pass

    def output_knowledge(self, sentence):
        sentence += " ."
        s = ""
        with open(self.output_filename, "a") as file:
            for word, pos_tag in self.st.tag(sentence.split()):
                file.write(("%s\t%s\n" % (word, pos_tag)).encode("utf-8"))
            file.write("\n")


if __name__ == "__main__":
    # Retrieve training and testing splits
    training_data_files = map(lambda filename: os.path.join(
        TRAINING_SAMPLES_FOLDER, filename), os.listdir(TRAINING_SAMPLES_FOLDER))
    testing_data_files = map(lambda filename: os.path.join(
        TESTING_SAMPLES_FOLDER, filename), os.listdir(TESTING_SAMPLES_FOLDER))

    # Create Naive Bayes client
    naive_bayes = NaiveBayes()
    naive_bayes.add_samples(files=training_data_files)

    # Create POSTagger client
    pos_tagger = POSTagger(TAGGER_PATH, MODEL_PATH, OUTPUT_FILE)
    # Create trainning data file for CRF++ with POS tags
    for sample in naive_bayes.samples[KNOWLEDGE]:
        pos_tagger.output_knowledge(sample.sentence)

    # Learn the parameters of the training split
    naive_bayes.learn_parameters()
    naive_bayes.output_probabilities()

    # Classify the testing split and report accuracy
    knowledge_samples, general_samples = naive_bayes.predict_samples(
        training_data_files)
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

    knowledge_samples, general_samples = naive_bayes.predict_samples(
        testing_data_files)
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
