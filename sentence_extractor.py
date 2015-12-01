# From scikit-learn website
# http://scikit-learn.org/stable/auto_examples/text/document_classification_20newsgroups.html
# Other notes
# http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html

import os
import json
import numpy as np
from naive_bayes import Sample
from time import time
import matplotlib.pyplot as plt
from scipy import sparse


from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.utils.extmath import density
from sklearn import metrics

from unbalanced_dataset import UnbalancedDataset
from over_sampling import OverSampler
from over_sampling import SMOTE


SAMPLES_FOLDER = os.getcwd() + "/samples"
TRAINING_SAMPLES_FOLDER = SAMPLES_FOLDER + "/training"
TESTING_SAMPLES_FOLDER = SAMPLES_FOLDER + "/testing"
USE_HASHING = True
USE_CHI2 = True
SELECT_CHI2 = 10
PRINT_TOP10 = True
N_FEATURES = 2 ** 16

categories = ['definition', 'law', 'none']

def get_samples(files=[]):
    samples = []
    for filename in files:
        with open(filename, "r") as sample_file:
            samples_raw = sample_file.read()
            samples_json = json.loads(samples_raw)
            for sample_json in samples_json:
                sample = Sample(features=sample_json)
                sample.add_features()
                samples.append(sample)

    data = [sample.features["main"] for sample in samples]
    labels = [sample.features["type"] for sample in samples]
    return (data, labels)


# Retrieve training and testing splits
training_data_files = map(lambda filename: os.path.join(
    TRAINING_SAMPLES_FOLDER, filename), os.listdir(TRAINING_SAMPLES_FOLDER))
testing_data_files = map(lambda filename: os.path.join(
    TESTING_SAMPLES_FOLDER, filename), os.listdir(TESTING_SAMPLES_FOLDER))

training_data, y_train = get_samples(training_data_files)
testing_data, y_test = get_samples(testing_data_files)

# Remove incorrect labels
new_ytrain = [x for x in y_train if x != "none"]
new_ytrain = [x for x in new_ytrain if x != "definition"]
new_ytrain = [x for x in new_ytrain if x != "law"]
print new_ytrain

# Change "law" labels to "definition"
y_train = [x if x != "law" else "definition" for x in y_train]
y_test = [x if x != "law" else "definition" for x in y_test]
print len(training_data), len(y_train), y_train.count("none"), y_train.count("definition"), y_train.count("law")
print len(testing_data), len(y_test), y_test.count("none"), y_test.count("definition"), y_test.count("law")

# Extract features using sparse vectorizer
if USE_HASHING:
    vectorizer = HashingVectorizer(stop_words='english', non_negative=True,
                                   n_features=N_FEATURES, ngram_range=(1, 2))
    X_train = vectorizer.transform(training_data)
else:
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                 stop_words='english', ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(training_data)

X_test = vectorizer.transform(testing_data)

# Oversampling
y_train_new = [0 if x == "definition" else 1 for x in y_train]
# print y_train_new
sm = SMOTE(kind='regular', verbose=True, ratio=10)
X_train, y_train = sm.fit_transform(X_train.toarray(), np.asarray(y_train_new))
# OS = OverSampler(verbose=True, ratio=10)
# X_train, y_train = OS.fit_transform(X_train.toarray(), np.asarray(y_train_new))
X_train = sparse.csr_matrix(X_train)
y_train = y_train.tolist()
y_train = ["definition" if x == 0 else "none" for x in y_train]

# mapping from integer feature name to original token string
if USE_HASHING:
    feature_names = None
else:
    feature_names = vectorizer.get_feature_names()

# Extracting best features with chi-squared test
if USE_CHI2:
    ch2 = SelectKBest(chi2, k=SELECT_CHI2)
    X_train = ch2.fit_transform(X_train, y_train)
    X_test = ch2.transform(X_test)
    if feature_names:
        # keep selected feature names
        feature_names = [feature_names[i] for i
                         in ch2.get_support(indices=True)]

if feature_names:
    feature_names = np.asarray(feature_names)


def trim(s):
    """Trim string to fit on terminal (assuming 80-column display)"""
    return s if len(s) <= 80 else s[:77] + "..."


###############################################################################
# Benchmark classifiers
def benchmark(clf):
    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(X_test)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    score = metrics.accuracy_score(y_test, pred)
    print("accuracy:   %0.3f" % score)

    if hasattr(clf, 'coef_'):
        print("dimensionality: %d" % clf.coef_.shape[1])
        print("density: %f" % density(clf.coef_))

        if PRINT_TOP10 and feature_names is not None:
            print("top 10 keywords per class:")
            for i, category in enumerate(categories):
                top10 = np.argsort(clf.coef_[i])[-10:]
                print(trim("%s: %s"
                      % (category, " ".join(feature_names[top10]))))
        print()

    print("classification report:")
    print(metrics.classification_report(y_test, pred,
                                        target_names=categories))

    print("confusion matrix:")
    print(metrics.confusion_matrix(y_test, pred))

    print()
    clf_descr = str(clf).split('(')[0]
    return clf_descr, score, train_time, test_time

results = []
for clf, name in (
        (RidgeClassifier(tol=1e-2, class_weight='balanced', solver="lsqr"), "Ridge Classifier"),
        (Perceptron(n_iter=50, class_weight='balanced'), "Perceptron"),
        (PassiveAggressiveClassifier(n_iter=50, class_weight='balanced'), "Passive-Aggressive"),
        (KNeighborsClassifier(n_neighbors=10), "kNN"),
        (RandomForestClassifier(n_estimators=100, class_weight='balanced'), "Random forest")):
    print('=' * 80)
    print(name)
    results.append(benchmark(clf))

for penalty in ["l2", "l1"]:
    print('=' * 80)
    print("%s penalty" % penalty.upper())
    # Train Liblinear model
    results.append(benchmark(LinearSVC(loss='l2', penalty=penalty,
                                            class_weight='balanced',
                                            dual=False, tol=1e-3)))

    # Train SGD model
    results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,
                                           penalty=penalty,
                                           class_weight='balanced')))

# Train SGD with Elastic Net penalty
print('=' * 80)
print("Elastic-Net penalty")
results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,
                                       penalty="elasticnet",
                                       class_weight='balanced')))

# Train NearestCentroid without threshold
print('=' * 80)
print("NearestCentroid (aka Rocchio classifier)")
results.append(benchmark(NearestCentroid()))

# Train sparse Naive Bayes classifiers
print('=' * 80)
print("Naive Bayes")
results.append(benchmark(MultinomialNB(alpha=.01)))
results.append(benchmark(BernoulliNB(alpha=.01)))

print('=' * 80)
print("LinearSVC with L1-based feature selection")
# The smaller C, the stronger the regularization.
# The more regularization, the more sparsity.
results.append(benchmark(Pipeline([
  ('feature_selection', LinearSVC(penalty="l1", dual=False,
                                  class_weight='balanced', tol=1e-3)),
  ('classification', LinearSVC())
])))

# Train Ensemble classifiers
print('=' * 80)
print("Ensemble")
results.append(benchmark(AdaBoostClassifier(n_estimators=100)))

# make some plots

indices = np.arange(len(results))

results = [[x[i] for x in results] for i in range(4)]

clf_names, score, training_time, test_time = results
training_time = np.array(training_time) / np.max(training_time)
test_time = np.array(test_time) / np.max(test_time)

plt.figure(figsize=(12, 8))
plt.title("Score")
plt.barh(indices, score, .2, label="score", color='r')
plt.barh(indices + .3, training_time, .2, label="training time", color='g')
plt.barh(indices + .6, test_time, .2, label="test time", color='b')
plt.yticks(())

# Shrink current axis by 20%
ax = plt.subplot(111)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.gcf().set_facecolor('white')

plt.subplots_adjust(left=.25)
plt.subplots_adjust(top=.95)
plt.subplots_adjust(bottom=.05)

for i, c in zip(indices, clf_names):
    plt.text(-.3, i, c)

plt.show()
