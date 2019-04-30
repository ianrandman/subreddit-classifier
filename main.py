"""

usage: main.py [-h] [-t] [-d]
               {MultinomialNB,SVC,RandomForestClassifier,SGDClassifier}

positional arguments:
  {MultinomialNB,SVC,RandomForestClassifier,SGDClassifier}
                        The name of the classifier

optional arguments:
  -h, --help            show this help message and exit
  -t, --train           Use to train. Otherwise, testing.
  -d, --use_development_data
                        Use to specify testing on development data. Otherwise,
                        testing data will be used

For example:

__author__ = Ian Randman
__author__ = David Dunlap
"""

import random
import sys
import argparse
import threading
import re
import itertools
import time

import numpy as np
from joblib import dump, load

from nltk.stem.snowball import SnowballStemmer

from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier

from find_hyperparameters import parse_reddit_data

from get_data import subreddit_names
from get_data import sub_to_num
from get_data import num_to_sub

from parameters import classifier_name_to_params

#validation_data, validation_sub_classifications = parse_reddit_data('data/development_data.txt')

TRAIN = False

class StemmedCountVectorizer(CountVectorizer):

    def __init__(self, stop_words=None):
        super(StemmedCountVectorizer, self).__init__(stop_words)
        self.stemmer = SnowballStemmer("english", ignore_stopwords=True)

    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([self.stemmer.stem(w) for w in analyzer(doc)])


def split(a, n):
    """
    Split a collection into a number of evenly sized chunks.

    :param a: the collection to be split
    :param n: the number of evenly sized chunks
    :return: a collection of all the chunks
    """

    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


def get_new_pipeline(classifier_name):
    """
    Create a new classifying Pipeline with a specified estimator.

    :param classifier_name: the name of the estimator to use
    :return: the constructed Pipeline
    """

    if classifier_name == 'MultinomialNB':
        clf = MultinomialNB()
    elif classifier_name == 'SVC':
        clf = SVC()
    elif classifier_name == 'RandomForestClassifier':
        clf = RandomForestClassifier()
    else:
        clf = SGDClassifier()

    clf_pipeline = Pipeline([
        ('vect', CountVectorizer(stop_words='english')),
        ('tfidf', TfidfTransformer()),
        ('clf', clf),
    ])

    return clf_pipeline


def train(classifier_name):
    train_data, train_sub_classifications = parse_reddit_data('data/training_data.txt')

    clf_pipeline = get_new_pipeline(classifier_name)

    parameters = classifier_name_to_params[classifier_name]
    clf_pipeline.set_params(**parameters)

    clf_pipeline.fit(train_data, train_sub_classifications)

    test_data, test_sub_classifications = parse_reddit_data('data/development_data.txt')
    print("Best Score: %s%%" % (round(100 * clf_pipeline.score(test_data, test_sub_classifications), 2)))

    dump(clf_pipeline, 'models/' + classifier_name + '.joblib')


def evaluate(classifier_name, use_development_data):
    clf_pipeline = load('models/' + classifier_name + '.joblib')

    if use_development_data:
        test_data, test_sub_classifications = parse_reddit_data('data/development_data.txt')
    else:
        test_data, test_sub_classifications = parse_reddit_data('data/testing_data.txt')

    predicted = clf_pipeline.predict(test_data)

    for actual_sub, data, predicted_sub in zip(test_sub_classifications, test_data, predicted):
        if actual_sub != predicted_sub:
            print('%s: %s <= %r' % (num_to_sub[actual_sub], num_to_sub[predicted_sub], data))

    num_correct = 0
    num_total = 0
    for subreddit_name, category in zip(test_sub_classifications, predicted):
        if subreddit_name == category:
            num_correct += 1

        num_total += 1

    print('\nClassifier name: ', classifier_name)
    print('%s/%s (%s%%) correct' % (
        num_correct, num_total, round(100 * (np.mean(test_sub_classifications == predicted)), 2)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('classifier_name', help='The name of the classifier', choices=['MultinomialNB', 'SVC', 'RandomForestClassifier', 'SGDClassifier'])
    parser.add_argument('-t', '--train', help='Use to train. Otherwise, testing.', action="store_true")
    parser.add_argument('-d', '--use_development_data', help='Use to specify testing on development data. Otherwise, testing data will be used', action="store_true")
    args = parser.parse_args()

    classifier_name = args.classifier_name
    print(classifier_name)

    if args.train:
        train(classifier_name)
    else:
        evaluate(classifier_name, args.use_development_data)
