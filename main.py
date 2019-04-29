import random
import threading
import re
import itertools
import time

import praw
from praw.models import MoreComments

import numpy as np
from joblib import dump, load

from nltk.stem.snowball import SnowballStemmer

from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import PredefinedSplit

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier

TRAIN = False

CLIENT_ID = 'E9cBapTtE2vUbQ'
CLIENT_SECRET = 'K4eUnFYNbtD-S32h7EpoaOmGVc8'
PASSWORD = 'awhMgfH4FBnYD24'
USERAGENT = 'a subreddit classifier'
USERNAME = 'cheermeup12'

reddit = praw.Reddit(client_id=CLIENT_ID, client_secret=CLIENT_SECRET,
                     password=PASSWORD, user_agent=USERAGENT,
                     username=USERNAME)

print(reddit.user.me())

subreddit_names = ['nba', 'nhl', 'nfl', 'mlb', 'soccer', 'formula1', 'CFB', 'sports']
sub_to_num = {'r/nba': 0, 'r/nhl': 1, 'r/nfl': 2, 'r/mlb': 3, 'r/soccer': 4, 'r/formula1': 5, 'r/CFB': 6, 'r/sports': 7}
num_to_sub = {0: 'r/nba', 1: 'r/nhl', 2: 'r/nfl', 3: 'r/mlb', 4: 'r/soccer', 5: 'r/formula1', 6: 'r/CFB', 7: 'r/sports'}


def file_list(file_name):
    """
    This function opens a file and returns it as a list.
    All new line characters are stripped.
    All lines that start with '#' are considered comments and are not included.

    :param file_name: the name of the file to be put into a list
    :return: a list containing each line of the file, except those that start with '#'
    """

    f_list = []
    with open(file_name, encoding='utf-8') as f:
        for line in f:
            if line[0] != '#' and line[0] != '\n' and len(line[0]) > 0:
                f_list.append(line.strip('\n'))
    return f_list


def parse_reddit_data(file_name):
    data = list()
    sub_classifications = list()

    posts = file_list(file_name)
    for post in posts:
        post_split = post.split(',', 1)

        comments = post_split[1]
        comments = comments.replace(' comment_separator ', ' ')

        comments = re.sub('[^0-9a-zA-Z]+', ' ', comments)

        data.append(comments)
        sub_classifications.append(sub_to_num[post_split[0]])

    return data, sub_classifications


class StemmedCountVectorizer(CountVectorizer):

    def __init__(self, stop_words=None):
        super(StemmedCountVectorizer, self).__init__(stop_words)
        self.stemmer = SnowballStemmer("english", ignore_stopwords=True)

    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([self.stemmer.stem(w) for w in analyzer(doc)])


def get_params_to_test(grid_params):
    """
    Create a complete list of combinations of parameters to test based on possible values for each parameter

    :param grid_params: a map from the parameter name to its possible values
    :return: a list of dicts, with each dict being a set of parameters to test
    """

    combinations = list(itertools.product(*grid_params.values()))

    params_to_test = list()

    param_names = list(grid_params.keys())
    for param_combination in combinations:
        params = dict()

        for i in range(len(param_combination)):
            params[param_names[i]] = param_combination[i]

        params_to_test.append(params)

    return (params_to_test)


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


def fit(num_thread, params_to_test):
    best_pipeline = None
    best_score = 0

    for parameters in params_to_test:
        clf_pipeline = Pipeline([
            ('vect', CountVectorizer(stop_words='english')),
            ('tfidf', TfidfTransformer()),
            ('clf', SVC()),
        ])

        clf_pipeline.set_params(**parameters)

        clf_pipeline.fit(train_data, train_sub_classifications)
        score = clf_pipeline.score(validation_data, validation_sub_classifications)

        if score > best_score:
            best_score = score
            best_pipeline = clf_pipeline

    best_classifier[num_thread] = best_pipeline
    best_scores[num_thread] = best_score


def train():
    train_data, train_sub_classifications = parse_reddit_data('data/training_data.txt')
    validation_data, validation_sub_classifications = parse_reddit_data('data/development_data.txt')

    print('hi')

    clf_pipeline = Pipeline([
        ('vect', CountVectorizer(stop_words='english')),
        ('tfidf', TfidfTransformer()),
        ('clf', MultinomialNB()),
    ])

    parameters = {'clf__alpha': 0.001,
                  'clf__fit_prior': False,
                  'tfidf__norm': 'l1',
                  'tfidf__use_idf': False,
                  'tfidf__sublinear_tf': False,
                  'vect__max_df': 0.50,
                  'vect__ngram_range': (1, 1)
                  }

    clf_pipeline.set_params(**parameters)

    #
    #
    # print(clf_pipeline.get_params().keys())
    # print()
    # print(clf_pipeline.get_params())

    # clf.fit(train_data + validation_data, train_sub_classifications + validation_sub_classifications)
    # print("Best Score: %s%%" % (round(100*clf.best_estimator_.score(train_data, train_sub_classifications), 2)))
    # print("Best Score: %s%%" % (round(100 * clf.best_estimator_.score(validation_data, validation_sub_classifications), 2)))
    #
    # dump(clf.best_estimator_, 'models/classifier.joblib')

    clf_pipeline.fit(train_data, train_sub_classifications)
    print("Best Score: %s%%" % (round(100 * clf_pipeline.score(validation_data, validation_sub_classifications), 2)))

    dump(clf_pipeline, 'models/old_classifier.joblib')


def predict_sub():
    clf = load('models/MultinomialNB.joblib')

    test_data, test_sub_classifications = parse_reddit_data('data/development_data.txt')

    predicted = clf.predict(test_data)

    for actual_sub, data, predicted_sub in zip(test_sub_classifications, test_data, predicted):
        if actual_sub != predicted_sub:
            print('%s: %s <= %r' % (num_to_sub[actual_sub], num_to_sub[predicted_sub], data))

    num_correct = 0
    num_total = 0
    for subreddit_name, category in zip(test_sub_classifications, predicted):
        if subreddit_name == category:
            num_correct += 1

        num_total += 1

    print('%s/%s (%s%%) correct' % (
        num_correct, num_total, round(100 * (np.mean(test_sub_classifications == predicted)), 2)))


if __name__ == '__main__':
    if TRAIN:
        train()
    else:
        predict_sub()
