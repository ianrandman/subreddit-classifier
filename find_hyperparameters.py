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

TRAIN = True

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

classifiers = {'MultinomialNB': MultinomialNB(), 'SVC': SVC(), 'RandomForestClassifier': RandomForestClassifier(),
               'SGDClassifier': SGDClassifier()}

parameters = {'MultinomialNB':
                  {'clf__alpha': np.logspace(-5, 0, num=6),
                   'clf__fit_prior': [True, False],
                   'tfidf__norm': ['l1', 'l2', None],
                   'tfidf__use_idf': [True, False],
                   'tfidf__sublinear_tf': [True, False],
                   'vect__max_df': [0.50, 0.75, 1.0],
                   'vect__ngram_range': [(1, 1), (1, 2)]
                   },
              'SVC': SVC(),
              'RandomForestClassifier': RandomForestClassifier(),
              'SGDClassifier': SGDClassifier()}

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

    return data[:100], sub_classifications[:100]


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

train_data, train_sub_classifications = parse_reddit_data('data/training_data.txt')
validation_data, validation_sub_classifications = parse_reddit_data('data/development_data.txt')

def fit(num_thread, params_to_test):
    best_classifier = None
    best_score = 0

    for parameters in params_to_test:
        clf_pipeline = Pipeline([
            ('vect', CountVectorizer(stop_words='english')),
            ('tfidf', TfidfTransformer()),
            ('clf', MultinomialNB()),
        ])

        clf_pipeline.set_params(**parameters)

        #print("Thread number: %s -- here" % (num_thread))

        clf_pipeline.fit(train_data, train_sub_classifications)

        #print("Thread number: %s -- hi" % (num_thread))

        score = clf_pipeline.score(validation_data, validation_sub_classifications)

        if score > best_score:
            best_score = score
            best_classifier = clf_pipeline

        print("Thread number: %s -- model trained and tested" % (num_thread))

    best_classifiers[num_thread] = best_classifier
    best_scores[num_thread] = best_score


def train():
    print('hi')

    for i in range(num_threads):
        threads[i] = threading.Thread(target=fit, args=(i, thread_params[i],))
        threads[i].start()

    for i in range(num_threads):
        threads[i].join()

    best_score = max(best_scores)
    best_classifier = best_classifiers[best_scores.index(best_score)]

    print("Best Score: %s%%" % (round(100 * best_score, 2)))
    print(best_classifier.get_params())

    dump(best_classifiers, 'models/classifier.joblib')



if __name__ == '__main__':
    grid_params = {'clf__alpha': np.logspace(-5, 0, num=6),
                   'clf__fit_prior': [True, False],
                   'tfidf__norm': ['l1', 'l2', None],
                   'tfidf__use_idf': [True, False],
                   'tfidf__sublinear_tf': [True, False],
                   'vect__max_df': [0.50, 0.75, 1.0],
                   'vect__ngram_range': [(1, 1), (1, 2)]
                   }

    params_to_test = get_params_to_test(grid_params)
    num_params_in_thread = 10

    thread_params = list(split(params_to_test, int(len(params_to_test) / num_params_in_thread)))

    num_threads = len(thread_params)

    threads = [None] * num_threads
    best_classifiers = [None] * num_threads
    best_scores = [None] * num_threads

    train()
