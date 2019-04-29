import random
import sys
import os
import threading
import multiprocessing
import re
import itertools
import time

import praw
from praw.models import MoreComments

import numpy as np
import joblib
import pickle

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

subreddit_names = ['nba', 'nhl', 'nfl', 'mlb', 'soccer', 'formula1', 'CFB', 'sports']
sub_to_num = {'r/nba': 0, 'r/nhl': 1, 'r/nfl': 2, 'r/mlb': 3, 'r/soccer': 4, 'r/formula1': 5, 'r/CFB': 6, 'r/sports': 7}
num_to_sub = {0: 'r/nba', 1: 'r/nhl', 2: 'r/nfl', 3: 'r/mlb', 4: 'r/soccer', 5: 'r/formula1', 6: 'r/CFB', 7: 'r/sports'}

classifiers = {'MultinomialNB': MultinomialNB(), 'SVC': SVC(), 'RandomForestClassifier': RandomForestClassifier(),
               'SGDClassifier': SGDClassifier()}

random_forest_max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
random_forest_max_depth.append(None)

grid_params = {'MultinomialNB':
                  {'clf__alpha': np.logspace(-5, 0, num=6),
                   'clf__fit_prior': [True, False],
                   'tfidf__norm': ['l1', 'l2', None],
                   'tfidf__use_idf': [True, False],
                   'tfidf__sublinear_tf': [True, False],
                   'vect__max_df': [0.50, 0.75, 1.0],
                   'vect__ngram_range': [(1, 1), (1, 2)]
                   },
              'SVC':
                  {'clf__C': np.logspace(-2, 2, num=5),
                   'clf__gamma': np.logspace(-2, 2, num=5),
                   'clf__kernel': ['rbf', 'linear'],
                   'tfidf__norm': ['l1', 'l2', None],
                   'tfidf__use_idf': [True, False],
                   'tfidf__sublinear_tf': [True, False],
                   'vect__max_df': [0.50, 0.75, 1.0],
                   'vect__ngram_range': [(1, 1), (1, 2)]
                   },
              'RandomForestClassifier':
                  {'clf__n_estimators': [int(x) for x in np.linspace(start=10, stop=2000, num=10)],
                   'clf__max_depth': random_forest_max_depth,
                   'clf__min_samples_split': [2, 5, 10],
                   'clf__min_samples_leaf': [1, 2, 4],
                   'clf__n_jobs': [-1],
                   'tfidf__norm': ['l1', 'l2', None],
                   'tfidf__use_idf': [True, False],
                   'tfidf__sublinear_tf': [True, False],
                   'vect__max_df': [0.50, 0.75, 1.0],
                   'vect__ngram_range': [(1, 1), (1, 2)]
                   },
              'SGDClassifier':
                  {'clf__loss': ['hinge', 'log'],
                   'clf__penalty': ['l1', 'l2', None],
                   'clf__alpha': np.logspace(-4, 2, num=7),
                   'clf__n_jobs': [-1],
                   'tfidf__norm': ['l1', 'l2', None],
                   'tfidf__use_idf': [True, False],
                   'tfidf__sublinear_tf': [True, False],
                   'vect__max_df': [0.50, 0.75, 1.0],
                   'vect__ngram_range': [(1, 1), (1, 2)]
                   },
              }

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


full_train_data, full_train_sub_classifications = parse_reddit_data('data/training_data.txt')
full_validation_data, full_validation_sub_classifications = parse_reddit_data('data/development_data.txt')

num_posts_train_hyperparameters = 500

partial_train_data, partial_train_sub_classifications = full_train_data[:num_posts_train_hyperparameters], full_train_sub_classifications[:num_posts_train_hyperparameters]
partial_validation_data, partial_validation_sub_classifications = full_validation_data[:num_posts_train_hyperparameters], full_validation_sub_classifications[:num_posts_train_hyperparameters]


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


def get_new_pipeline(classifier_name):
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


def fit_get_hyperparameters(num_thread, params_to_test, classifier_name):
    best_params = None
    best_score = 0

    run_num = 0

    for parameters in params_to_test:
        run_num += 1

        clf_pipeline = get_new_pipeline(classifier_name)

        clf_pipeline.set_params(**parameters)
        clf_pipeline.fit(partial_train_data, partial_train_sub_classifications)

        score = clf_pipeline.score(partial_validation_data, partial_validation_sub_classifications)

        if score > best_score:
            best_score = score
            best_params = parameters

        print("Thread number: %s -- model trained and tested -- run number %s" % (num_thread, run_num))

    with open('temp/' + classifier_name + '.temp', "ab") as temp_file:
        pickle.dump((best_score, best_params), temp_file)

    # threads.best_params_sets[num_thread] = best_params
    # threads.best_scores[num_thread] = best_score


def train(num_threads, classifier_name):
    threads = [None] * num_threads

    for i in range(num_threads):
        threads[i] = multiprocessing.Process(target=fit_get_hyperparameters, args=(i, thread_params[i], classifier_name,))
        threads[i].start()

    for i in range(num_threads):
        threads[i].join()

    # best_score = max(threads.best_scores)
    # best_params = threads.best_params_sets[threads.best_scores.index(best_score)]

    best_score = 0
    best_params = None
    with open('temp/' + classifier_name + '.temp', "rb") as temp_file:
        try:
            while True:
                score, parameters = pickle.load(temp_file)

                if score > best_score:
                    best_score = score
                    best_params = parameters
        except EOFError: # reached end of file
            pass

    os.remove('temp/' + classifier_name + '.temp')


    clf_pipeline = get_new_pipeline(classifier_name)
    clf_pipeline.set_params(**best_params)

    clf_pipeline.fit(full_train_data, full_train_sub_classifications)

    print("Score: %s%%" % (round(100 * clf_pipeline.score(full_validation_data, full_validation_sub_classifications), 2)))
    print(best_params)

    joblib.dump(clf_pipeline, 'models/' + classifier_name + '.joblib')



if __name__ == '__main__':
    classifier_name = sys.argv[1]
    print(classifier_name)

    open('temp/' + classifier_name + '.temp', "w").close() # clear the temp file

    params_to_test = get_params_to_test(grid_params[classifier_name])
    random.shuffle(params_to_test)

    params_to_test = params_to_test[:500] # test on a random selection of 500 parameter combinations

    num_params_in_thread = 50 # 50 parameter selection in each thread for a total of 10 threads

    thread_params = list(split(params_to_test, int(len(params_to_test) / num_params_in_thread)))

    num_threads = len(thread_params)

    start_training_time = time.time()
    train(num_threads, classifier_name)

    print('Time to find best hyperparameters and train on full training set: %s seconds' % (time.time() - start_training_time))
