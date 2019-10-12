"""
Perform a random search over a set of possible hyperparameters for a specified classifier. Each set of
hyperparameters to test will be trained using a subset of the training data and validated on a subset of the
development data.

usage: find_hyperparameters.py [-h]
                               {MultinomialNB,SVC,RandomForestClassifier,SGDClassifier}

positional arguments:
  {MultinomialNB,SVC,RandomForestClassifier,SGDClassifier}
                        The name of the classifier.

optional arguments:
  -h, --help            show this help message and exit

For example: python find_hyperparameters.py MultinomialNB

__author__ = Ian Randman
__author__ = David Dunlap
"""

import argparse
import queue
import random
import os
import multiprocessing
import threading
import re
import itertools
import time

import joblib
import pickle

from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier

from get_data import sub_to_num
from get_data import DATA_PATH

from get_data import file_list
from parameters import params_to_search # these are the parameters to search over

from io import open
import json

NUM_PARAMS_TO_TEST = 500
NUM_CORES = multiprocessing.cpu_count()

print('Using %s cores\n' % (NUM_CORES))

NUM_POSTS_TRAIN_HYPERPARAMETERS = 500  # the number of posts to train and test on when finding best hyperparameters

BEST_HYPERPARAMETERS_PATH = os.path.dirname(os.path.abspath(__file__)) + '/best_hyperparamters/'
TEMP_PATH = os.path.dirname(os.path.abspath(__file__)) + '/temp/'
MODELS_PATH = os.path.dirname(os.path.abspath(__file__)) + '/models/'

if not os.path.exists(BEST_HYPERPARAMETERS_PATH):
    os.makedirs(BEST_HYPERPARAMETERS_PATH)

if not os.path.exists(TEMP_PATH):
    os.makedirs(TEMP_PATH)

if not os.path.exists(MODELS_PATH):
    os.makedirs(MODELS_PATH)


def parse_reddit_data(file_name):
    """
    Parse a file with each entry being a subreddit name followed by a concatenation of all the comments from a post,
    separated by a comma.

    :param file_name: the name of the file in which the Reddit data is contained
    :return: a pair containing a list of comment concatenations and a list of numerical classifications for each post
    """

    data = list()
    sub_classifications = list()

    posts = file_list(file_name)
    for post in posts:
        post_split = post.split(',', 1)

        comments = post_split[1]

        comments = comments.replace(' comment_separator ', ' ')
        comments = re.sub("[^0-9a-zA-Z']+", ' ', comments)

        data.append(comments)
        sub_classifications.append(sub_to_num[post_split[0]])

    return data, sub_classifications


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


def fit_get_hyperparameters(num_process, params_to_test, classifier_name):
    """
    Find the best set of hyperparameters for a specified classifier. Save the best score on validation data and best
    parameters to a file.

    :param num_process: the process number for the process
    :param params_to_test: the collection of parameters to test on
    :param classifier_name: the name of the estimator
    :return: none
    """

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

        print("Process number: %s -- model trained and tested for %s -- run number %s" %
              (num_process, classifier_name, run_num))

    with open('temp/' + classifier_name + '.temp', "ab") as temp_file:
        pickle.dump((best_score, best_params), temp_file)


def train(num_processs, process_params, classifier_name):
    """
    Train to find the best set of hyperparameters for a specified classifier. Use multiprocessing to break up the
    task into a given number of processes to speed up computation. Finally, train on the full training set and save
    the results.

    :param num_processs: the number of processes to use for finding hyperparameters
    :param classifier_name: the name of the estimator
    :return: the time it took to find the hyperparameters and train the classifier
    """

    processs = [None] * num_processs

    for i in range(num_processs):
        processs[i] = multiprocessing.Process(target=fit_get_hyperparameters, args=(i, process_params[i], classifier_name,))
        processs[i].start()

    for i in range(num_processs):
        processs[i].join()

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

    start_training_time = time.time()
    clf_pipeline.fit(full_train_data, full_train_sub_classifications)
    training_time = time.time() - start_training_time

    print("Score for %s: %s%%" %
          (classifier_name, round(100 * clf_pipeline.score(full_validation_data,
                                                           full_validation_sub_classifications), 2)))

    with open(BEST_HYPERPARAMETERS_PATH + '/' + classifier_name + '.json', 'w') as f:
        json.dump(best_params, f, indent=4)

    joblib.dump(clf_pipeline, MODELS_PATH + classifier_name + '.joblib')

    return training_time


def set_up_training_for_classifier(num_processes, classifier_name):
    """
    Split up the parameters to test on so that they can be used in a multiprocessing environment

    :param num_processes:
    :param classifier_name:
    :return: the time it took to find the hyperparameters and train the classifier
    """

    open('temp/' + classifier_name + '.temp', "w").close()  # clear the temp file

    params_to_test = get_params_to_test(params_to_search[classifier_name])
    random.shuffle(params_to_test)

    params_to_test = params_to_test[:NUM_PARAMS_TO_TEST]  # test on a random selection of 500 parameter combinations

    process_params = list(split(params_to_test, num_processes))

    training_time = train(num_processes, process_params, classifier_name)

    print('Time to find best hyperparameters and train on full training set for %s classifier: %s minutes' %
          (classifier_name, round(training_time / 60, 2)))

    return training_time


if __name__ == '__main__':
    classifiers = ['MultinomialNB', 'SVC', 'RandomForestClassifier', 'SGDClassifier']

    parser = argparse.ArgumentParser()
    parser.add_argument('-clf', '--classifier_name', help='The name of the classifier. All classifiers will be '
                                                          'used if none specified.', choices=classifiers)
    args = parser.parse_args()

    try:
        full_train_data, full_train_sub_classifications = parse_reddit_data(DATA_PATH + '/training_data.txt')
        full_validation_data, full_validation_sub_classifications = parse_reddit_data(DATA_PATH + '/development_data.txt')

        partial_train_data, partial_train_sub_classifications = full_train_data[
                                                                :NUM_POSTS_TRAIN_HYPERPARAMETERS], full_train_sub_classifications[
                                                                                                   :NUM_POSTS_TRAIN_HYPERPARAMETERS]
        partial_validation_data, partial_validation_sub_classifications = full_validation_data[
                                                                      :NUM_POSTS_TRAIN_HYPERPARAMETERS], full_validation_sub_classifications[
                                                                                                         :NUM_POSTS_TRAIN_HYPERPARAMETERS]

        classifier_name = args.classifier_name
        if classifier_name is None:
            print('Using all classifiers: %s\n' % ', '.join(classifiers))

            num_processes = NUM_CORES / len(classifiers)
            que = queue.Queue()
            threads = list()

            for classifier in classifiers:
                if not os.path.exists(BEST_HYPERPARAMETERS_PATH + '/' + classifier + '.json'):
                    os.remove(BEST_HYPERPARAMETERS_PATH + '/' + classifier + '.json')

                thread = threading.Thread(target=lambda q, n, c: q.put(
                    set_up_training_for_classifier(n, c)), args=(que, num_processes, classifier,))
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()

            total_training_time = 0
            while not que.empty():
                total_training_time += que.get()

            print('\nTime to find best hyperparameters and train on full training set for all classifiers: %s minutes' %
                  round(total_training_time / 60, 2))
        else:
            print('Using %s classifier\n' % classifier_name)
            set_up_training_for_classifier(NUM_CORES, classifier_name)

    except FileNotFoundError:
        print('Please get data first')
