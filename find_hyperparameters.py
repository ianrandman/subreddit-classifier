"""
Perform a random search over a set of possible hyperparameters for a specified classifier. Each set of
hyperparameters to test will be trained using a subset of the training data and validated on a subset of the
development data.

usage: find_hyperparameters.py [-h]
                               [-clf {MultinomialNB,SVC,RandomForestClassifier,SGDClassifier}]

optional arguments:
  -h, --help            show this help message and exit
  -clf {MultinomialNB,SVC,RandomForestClassifier,SGDClassifier}, --classifier_name {MultinomialNB,SVC,RandomForestClassifier,SGDClassifier}
                        The name of the classifier. All classifiers will be
                        used if none specified.

For example: python find_hyperparameters.py MultinomialNB

__author__ = Ian Randman
__author__ = David Dunlap
"""

import argparse
import math
import queue
import random
import os
import multiprocessing
import threading
import re
import itertools
import time
import warnings

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
from parameters import params_to_search  # these are the parameters to search over

from io import open
import json

warnings.filterwarnings("ignore")

NUM_PARAMS_TO_TEST = 500
NUM_CORES = multiprocessing.cpu_count()

NUM_POSTS_TRAIN_HYPERPARAMETERS = 500  # the number of posts to train and test on when finding best hyperparameters

BEST_HYPERPARAMETERS_PATH = os.path.dirname(os.path.abspath(__file__)) + '/best_hyperparameters/'
TEMP_PATH = os.path.dirname(os.path.abspath(__file__)) + '/temp/'
MODELS_PATH = os.path.dirname(os.path.abspath(__file__)) + '/models/'


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


def fit_get_hyperparameters(data, num_process, params_to_test, classifier_name):
    """
    Find the best set of hyperparameters for a specified classifier. Save the best score on validation data and best
    parameters to a file.

    :param data: a 6-tuple containing the full training_data, subset of train data, and the full validation data
    :param num_process: the process number for the process
    :param params_to_test: the collection of parameters to test on
    :param classifier_name: the name of the estimator
    :return: none
    """

    partial_train_data, partial_train_sub_classifications, partial_validation_data, partial_validation_sub_classifications = data[2:]

    best_params = None
    best_score = 0

    run_num = 1

    for parameters in params_to_test:
        clf_pipeline = get_new_pipeline(classifier_name)

        clf_pipeline.set_params(**parameters)
        clf_pipeline.fit(partial_train_data, partial_train_sub_classifications)

        score = clf_pipeline.score(partial_validation_data, partial_validation_sub_classifications)

        if score > best_score:
            best_score = score
            best_params = parameters

        print("Process number: %s/%s -- model trained and tested for %s -- run number: %s/%s" %
              (num_process + 1, NUM_CORES, classifier_name, run_num, math.ceil(NUM_PARAMS_TO_TEST / NUM_CORES)))

        run_num += 1


    with open(TEMP_PATH + classifier_name + '.temp', "ab") as temp_file:
        pickle.dump((best_score, best_params), temp_file)


def train(data, process_params, classifier_name):
    """
    Train to find the best set of hyperparameters for a specified classifier. Use multiprocessing to break up the
    task into a given number of processes to speed up computation. Finally, train on the full training set and save
    the results.

    :param data: a 6-tuple containing the full training_data, subset of train data, and the full validation data
    :param classifier_name: the name of the estimator
    :return: the time it took to find the hyperparameters
    """

    start_training_time = time.time()

    # full_train_data, full_train_sub_classifications = data[:2]

    processes = [None] * NUM_CORES

    for i in range(NUM_CORES):
        processes[i] = multiprocessing.Process(target=fit_get_hyperparameters,
                                              args=(data, i, process_params[i], classifier_name,))
        processes[i].start()

    for i in range(NUM_CORES):
        processes[i].join()

    best_score = 0
    best_params = None
    with open(TEMP_PATH + classifier_name + '.temp', "rb") as temp_file:
        try:
            while True:
                score, parameters = pickle.load(temp_file)

                if score > best_score:
                    best_score = score
                    best_params = parameters
        except EOFError:  # reached end of file
            pass

    os.remove(TEMP_PATH + classifier_name + '.temp')

    # clf_pipeline = get_new_pipeline(classifier_name)
    # clf_pipeline.set_params(**best_params)
    #
    # clf_pipeline.fit(full_train_data, full_train_sub_classifications)
    training_time = time.time() - start_training_time

    # score = round(100 * clf_pipeline.score(full_validation_data,
    #                                        full_validation_sub_classifications), 2)

    with open(BEST_HYPERPARAMETERS_PATH + '/' + classifier_name + '.json', 'w') as f:
        json.dump(best_params, f, indent=4)

    # joblib.dump(clf_pipeline, MODELS_PATH + classifier_name + '.joblib')

    return training_time


def set_up_training_for_classifier(data, classifier_name):
    """
    Split up the parameters to test on so that they can be used in a multiprocessing environment

    :param data: a 6-tuple containing the full training_data, subset of train data, and the full validation data
    :param classifier_name: the name of the estimator
    :return: the time it took to find the hyperparameters
    """

    open(TEMP_PATH + classifier_name + '.temp', "w").close()  # clear the temp file

    params_to_test = get_params_to_test(params_to_search[classifier_name])
    random.shuffle(params_to_test)

    params_to_test = params_to_test[:NUM_PARAMS_TO_TEST]  # test on a random selection of 500 parameter combinations
    process_params = list(split(params_to_test, NUM_CORES))

    training_time = train(data, process_params, classifier_name)
    training_time_minutes = training_time / 60

    return training_time_minutes


if __name__ == '__main__':
    classifiers = ['MultinomialNB', 'SVC', 'RandomForestClassifier', 'SGDClassifier']

    parser = argparse.ArgumentParser()
    parser.add_argument('-clf', '--classifier_name', help='The name of the classifier. All classifiers will be '
                                                          'used if none specified.', choices=classifiers)
    args = parser.parse_args()

    print('Using %s cores' % NUM_CORES)
    print('Testing on %s parameters\n' % NUM_PARAMS_TO_TEST)

    if not os.path.exists(BEST_HYPERPARAMETERS_PATH):
        os.makedirs(BEST_HYPERPARAMETERS_PATH)

    if not os.path.exists(TEMP_PATH):
        os.makedirs(TEMP_PATH)

    if not os.path.exists(MODELS_PATH):
        os.makedirs(MODELS_PATH)

    try:
        full_train_data, full_train_sub_classifications = parse_reddit_data(DATA_PATH + '/training_data.txt')
        full_validation_data, full_validation_sub_classifications = parse_reddit_data(
            DATA_PATH + '/development_data.txt')

        partial_train_data, partial_train_sub_classifications = full_train_data[
                                                                :NUM_POSTS_TRAIN_HYPERPARAMETERS], full_train_sub_classifications[
                                                                                                   :NUM_POSTS_TRAIN_HYPERPARAMETERS]
        partial_validation_data, partial_validation_sub_classifications = full_validation_data[
                                                                          :NUM_POSTS_TRAIN_HYPERPARAMETERS], full_validation_sub_classifications[
                                                                                                             :NUM_POSTS_TRAIN_HYPERPARAMETERS]

        data = full_train_data, full_train_sub_classifications, partial_train_data, partial_train_sub_classifications, partial_validation_data, partial_validation_sub_classifications

        print('Number of runs for each process is: %s' % (math.ceil(NUM_PARAMS_TO_TEST / NUM_CORES)))
        classifier_name = args.classifier_name
        if classifier_name is None:
            print('Using all classifiers: %s' % ', '.join(classifiers))

            # scores = list()
            training_times = list()

            for classifier in classifiers:
                print('\nFinding best hyperparameters for %s classifier...' % classifier)
                if os.path.exists(BEST_HYPERPARAMETERS_PATH + '/' + classifier + '.json'):
                    os.remove(BEST_HYPERPARAMETERS_PATH + '/' + classifier + '.json')

                training_time_minutes = set_up_training_for_classifier(data, classifier)
                # scores.append(score)
                training_times.append(training_time_minutes)

            print()
            total_training_time = 0
            for i in range(len(classifiers)):
                # score = scores[i]
                training_time_minutes = training_times[i]

                total_training_time += training_time_minutes

                # print("\nScore for %s: %s%%" %
                #       (classifiers[i], score))

                print('Time to find best hyperparameters and train on full training set for %s classifier: %s minutes'
                      % (classifiers[i], round(training_time_minutes, 2)))

            print('\nTime to find best hyperparameters and train on full training set for all classifiers: %s minutes' %
                  round(total_training_time, 2))

        else:
            print('Using %s classifier\n' % classifier_name)
            print('Finding best hyperparameters for %s classifier...' % classifier_name)

            training_time_minutes = set_up_training_for_classifier(data, classifier_name)

            # print("\nScore for %s: %s%%" %
            #       (classifier_name, score))

            print('Time to find best hyperparameters and train on full training set for %s classifier: %s minutes'
                  % (classifier_name, round(training_time_minutes)))

    except FileNotFoundError:
        print('Please get data first')
