"""

usage: predict_subreddit.py [-h]
                            [-clf {MultinomialNB,SVC,RandomForestClassifier,SGDClassifier}]
                            [--url URL]

optional arguments:
  -h, --help            show this help message and exit
  -clf {MultinomialNB,SVC,RandomForestClassifier,SGDClassifier}, --classifier_name {MultinomialNB,SVC,RandomForestClassifier,SGDClassifier}
                        The name of the classifier. A random classifier will
                        be used if none specified.
  --url URL             The url of a post to test. If none, a random post will
                        be used.

For example: python predict_subreddit.py -clf MultinomialNB

__author__ = Ian Randman
"""
import os
import random
import argparse

from joblib import load

from praw.models import MoreComments

from get_data import subreddit_names
from get_data import num_to_sub

from get_data import reddit

from find_hyperparameters import MODELS_PATH


def predict_random_post(classifier_name):
    """
    Get a random post from one of 8 subreddits and classify it using a specified classifier.

    :param classifier_name: the name of the estimator
    :return: none
    """

    sub_num = random.randint(0, len(subreddit_names) - 1)

    submission = reddit.subreddit(num_to_sub[sub_num][2:]).random()
    data = submission.selftext + ' '
    for comment in submission.comments.list():
        if isinstance(comment, MoreComments):
            continue

        data += comment.body + ' '

    data = data.replace('\n', ' ')
    data = data.replace('\r', ' ')

    print(submission.shortlink)
    print('Title: ', submission.title)
    print('Actual subreddit: ', submission.subreddit_name_prefixed)

    print('\nPredicting subreddit...\n')

    clf_pipeline = load(MODELS_PATH + classifier_name + '.joblib')
    prediction = num_to_sub[clf_pipeline.predict([data])[0]]

    print('Predicted Subreddit', prediction)


def predict_post(classifier_name, url):
    """
    Get a post a specified Reddit URL and classify it using a specified classifier.

    :param classifier_name: classifier_name: the name of the estimator
    :param url: the url of the post to predict
    :return: none
    """

    submission = reddit.submission(url=url)
    data = submission.selftext + ' '
    for comment in submission.comments.list():
        if isinstance(comment, MoreComments):
            continue

        data += comment.body + ' '

    data = data.replace('\n', ' ')
    data = data.replace('\r', ' ')

    print(submission.shortlink)
    print('Title: ', submission.title)
    print('Actual subreddit: ', submission.subreddit_name_prefixed)

    print('\nPredicting subreddit...\n')

    clf_pipeline = load(MODELS_PATH + classifier_name + '.joblib')
    prediction = num_to_sub[clf_pipeline.predict([data])[0]]

    print('Predicted Subreddit', prediction)


if __name__ == '__main__':
    classifiers = ['MultinomialNB', 'SVC', 'RandomForestClassifier', 'SGDClassifier']

    parser = argparse.ArgumentParser()
    parser.add_argument('-clf', '--classifier_name', help='The name of the classifier. A random classifier will be '
                                                          'used if none specified.', choices=classifiers)

    parser.add_argument('--url', help='The url of a post to test. If none, a random post will be used.')

    args = parser.parse_args()

    classifier_name = args.classifier_name
    if classifier_name is None:
        classifier_name = random.choice(classifiers)

    print('Using %s classifier\n' % classifier_name)

    if os.path.exists(MODELS_PATH + classifier_name + '.joblib'):
        url = args.url
        if url is None:
            predict_random_post(classifier_name)
        else:
            predict_post(classifier_name, url)

    else:
        print('Please make sure %s is trained first' % classifier_name)
