import random

import praw
import json
import numpy as np
from joblib import dump, load

from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups
from sklearn import svm

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

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

subreddit_names = ['AmItheAsshole', 'tifu']
sub_to_num = {'r/AmItheAsshole' : 0, 'r/tifu' : 1}
num_to_sub = {0 : 'r/AmItheAsshole', 1 : 'r/tifu'}


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
            if line[0] != '#' and line[0] != '\n':
                f_list.append(line.strip('\n'))
    return f_list


def parse_reddit_data():
    titles = list()
    sub_classifications = list()

    posts = file_list('data')
    for post in posts:
        data = post.split(',', 1)

        titles.append(data[1])
        sub_classifications.append(sub_to_num[data[0]])

    return titles, sub_classifications



def save_posts():
    output = open('data', "w", encoding='utf-8')

    for subreddit_name in subreddit_names:
        for submission in reddit.subreddit(subreddit_name).top(time_filter='all', limit=20):
            #submission_dict = vars(submission)

            output.write(submission.subreddit_name_prefixed + ',' + submission.title + '\n')


def train():
    # categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
    # twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
    #
    #
    # data = [' aa bb cc cc cc dd dd ee', 'ee ee ee dd cc bb bb aa']
    titles, sub_classifications = parse_reddit_data()

    count_vect = CountVectorizer()
    counts = count_vect.fit_transform(titles)

    clf = MultinomialNB().fit(counts, sub_classifications)

    dump(clf, 'classifier.joblib')
    dump(count_vect, 'count_vectorizer.joblib')


def predict_sub():
    clf = load('classifier.joblib')
    count_vect = load('count_vectorizer.joblib')

    submissions = list()
    test_titles = list()
    test_subreddit_categories = list()

    for subreddit_name in subreddit_names:
        submissions.extend(reddit.subreddit(subreddit_name).new(limit=500))

    random.shuffle(submissions)

    for submission in submissions:
        test_titles.append(submission.title)
        test_subreddit_categories.append(sub_to_num[submission.subreddit_name_prefixed])

    new_counts = count_vect.transform(test_titles)

    predicted = clf.predict(new_counts)

    for title, category in zip(test_titles, predicted):
        print('%r => %s' % (title, num_to_sub[category]))

    num_correct = 0
    num_total = 0
    for subreddit_name, category in zip(test_subreddit_categories, predicted):
        if subreddit_name == category:
            num_correct += 1

        num_total += 1

    print('%s/%s correct' % (num_correct, num_total))



if __name__ == '__main__':
    if TRAIN:
        train()
    else:
        predict_sub()
