import random
import threading
from time import sleep

import praw
from praw.models import MoreComments

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

subreddit_names = ['nba', 'nhl', 'nfl']
sub_to_num = {'r/nba': 0, 'r/nhl': 1, 'r/nfl': 2}
num_to_sub = {0: 'r/nba', 1: 'r/nhl', 2: 'r/nfl'}


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


def parse_reddit_data():
    data = list()
    sub_classifications = list()

    posts = file_list('train')
    for post in posts:
        post_split = post.split(',', 1)

        data.append(post_split[1])
        sub_classifications.append(sub_to_num[post_split[0]])

    return data, sub_classifications


def train():
    # categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
    # twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
    #
    #
    # data = [' aa bb cc cc cc dd dd ee', 'ee ee ee dd cc bb bb aa']
    #save_posts()
    data, sub_classifications = parse_reddit_data()

    count_vect = CountVectorizer()
    counts = count_vect.fit_transform(data)

    clf = MultinomialNB().fit(counts, sub_classifications)

    dump(clf, 'classifier.joblib')
    dump(count_vect, 'count_vectorizer.joblib')


def predict_sub():
    clf = load('classifier.joblib')
    count_vect = load('count_vectorizer.joblib')

    submissions = list()
    test_data = list()
    test_subreddit_categories = list()

    for subreddit_name in subreddit_names:
        for submission in reddit.subreddit(subreddit_name).new(limit=20):
            data = submission.selftext + ' '
            for comment in submission.comments.list():
                if isinstance(comment, MoreComments):
                    continue

                data += comment.body + ' '

            data = data.replace('\n', ' ')
            data = data.replace('\r', ' ')
            
            test_data.append(data)
            test_subreddit_categories.append(sub_to_num[submission.subreddit_name_prefixed])

    random.shuffle(submissions)

    for submission in submissions:
        test_data.append(submission.title)
        test_subreddit_categories.append(sub_to_num[submission.subreddit_name_prefixed])

    new_counts = count_vect.transform(test_data)

    predicted = clf.predict(new_counts)

    for actual_sub, title, predicted_sub in zip(test_subreddit_categories, test_data, predicted):
        if actual_sub != predicted_sub:
            print('%s: %r => %s' % (num_to_sub[actual_sub], title, num_to_sub[predicted_sub]))

    num_correct = 0
    num_total = 0
    for subreddit_name, category in zip(test_subreddit_categories, predicted):
        if subreddit_name == category:
            num_correct += 1

        num_total += 1

    print('%s/%s correct' % (num_correct, num_total))

    print(np.mean(test_subreddit_categories == predicted))


def save_posts(subreddit_name, thread_num, num_posts_to_get, num_total_posts):
    # output = open('train', "w", encoding='utf-8')
    # output.write("# this file contains all the data from the subreddits to be tested\n# subreddit_name, title\n\n")

    output = ''

    starting_post_num = thread_num * num_posts_to_get

    post_number = 0
    for submission in reddit.subreddit(subreddit_name).top(time_filter='all', limit=num_total_posts):
        if starting_post_num <= post_number < starting_post_num + num_posts_to_get:
            data = submission.selftext + ' comment_separator '
            for comment in submission.comments.list():
                if isinstance(comment, MoreComments):
                    continue

                data += comment.body + ' comment_separator '

            data = data.replace('\n', ' ')
            data = data.replace('\r', ' ')

            output += submission.subreddit_name_prefixed + ',' + data + '\n'

        post_number += 1

    output_file = open(subreddit_name + '.txt', "a", encoding='utf-8')
    output_file.write(output)

        # output.write(submission.subreddit_name_prefixed + ',' + data + '\n')


if __name__ == '__main__':
    num_posts_to_get = 10
    num_total_posts = 1000

    threads = list()

    for subreddit_name in subreddit_names:
        for thread_num in range(int(num_total_posts / num_posts_to_get)):
            print(thread_num)
            thread = threading.Thread(target=save_posts, args=(subreddit_names[0], thread_num, num_posts_to_get, num_total_posts))
            thread.start()
            threads.append(thread)

    for thread in threads:
        thread.join()

    # if TRAIN:
    #     train()
    # else:
    #     predict_sub()
