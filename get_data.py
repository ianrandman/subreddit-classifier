"""
This should be used to retrieve Reddit posts using Reddit's Praw. Then split the data.

usage: get_data.py

__author__ = Ian Randman
__author__ = David Dunlap
"""

import random
import threading
import numpy as np
import os
import shutil

import praw
from praw.models import MoreComments

from io import open

CLIENT_ID = 'E9cBapTtE2vUbQ'
CLIENT_SECRET = 'K4eUnFYNbtD-S32h7EpoaOmGVc8'
PASSWORD = 'awhMgfH4FBnYD24'
USERAGENT = 'a subreddit classifier'
USERNAME = 'cheermeup12'

reddit = praw.Reddit(client_id=CLIENT_ID, client_secret=CLIENT_SECRET,
                     password=PASSWORD, user_agent=USERAGENT,
                     username=USERNAME)

subreddit_names = ['nba', 'nhl', 'nfl', 'mlb', 'soccer', 'formula1', 'CFB', 'sports']
sub_to_num = {'r/nba': 0, 'r/nhl': 1, 'r/nfl': 2, 'r/mlb': 3, 'r/soccer': 4, 'r/formula1': 5, 'r/CFB': 6, 'r/sports': 7}
num_to_sub = {0: 'r/nba', 1: 'r/nhl', 2: 'r/nfl', 3: 'r/mlb', 4: 'r/soccer', 5: 'r/formula1', 6: 'r/CFB', 7: 'r/sports'}

DATA_PATH = os.path.dirname(os.path.abspath(__file__)) + '/data/'


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


def save_posts(subreddit_name, limit):
    """
    From a specified subreddit, retrieve a specified number of posts from top of all tim. Save them to a file in the
    data folder as [subreddit_name].txt, where each line is in the format
    (subreddit_name, [concatenation of comments from post]).

    :param subreddit_name: the subreddit to get posts from
    :param limit: the number of posts to get
    :return: none
    """

    data_file_path = DATA_PATH + subreddit_name + '.txt'

    if os.path.isfile(data_file_path):
        os.remove(data_file_path)

    with open(data_file_path, 'a', encoding='utf-8') as output:
        output.write("# this file contains all the data from the " + subreddit_name + " to be tested\n# subreddit_name, title\n\n")

        post_num = 1
        for submission in reddit.subreddit(subreddit_name).top(time_filter='all', limit=limit):
            data = submission.selftext + ' '
            for comment in submission.comments.list():
                if isinstance(comment, MoreComments):
                    continue

                data += comment.body + ' '

            data = data.replace('\n', ' ')
            data = data.replace('\r', ' ')

            output.write(submission.subreddit_name_prefixed + ',' + data + '\n')

            print("Post", post_num, "from", subreddit_name, "downloaded")

            post_num += 1


def split_data():
    """
    Split data from all subreddits (read in from that subreddit's data file) into training, development, and testing.
    The splits for training, development, and testing are 50%, 25%, 25%, respectively.

    Save each of the splits to file, where each line is in the format (subreddit_name, [concatenation of comments
    from post]).

    :return: none
    """

    training_data_file = open(DATA_PATH + '/training_data.txt', 'w', encoding='utf-8')
    development_data_file = open(DATA_PATH + '/development_data.txt', 'w', encoding='utf-8')
    test_data_file = open(DATA_PATH + '/testing_data.txt', 'w', encoding='utf-8')

    training_data, development_data, test_data = list(), list(), list()

    for subreddit_name in subreddit_names:
        data = np.asarray(file_list(DATA_PATH + subreddit_name + '.txt'))
        random.shuffle(data)

        train, dev, test = np.split(data, [int(0.5 * len(data)), int(0.75 * len(data))])

        training_data.extend(train)
        development_data.extend(dev)
        test_data.extend(test)

    random.shuffle(training_data)
    random.shuffle(development_data)
    random.shuffle(test_data)

    for post in training_data:
        training_data_file.write(post + '\n')

    for post in development_data:
        development_data_file.write(post + '\n')

    for post in test_data:
        test_data_file.write(post + '\n')

    training_data_file.close()
    development_data_file.close()
    test_data_file.close()


if __name__ == '__main__':
    if os.path.exists(DATA_PATH):
        shutil.rmtree(DATA_PATH)

    os.makedirs(DATA_PATH)

    threads = list()
    limit = 1000
    print('Downloading', limit, 'posts from each subreddit\n')

    for subreddit_name in subreddit_names:
        thread = threading.Thread(target=save_posts, args=(subreddit_name, limit,))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    split_data()

    print('\nDownload finished')
