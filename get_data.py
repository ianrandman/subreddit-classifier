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

subreddit_names = ['nba', 'nhl', 'nfl']
sub_to_num = {'r/nba': 0, 'r/nhl': 1, 'r/nfl': 2}
num_to_sub = {0: 'r/nba', 1: 'r/nhl', 2: 'r/nfl'}


def save_posts(submissions, subreddit_name, thread_num, num_posts_to_get, num_total_posts):
    # output = open('train', "w", encoding='utf-8')
    # output.write("# this file contains all the data from the subreddits to be tested\n# subreddit_name, title\n\n")

    output = ''

    starting_post_num = thread_num * num_posts_to_get

    post_number = 0
    for submission in submissions:#reddit.subreddit(subreddit_name).top(time_filter='all', limit=num_total_posts):
        if starting_post_num <= post_number < starting_post_num + num_posts_to_get:
            print("Post number: " + str(post_number))
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
    num_posts_to_get = 5
    num_total_posts = 100

    threads = list()

    for subreddit_name in subreddit_names:
        open(subreddit_name + '.txt', 'w', encoding='utf-8').close()

        submissions = reddit.subreddit(subreddit_name).top(time_filter='all', limit=num_total_posts)

        for thread_num in range(int(num_total_posts / num_posts_to_get)):
            print("Thread num: " + str(thread_num))
            thread = threading.Thread(target=save_posts, args=(submissions, subreddit_name, thread_num, num_posts_to_get, num_total_posts))
            thread.start()
            threads.append(thread)

            sleep(0.4)

    for thread in threads:
        thread.join()