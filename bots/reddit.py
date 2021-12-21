import time

import praw
from dotenv import load_dotenv

from bots.bots_base import process_message
import os


def scan_comment(url):  # scanning all comments of one post
    reddit = login()
    submission = reddit.submission(url=url)
    scan_for_comments(submission)


def scan_for_comments(submission):  # scanning all comments of one post
    submission.comments.replace_more(limit=None)
    flat_comments = submission.comments.list()  # all comments of submission
    print("There are " + str(len(flat_comments)) + " comments to scan.")
    for comment in flat_comments:
        warning = process_message(comment.body)
        if warning:
            response = f"Beep boop. {warning}"
            print(response)
            # comment.reply(response)


def scan_subreddit(subName):  # scanning all posts of one subreddit
    reddit = login()
    subreddit = reddit.subreddit(subName)
    for submission in subreddit.hot(limit=2):  # chooses the 2 hottest submissions of subreddit
        print("-------------------------submission-------------------------")
        print("r/" + subName)
        print("title: " + submission.title)  # Output: the submission's title
        print("id: " + str(submission.id))  # Output: the submission's ID
        time.sleep(2)  # max requests with Reddit API: 1 per 2 seconds
        print("--------------------------comments--------------------------")
        scan_for_comments(submission)


def login():
    load_dotenv()

    reddit = praw.Reddit(
        client_id=os.getenv("R_CLIENT_ID"),
        client_secret=os.getenv("R_CLIENT_SECRET"),
        username=os.getenv("R_USERNAME"),
        password=os.getenv("R_PASSWORD"),
        user_agent=os.getenv("R_USER_AGENT"),
    )
    return reddit


# scan_comment("https://www.reddit.com/r/me_irl/comments/qs6stp/me_irl/")  # scan_comment("Link to post")
scan_subreddit("me_irl")  # scan the 2 hottest submissions of a specific subreddit
