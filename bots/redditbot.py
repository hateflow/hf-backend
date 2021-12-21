import praw
import time
import requests
import json

def comment_comment(comment, properties): # building the sentence and commenting
    text = "Beep boop. I detected "
    propsleft = []
    for i in properties:
        propsleft.append(i)
    while propsleft != []:
        if len(propsleft) == 1 and len(properties) > 1:
            text = text + " and "
        elif len(propsleft) != len(properties):
            text = text + ", "
        if "identity_hate" in propsleft:
            text = text + "identity hate"
            propsleft.remove("identity_hate")
        elif "insult" in propsleft:
            text = text + "insult"
            propsleft.remove("insult")
        elif "obscene" in propsleft:
            text = text + "obscenity"
            propsleft.remove("obscene")
        elif "severe_toxic" in propsleft:
            text = text + "severe toxicity"
            propsleft.remove("severe_toxic")
            try:
                propsleft.remove("toxic")
            except:
                continue
        elif "threat" in propsleft:
            text = text + "threat"
            propsleft.remove("threat")
        elif "toxic" in propsleft:
            text = text + "toxicity"
            propsleft.remove("toxic")
    text = text + ".\n\nI am a bot developed by HateFlow. This action was performed automatically."
    print(text)
    # uncomment to send reply
    #comment.reply(text)

def check_comment(comment): # API call
    API_URL = "http://api.hateflow.de/"
    response = requests.get(f"{API_URL}simpleCheck?text={comment.body}")
    if response.status_code != 200:
        return "Something went wrong!"
    else:
        data = json.loads(response.text)
        #print(data['results'])
        return data['results']

def scan_comment(url): # scanning all comments of one post
    reddit = login()
    submission = reddit.submission(url=url)
    scan_for_comments(submission)

def scan_for_comments(submission): # scanning all comments of one post
    submission.comments.replace_more(limit=None)
    flat_comments = submission.comments.list() # all comments of submission
    print("There are "+str(len(flat_comments))+" comments to scan.")
    commentcount = 1
    for comment in flat_comments:
        print("---------new-comment---------"+"    comment "+str(commentcount)+"/"+str(len(flat_comments)))
        #print("comment_id: "+comment.id)
        print(comment.body)
        result = check_comment(comment)
        if 1 in result.values():
            properties = []
            i=0
            while i < len(list(result)):
                if list(result.values())[i] == 1 and list(result.keys())[i] != "toxic":
                    properties.append(list(result.keys())[i])
                i+=1
            print(properties)
            comment_comment(comment, properties)
        commentcount += 1

def scan_subreddit(subName): # scanning all posts of one subreddit
    reddit = login()
    subreddit = reddit.subreddit(subName)
    for submission in subreddit.hot(limit=2): # chooses the 2 hottest submissions of subreddit
        print("-------------------------submission-------------------------")
        print("r/"+subName)
        print("title: "+submission.title)           # Output: the submission's title
        #print("score: "+str(submission.score))     # Output: the submission's score
        print("id: "+str(submission.id))            # Output: the submission's ID
        #print("url: "+submission.url)              # Output: the URL the submission points to
                                                    # or the submission's URL if it's a self post
        time.sleep(2) # max requests with Reddit-API: 1 per 2 seconds
        print("--------------------------comments--------------------------")
        scan_for_comments(submission)

def login():
    reddit = praw.Reddit(
        ...
        )
    return reddit

# uncomment to scan one post for comments
scan_comment("https://www.reddit.com/r/me_irl/comments/qs6stp/me_irl/") #scan_comment("Link to post")

# uncomment to scan one post for comments
scan_subreddit("me_irl") #scan_subreddit("name_of_subreddit") to scan the 2 hottest submissions of a specific subreddit

# uncomment line 41 to also send reply