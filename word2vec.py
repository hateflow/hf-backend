import os
from multiprocessing import Pool

import gensim.downloader as gensim_api
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

LABELS = "toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"


def load(path, max_lines=10000):
    df = pd.read_csv(path)[-max_lines:]  # 159571 lines
    x = df['comment_text']
    y = [df[i] for i in LABELS]
    return x, y


def tokenize(comment: str):
    # split into tokens
    comment = word_tokenize(comment.lower())
    # remove punctuation and stopwords
    comment = [i for i in comment if i.isalpha() and i not in english_stopwords and i in corpus]
    return comment


def vectorize(words: list):
    return np.mean(corpus[words], axis=0)


def process_chunk(comments: list, starting_index=0):
    tokenized = [tokenize(comment) for comment in comments]

    removed_indices = filter(lambda i: not bool(tokenized[i]), range(len(tokenized)))
    removed_indices = [i + starting_index for i in removed_indices]
    tokenized = filter(None, tokenized)  # remove empty lists
    vectorized = [vectorize(words) for words in tokenized]

    return vectorized, removed_indices


def preprocess(comments: list, workers: int = None):
    if workers is None:
        workers = os.cpu_count()

    chunk_size = len(comments) / workers
    chunk_starts = [int(chunk_size * i) for i in range(workers)]
    chunk_ends = chunk_starts[1:] + [len(comments)]
    with Pool() as p:
        results = []
        for start, end in zip(chunk_starts, chunk_ends):
            results.append(p.apply_async(
                func=process_chunk,
                args=(comments[start:end], start),
            ))

        tokenized, removed_indices = [], []
        for chunk in results:
            comments, indices = chunk.get()
            tokenized.extend(comments)
            removed_indices.extend(indices)
    return tokenized, removed_indices


print("Loading libraries...")
nltk.download("punkt")
nltk.download("stopwords")
corpus = gensim_api.load("glove-twitter-200")
english_stopwords = set(stopwords.words("english"))

if __name__ == "__main__":
    x_train, y_train = load("train_pretty.csv", max_lines=159_571)
    x_test, y_test = load("test_pretty.csv", max_lines=159_571)

    x_train_tokenized, train_removed = preprocess(x_train)
    x_test_tokenized, test_removed = preprocess(x_test)
