import os
import pickle
from multiprocessing import Pool

import nltk
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

nltk.download('punkt')
LABELS = "toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"


def load(path, max_lines=10000):
    df = pd.read_csv(path)[-max_lines:]  # 159571 Einträge insgesamt
    x = df['comment_text']
    y = [df[i] for i in LABELS]
    return x, y


def preprocess_comment(args):
    comment, stemmer = args
    tokenized_comment = []

    sentences = nltk.sent_tokenize(comment)
    for sentence in sentences:
        words = nltk.word_tokenize(sentence)
        tokenized_comment.append(" ".join([stemmer.stem(i) for i in words]))
    return " ".join(tokenized_comment)


def text_preprocess(data, stemmer):
    # unused
    res = []
    with Pool() as pool:
        res += pool.map_async(preprocess_comment,
                              ((comment, stemmer) for comment in data),
                              chunksize=int(len(data) / os.cpu_count())
                              ).get()
    return res


def fit_model(args, max_features=5000):
    x_train, y_train, x_test, y_test, label_index = args
    vectorizer = CountVectorizer(max_features=max_features)
    x_train_vectorized = vectorizer.fit_transform(x_train)
    x_test_vectorized = vectorizer.transform(x_test)

    model = MultinomialNB()
    model.fit(x_train_vectorized, y_train[label_index])

    with open(f"models/{LABELS[label_index]}.pickle", "wb") as f:
        pickle.dump((vectorizer, model), f)

    return model.score(x_test_vectorized, y_test[label_index])


def main():
    x_test_start, y_test_start = load("test_pretty.csv", max_lines=159_571)
    x_train, y_train = load("train_pretty.csv", max_lines=159_571)

    # stemmer = SnowballStemmer("english")
    # x_train = text_preprocess(x_train, stemmer)
    # x_test = text_preprocess(x_test_start, stemmer)

    with Pool() as pool:
        new_scores = pool.map_async(fit_model,
                                    [(x_train, y_train, x_test_start, y_test_start, i) for i in range(len(LABELS))])
        for i, score in enumerate(new_scores.get()):
            print(f"{LABELS[i]}: {score}")


if __name__ == '__main__':
    main()
