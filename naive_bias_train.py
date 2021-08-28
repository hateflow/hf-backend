import os
import pickle

import nltk
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB

LABELS = "toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"


def load(path, max_lines=10000):
    df = pd.read_csv(path)[-max_lines:]  # 159571 lines
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


def fit_model(x_train_vectorized, y_train, label_index):
    model = MultinomialNB()
    model.fit(x_train_vectorized, y_train[label_index])

    with open(f"models/{LABELS[label_index]}.pickle", "wb") as f:
        pickle.dump(model, f)

    return model


def plot_threshold_scores(label_index, model: MultinomialNB, x_test_vectorized, y_test):
    y_test = y_test[label_index]

    true_scores = dict()
    false_scores = dict()

    x_test_true = x_test_vectorized[y_test == True]
    y_test_true = y_test[y_test == True]
    x_test_false = x_test_vectorized[y_test == False]
    y_test_false = y_test[y_test == False]

    for i in range(1001):
        threshold = i / 1000

        y_true_pred = (model.predict_proba(x_test_true)[:, 1] > threshold) * 1
        true_scores[threshold] = accuracy_score(y_test_true, y_true_pred)

        y_false_pred = (model.predict_proba(x_test_false)[:, 1] > threshold) * 1
        false_scores[threshold] = accuracy_score(y_test_false, y_false_pred)

    plt.cla()
    plt.plot(true_scores.keys(), true_scores.values(), label=LABELS[label_index])
    plt.plot(false_scores.keys(), false_scores.values(), label=f"not {LABELS[label_index]}")

    best_threshold = max(true_scores.keys(), key=lambda k: true_scores[k] + false_scores[k])
    best_score = (true_scores[best_threshold] + false_scores[best_threshold]) / 2
    plt.plot([best_threshold, best_threshold], [0, 1],
             label=f"best score: {best_score:.3f} using threshold {best_threshold}")

    plt.legend()
    plt.title(LABELS[label_index])
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/{LABELS[label_index]}.jpg")


def main():
    nltk.download('punkt')
    x_test, y_test = load("test_pretty.csv", max_lines=159_571)
    x_train, y_train = load("train_pretty.csv", max_lines=159_571)

    vectorizer = CountVectorizer(max_features=5000)
    x_train_vectorized = vectorizer.fit_transform(x_train)
    x_test_vectorized = vectorizer.transform(x_test)
    with open("models/vectorizer.pickle", "wb") as f:
        pickle.dump(vectorizer, f)

    models = []
    for label_index, label in enumerate(LABELS):
        model = fit_model(x_train_vectorized, y_train, label_index)
        print(f"{label}: {model.score(x_test_vectorized, y_test[label_index])}")
        models.append(model)

    for label_index, model in enumerate(models):
        plot_threshold_scores(label_index, model, x_test_vectorized, y_test)


if __name__ == '__main__':
    main()
