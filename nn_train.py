import os

import nltk
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB

from word2vec import preprocess

LABELS = "toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"
torch.set_num_threads(os.cpu_count())


def load(path, max_lines=10000):
    df = pd.read_csv(path)[-max_lines:]  # 159571 lines
    x = df['comment_text']
    y = df[[i for i in LABELS]]
    return x, y


def fit_model(x_train_vectorized, y_train):
    input_dimensions, hidden = 200, 100
    model = torch.nn.Sequential(
        torch.nn.Linear(input_dimensions, hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden, 6),
        torch.nn.Sigmoid(),
    )

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    for i in range(500):
        y_pred = model(x_train_vectorized)
        loss = loss_fn(y_pred, y_train)
        print(i, loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), "models/nn.pt")

    return model


def plot_threshold_scores(model: torch.nn.Sequential, x_test: torch.tensor, y_test: torch.tensor):
    print("#"*20, "EVALUATION", "#"*20)
    for label_index, label in enumerate(LABELS):
        print(label)
        true_scores = dict()
        false_scores = dict()

        x_test_true = x_test[y_test[:, label_index] == 1]
        y_test_true = y_test[y_test[:, label_index] == 1]
        x_test_false = x_test[y_test[:, label_index] == 0]
        y_test_false = y_test[y_test[:, label_index] == 0]

        for i in range(1001):
            threshold = i / 1000

            y_true_pred = (model(x_test_true)[:, label_index] > threshold) * 1
            true_scores[threshold] = accuracy_score(y_test_true[:, label_index], y_true_pred)

            y_false_pred = (model(x_test_false)[:, label_index] > threshold) * 1
            false_scores[threshold] = accuracy_score(y_test_false[:, label_index], y_false_pred)

        plt.cla()
        plt.plot(true_scores.keys(), true_scores.values(), label=label)
        plt.plot(false_scores.keys(), false_scores.values(), label=f"not {label}")

        best_threshold = max(true_scores.keys(), key=lambda k: true_scores[k] + false_scores[k])
        print(f"    best threshold: {best_threshold}")

        best_score = (true_scores[best_threshold] + false_scores[best_threshold]) / 2
        print(f"    true score: {true_scores[best_threshold]:.3f}    false score: {false_scores[best_threshold]:.3f}")
        print(f"    --> unweighted average: {best_score:.4f}")

        plt.plot([best_threshold, best_threshold], [0, 1],
                 label=f"best score: {best_score:.3f} using threshold {best_threshold}")

        plt.legend()
        plt.title(label)
        plt.xlabel("Threshold")
        plt.ylabel("Score")
        os.makedirs("plots", exist_ok=True)
        plt.savefig(f"plots/{label}.jpg")

        if label != LABELS[-1]:
            print()
    print("#" * 52)


def main():
    print("Loading dataset...")
    x_test, y_test = load("test_pretty.csv", max_lines=159_571)
    x_train, y_train = load("train_pretty.csv", max_lines=159_571)

    print("Vectorizing input data...")
    x_train, train_removed = preprocess(x_train)
    x_test, test_removed = preprocess(x_test)
    y_train.drop(y_train.index[train_removed], inplace=True)
    y_test.drop(y_test.index[test_removed], inplace=True)
    print(f"{len(train_removed)} train comments and {len(test_removed)} test comments were removed.")

    print("Converting to tensors...")
    kwargs = {'dtype': torch.float32}
    x_train, y_train = torch.tensor(x_train, **kwargs), torch.tensor(y_train.values, **kwargs)
    x_test, y_test = torch.tensor(x_test, **kwargs), torch.tensor(y_test.values, **kwargs)

    print("Training model...")
    model = fit_model(x_train, y_train)

    plot_threshold_scores(model, x_test, y_test)


if __name__ == '__main__':
    main()
