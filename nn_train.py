import os

import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score

from word2vec import preprocess

DATASET_LINES = 159_571  # total lines: 159_571
LABELS = "toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"
torch.set_num_threads(os.cpu_count())


def load(path, max_lines=10000):
    df = pd.read_csv(path)[-max_lines:]
    x = df['comment_text']
    y = df[[i for i in LABELS]]
    return x, y


def evaluate_label(x_test, y_test, label_index, label, model):
    print(label)

    if label_index is None:
        x_test_true = x_test[torch.any(y_test, axis=-1)]
        y_test_true = y_test[torch.any(y_test, axis=-1)]
        x_test_false = x_test[torch.any(y_test, axis=-1)]
        y_test_false = y_test[torch.any(y_test, axis=-1)]
    else:
        x_test_true = x_test[y_test[:, label_index] == 1]
        y_test_true = y_test[y_test[:, label_index] == 1]
        x_test_false = x_test[y_test[:, label_index] == 0]
        y_test_false = y_test[y_test[:, label_index] == 0]

    evaluate_label_threshold(
        model,
        x_test_true,
        y_test_true,
        x_test_false,
        y_test_false,
        label_index,
    )


def train_model(x_train, y_train, x_test, y_test):
    input_dimensions, hidden = 200, 100
    model = torch.nn.Sequential(
        torch.nn.Linear(input_dimensions, hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden, 6),
        torch.nn.Sigmoid(),
    )

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    train_losses, test_losses = [], []
    for i in range(3000):
        if i and not i % 500:
            print("#" * 20, "EVALUATION", "#" * 20)
            for label_index, label in enumerate(LABELS):
                print(label)

                evaluate_label(x_test, y_test, label_index, label, model)
                print()
            evaluate_label(x_test, y_test, slice(None), "any", model)
            print("#" * 52)
        y_pred = model(x_train)
        loss = loss_fn(y_pred, y_train)
        train_losses.append(loss.item())
        test_loss = loss_fn(model(x_test), y_test).item()
        test_losses.append(test_loss)
        print(f"{i:3}   train: {loss.item():.6f}   test: {test_loss:.6f}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), "models/nn.pt")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.plot(train_losses, label="train")
    plt.plot(test_losses, label="test")
    plt.legend()
    plt.show()

    return model


def evaluate_label_threshold(model, x_test_true, y_test_true, x_test_false, y_test_false, label_index, plot=False):
    true_scores = dict()
    false_scores = dict()

    for i in range(1001):
        threshold = i / 1000

        y_true_pred = (model(x_test_true)[:, label_index] > threshold) * 1
        true_scores[threshold] = accuracy_score(y_test_true[:, label_index], y_true_pred)

        y_false_pred = (model(x_test_false)[:, label_index] > threshold) * 1
        false_scores[threshold] = accuracy_score(y_test_false[:, label_index], y_false_pred)

    best_threshold = max(true_scores.keys(), key=lambda k: true_scores[k] + false_scores[k])
    print(f"    best threshold: {best_threshold}")

    best_score = (true_scores[best_threshold] + false_scores[best_threshold]) / 2
    print(f"    true score: {true_scores[best_threshold]:.3f}    false score: {false_scores[best_threshold]:.3f}")
    print(f"    --> unweighted average: {best_score:.4f}")

    if plot:
        label = LABELS[label_index]
        plt.plot([best_threshold, best_threshold], [0, 1],
                 label=f"best score: {best_score:.3f} using threshold {best_threshold}")

        plt.cla()
        plt.plot(true_scores.keys(), true_scores.values(), label=label)
        plt.plot(false_scores.keys(), false_scores.values(), label=f"not {label}")

        plt.legend()
        plt.title(label)
        plt.xlabel("Threshold")
        plt.ylabel("Score")
        os.makedirs("plots", exist_ok=True)
        plt.savefig(f"plots/{label}.jpg")


def evaluate_model(model: torch.nn.Sequential, x_test: torch.tensor, y_test: torch.tensor):
    print("#" * 20, "EVALUATION", "#" * 20)
    for label_index, label in enumerate(LABELS):
        print(label)

        x_test_true = x_test[y_test[:, label_index] == 1]
        y_test_true = y_test[y_test[:, label_index] == 1]
        x_test_false = x_test[y_test[:, label_index] == 0]
        y_test_false = y_test[y_test[:, label_index] == 0]

        evaluate_label_threshold(
            model,
            x_test_true,
            y_test_true,
            x_test_false,
            y_test_false,
            label_index,
            plot=True,
        )

        if label != LABELS[-1]:
            print()
    print("#" * 52)


def main():
    print("Loading dataset...")

    x_test, y_test = load("test_pretty.csv", max_lines=DATASET_LINES)
    x_train, y_train = load("train_pretty.csv", max_lines=DATASET_LINES)

    print("Vectorizing input data...")
    x_train, train_removed = preprocess(x_train)
    y_train.drop(y_train.index[train_removed], inplace=True)
    x_test, test_removed = preprocess(x_test)
    y_test.drop(y_test.index[test_removed], inplace=True)
    print(f"{len(train_removed)} train comments and {len(test_removed)} test comments were removed.")

    print("Converting to tensors...")
    kwargs = {'dtype': torch.float32}
    x_train, y_train = torch.tensor(x_train, **kwargs), torch.tensor(y_train.values, **kwargs)
    x_test, y_test = torch.tensor(x_test, **kwargs), torch.tensor(y_test.values, **kwargs)

    print("Training model...")
    model = train_model(x_train, y_train, x_test, y_test)

    evaluate_model(model, x_test, y_test)


if __name__ == '__main__':
    main()
