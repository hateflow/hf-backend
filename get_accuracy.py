import torch

from predict import model, DEFAULT_THRESHOLDS
from train import load, LABELS
from word2vec import preprocess


def metrics(true_positives: int, false_positives: int, false_negatives: int) -> float:
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1 = 2 * (precision * recall) / (precision + recall)
    return precision, recall, f1


def get_n_correct(x_data, y_data, label_index, label_value, threshold):
    relevant_indices = torch.where(y_data[:, label_index] == float(label_value))
    x_data, y_data = x_data[relevant_indices], y_data[relevant_indices]
    y_pred = (model(x_data)).clone().detach() > 0.5
    n_correct = torch.sum(y_pred[:, label_index] == bool(label_value)).item()
    return n_correct, len(x_data) - n_correct

    # y_pred = torch.round(model(x_data)).clone().detach()

    # all labels
    # n_all_correct = torch.sum(torch.all(y_pred == y_data, axis=1)).item()
    # print(f"alle Labels richtig: {n_all_correct} Kommentare / {n_all_correct / len(y_data):.3f} der Tests")

    # n_any_correct = len(torch.where(y_pred.any(axis=1) == y_data.any(axis=1))[0])
    # print(f"mindestens ein Label bei unangemessenem oder kein Label bei sachlichem Kommentar: "
    #       f"{n_any_correct} Kommentare / {n_any_correct / len(y_data):.3f} der Tests")

    # severe_toxic_index = LABELS.index("severe_toxic")
    # are_severe_toxic = y_pred[torch.where(y_pred[:, severe_toxic_index] == 1.)]
    # are_not_severe_toxic = y_pred[torch.where(y_pred[:, severe_toxic_index] == 0.)]
    # true_positives = torch.sum(
    #     are_severe_toxic[:, severe_toxic_index] == are_severe_toxic[:, severe_toxic_index]).item()
    # n_severe_toxic_correct = torch.sum(y_pred[:, severe_toxic_index] == y_data[:, severe_toxic_index]).item()
    # print(f"Label 'severe_toxic' korrekt: {n_severe_toxic_correct} Kommentare / "
    #       f"{n_severe_toxic_correct / len(y_data):.3f} der Tests")
    # print()


def print_evaluation(dataset):
    x_data, y_data = load(f"{dataset}_pretty.csv", max_lines=159_571)
    print(f"{dataset}: {len(x_data)} Kommentare")
    x_data, x_removed = preprocess(x_data)
    n_unknown = len(x_removed)
    print(f"nicht klassifizierbar: {n_unknown} Kommentare / {n_unknown / len(y_data):.3f} des Datensatzes")
    y_data.drop(y_data.index[x_removed], inplace=True)

    x_data, y_data = torch.tensor(x_data, dtype=torch.float32), torch.tensor(y_data.values, dtype=torch.float32)
    for label_index, label in enumerate(LABELS):
        n_true_positive, n_false_negative = get_n_correct(
            x_data,
            y_data,
            label_index,
            label_value=1,
            threshold=DEFAULT_THRESHOLDS[label],
        )
        n_true_negative, n_false_positive = get_n_correct(
            x_data,
            y_data,
            label_index,
            label_value=0,
            threshold=DEFAULT_THRESHOLDS[label],
        )

        precision, recall, f1 = metrics(n_true_positive, n_false_positive, n_false_negative)
        print(f"{label}:{' '*(14-len(label))}"
              f"F1 = {f1:.3f}  Precision = {precision:.3f}  Recall = {recall:.3f}")


if __name__ == '__main__':
    for dataset in ("test", "train"):
        print_evaluation(dataset)
