import torch

from predict import model
from train import load, LABELS
from word2vec import preprocess

for dataset in ("test", "train"):
    x_data, y_data = load(f"{dataset}_pretty.csv", max_lines=159_571)
    print(f"{dataset}: {len(x_data)} Kommentare")
    x_data, x_removed = preprocess(x_data)
    n_unknown = len(x_removed)
    print(f"nicht klassifizierbar: {n_unknown} Kommentare / {n_unknown / len(y_data):.3f} des Datensatzes")
    y_data.drop(y_data.index[x_removed], inplace=True)

    x_data, y_data = torch.tensor(x_data, dtype=torch.float32), torch.tensor(y_data.values, dtype=torch.float32)
    y_pred = torch.round(model(x_data)).clone().detach()

    n_all_correct = torch.sum(torch.all(y_pred == y_data, axis=1)).item()
    print(f"alle Labels richtig: {n_all_correct} Kommentare / {n_all_correct / len(y_data):.3f} der Tests")

    n_any_correct = len(torch.where(y_pred.any(axis=1) == y_data.any(axis=1))[0])
    print(f"mindestens ein Label bei unangemessenem oder kein Label bei sachlichem Kommentar: "
          f"{n_any_correct} Kommentare / {n_any_correct / len(y_data):.3f} der Tests")

    for label_index, label in enumerate(LABELS):
        n_severe_toxic_correct = torch.sum(y_pred[:, label_index] == y_data[:, label_index]).item()
        print(f"Label '{label}' korrekt: {n_severe_toxic_correct} Kommentare / {n_severe_toxic_correct / len(y_data):.3f} der Tests")
    print()
