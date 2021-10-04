import torch

from word2vec import process_chunk, LABELS


def predict(data: list, probabilities: bool = None, thresholds: dict = None) -> (dict, list, list):
    results, errors, warnings = dict(), previous_errors.copy(), []

    if thresholds is None:
        thresholds = dict()
    elif probabilities:
        warnings.append("The given thresholds will be ignored since probabilities were requested. "
                        "Set 'probabilities' to False for using thresholds.")

    try:
        vectorized, removed_indices = process_chunk(data)
        if vectorized:
            vectorized = torch.tensor(vectorized, dtype=torch.float32)
            y_pred = model(vectorized)
            for label_index, label in enumerate(LABELS):
                label_pred = y_pred[0, label_index]
                if not probabilities:
                    threshold = thresholds.get(label) or DEFAULT_THRESHOLDS[label]
                    label_pred = (label_pred > threshold) * 1.
                results[label] = label_pred.item()
        else:
            for label in LABELS:
                results[label] = 0.
            warnings.append("The comment did not contain any known word and could not be classified.")
    except Exception as e:
        errors.append(f"unknown exception while predicting: {e}")
    return results, errors, warnings


current_dir = "/".join(__file__.split("/")[:-1])

DEFAULT_THRESHOLDS = {
    'identity_hate': 0.018,
    'insult': 0.05,
    'obscene': 0.064,
    'severe_toxic': 0.02,
    'threat': 0.026,
    'toxic': 0.118,
}

previous_errors = []

try:
    input_dimensions, hidden = 200, 100
    model = torch.nn.Sequential(
        torch.nn.Linear(input_dimensions, hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden, 6),
        torch.nn.Sigmoid(),
    )
    model.load_state_dict(torch.load(f"{current_dir}/models/nn.pt"))
    model.eval()
except Exception as e:
    previous_errors.append(f"could not load model: {e}")
