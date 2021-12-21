import torch

from word2vec import process_chunk, LABELS

# determined during the training process, see training_log.txt
DEFAULT_THRESHOLDS = {
    'identity_hate': 0.023,
    'insult': 0.045,
    'obscene': 0.060,
    'severe_toxic': 0.043,
    'threat': 0.023,
    'toxic': 0.121,
}


def predict(params: dict) -> (dict, list, list, int):
    """
    Predict the applying labels of a comment.
    :param params: API parameters
    :return: results dictionary, error list, warnings list, HTTP response code
    """
    results, errors, warnings = dict(), previous_errors.copy(), []

    text = params.get('text')
    if text is None:
        errors.append("The request did not contain the required text parameter.")
        return results, errors, warnings, 422

    thresholds = params.get("thresholds")
    probabilities = params.get("probabilities")

    if thresholds is None:
        thresholds = dict()
    elif probabilities:
        warnings.append("The given thresholds will be ignored since probabilities were requested. "
                        "Set 'probabilities' to False for using thresholds.")

    try:
        # calculate a vector using Word2Vec
        vectorized, removed_indices = process_chunk([text])
        if vectorized:
            # process it using the Neural Network
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
    return results, errors, warnings, 500 if errors else 200


current_dir = "/".join(__file__.split("/")[:-1])
previous_errors = []

try:
    # load the Neural Network
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
