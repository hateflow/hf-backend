import pickle

import nltk
from typing import List

nltk.download('punkt')
LABELS = "toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"
CURRENT_DIR = "/".join(__file__.split("/")[:-1])
DEFAULT_THRESHOLDS = {
    'identity_hate': 0.011,
    'insult': 0.062,
    'obscene': 0.099,
    'severe_toxic': 0.011,
    'threat': 0.008,
    'toxic': 0.104,
}

previous_errors = []
models = dict()
for label in LABELS:
    try:
        with open(f"{CURRENT_DIR}/models/{label}.pickle", "rb") as f:
            models[label] = pickle.load(f)
    except OSError as e:
        previous_errors.append(f"could not load model for label '{label}': {e}")
    except Exception as e:
        previous_errors.append(f"unknown exception while loading label '{label}': {e}")

try:
    with open(f"{CURRENT_DIR}/models/vectorizer.pickle", "rb") as f:
        vectorizer = pickle.load(f)
except Exception as e:
    previous_errors.append(f"unknown exception while loading the vectorizer: {e}")


def predict(data: List[str], probabilities: bool = None, thresholds: dict = None) -> (dict, list, list):
    results, errors, warnings = dict(), previous_errors.copy(), []

    if thresholds is None:
        thresholds = dict()
    elif probabilities:
        warnings.append("The given thresholds will be ignored since probabilities were requested. "
                        "Set 'probabilities' to False for using thresholds.")

    for label in LABELS:
        try:
            model = models[label]
            y_pred = model.predict_proba(vectorizer.transform(data))[:, 1]
            if not probabilities:
                threshold = thresholds.get(label) or DEFAULT_THRESHOLDS[label]
                y_pred = y_pred > threshold
            results[label] = y_pred.item()
        except Exception as e:
            errors.append(f"unknown exception while processing label '{label}': {e}")
    return results, errors, warnings
