import pickle

import nltk
from typing import List

nltk.download('punkt')
LABELS = "toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"
CURRENT_DIR = "/".join(__file__.split("/")[:-1])


def predict(data: List[str], probabilities=None) -> (dict, list, list):
    results, errors, warnings = dict(), [], []

    for label in LABELS:
        try:
            try:
                with open(f"{CURRENT_DIR}/models/{label}.pickle", "rb") as f:
                    vectorizer, model = pickle.load(f)
            except OSError as e:
                errors.append(f"could not load model for label '{label}': {e}")
            except Exception as e:
                errors.append(f"unknown exception while loading label '{label}': {e}")
            else:
                if probabilities:
                    y_pred = model.predict_proba(vectorizer.transform(data))[0][0]
                else:
                    y_pred = model.predict(vectorizer.transform(data))[0]
                results[label] = y_pred.item()
        except Exception as e:
            errors.append(f"unknown exception while processing label '{label}': {e}")
    return results, errors, warnings
