from flask import request


def main(*args, **kwargs):
    params: dict = request.args  # übergebene Parameter als Dictionary
    text = params.get("text")
    if text is None:
        return "No text submitted", 400

    if "!" in text:
        return dict(total_scores={
            'toxic': 1.0,
            'severe_toxic': 1.0,
            'obscene': 1.0,
            'threat': 1.0,
            'insult': 1.0,
            'identity_hate': 1.0,
        })

    return dict(total_scores={
        'toxic': 0.0,
        'severe_toxic': 0.0,
        'obscene': 0.0,
        'threat': 0.0,
        'insult': 0.0,
        'identity_hate': 0.0,
    })
