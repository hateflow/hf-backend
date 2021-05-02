from flask import request

try:
    from .naive_bias_predict import predict
except ImportError:
    from naive_bias_predict import predict


def main(*args, **kwargs):
    params: dict = request.args  # übergebene Parameter als Dictionary
    text = params.get("text")
    if text is None:
        return "No text submitted", 400

    return predict([text])
