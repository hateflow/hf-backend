from flask import request

try:
    from . import naive_bias_predict
except ImportError:
    import naive_bias_predict


def main(*args, **kwargs):
    params: dict = request.args  # übergebene Parameter als Dictionary
    text = params.get("text")
    if text is None:
        return "No text submitted", 400

    return naive_bias_predict.predict([text])
