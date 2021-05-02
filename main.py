from flask import request

try:
    from . import naive_bias_predict
except ImportError:
    import naive_bias_predict


def main(*args, **kwargs) -> (dict, int):
    res = {
        'results': dict(),
        'errors': [],
        'warnings': [],
    }

    params: dict = request.args  # parameters passed in the HTTP request
    text = params.get("text")
    if text is None:
        res['errors'] = "no text submitted"
        return res, 400

    results, errors, warnings = naive_bias_predict.predict([text])
    res['results'].update(results)
    res['errors'].append(errors)
    res['warnings'].append(warnings)

    return res, 200
