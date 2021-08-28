import logging

from flask import Flask, request
from flask_cors import CORS

try:
    try:
        from . import nn_predict
    except ImportError:
        import nn_predict
except Exception as e:
    logging.critical(f"Could not import nn_predict: {e}")


def main():
    app = Flask(__name__)
    CORS(app)

    @app.route("/<api_request>")
    def neseps_api(api_request):
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

        results, errors, warnings = nn_predict.predict(
            [text],
            probabilities=params.get("probabilities"),
            thresholds=params.get("thresholds"),
        )
        res['results'].update(results)
        res['errors'].append(errors)
        res['warnings'].append(warnings)

        return res, 200

    return app


app = main()
if __name__ == "__main__":
    app.run()
