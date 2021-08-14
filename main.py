import logging
import os
from threading import Timer

from flask import Flask, request
from flask_cors import CORS
from git import Repo

try:
    from . import naive_bias_predict
except ImportError:
    try:
        import naive_bias_predict
    except ImportError as e:
        logging.critical(f"Could not import naive_bias_predict: {e}")


def main():
    app = Flask(__name__)
    CORS(app)

    @app.route("/ping")
    def ping():
        return "pong"

    @app.route("/git/neseps/webhook", methods=["POST"])
    def update_git():
        Repo("/root/neseps").remotes.origin.pull()
        Timer(0.01, lambda: os.system("service apache2 restart")).start()
        return "git pull successful, restarting Apache..", 200

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

        results, errors, warnings = naive_bias_predict.predict(
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
