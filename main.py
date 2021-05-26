import requests
from flask import Flask, request
from flask_cors import CORS
from git import Repo

try:
    from . import naive_bias_predict
except ImportError:
    import naive_bias_predict


def main():
    app = Flask(__name__)
    CORS(app)

    @app.route("/git/neseps/webhook", methods=["POST"])
    def update_git():
        return
        Repo("/home/JSchoedl/site/neseps").remotes.origin.pull()
        Repo("/home/JSchoedl/site/neseps-docs").remotes.origin.pull()
        # Repo.clone_from("https://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@github.com/jschoedl/neseps.git", "neseps")

        username = 'JSchoedl'
        token = '06b26af687480abff2701135ce4906fde5cb3af9'
        domain_name = "JSchoedl.eu.pythonanywhere.com"

        res = requests.post(
            f"https://eu.pythonanywhere.com/api/v0/user/{username}/webapps/{domain_name}/reload/",
            headers={'Authorization': f'Token {token}'}
        )
        return res

    @app.route("/api/<api_request>")
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

        results, errors, warnings = naive_bias_predict.predict([text], probabilities=params.get("probabilities"))
        res['results'].update(results)
        res['errors'].append(errors)
        res['warnings'].append(warnings)

        return res, 200

    return app

app = main()
if __name__ == "__main__":
    app.run()
