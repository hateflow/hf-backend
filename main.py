import logging

from flask import Flask, request, redirect
from flask_cors import CORS

try:
    try:
        from . import predict
    except ImportError:
        import predict
except Exception as e:
    logging.critical(f"Could not import nn_predict: {e}")


def main():
    """
    The main API function.

    Receives all calls to http://api.hateflow.de.
    :return:
    """
    app = Flask(__name__)  # initialize a new Flask app
    CORS(app)  # make it accessible from all domains

    @app.route("/")
    def hf_home():
        """
        Redirect to hateflow.de.

        Calling the root page does not correspond to any API action.
        """
        return redirect("https://hateflow.de", 307)

    @app.route("/<requested_action>")
    def hateflow_api(requested_action):
        """
        Map API actions with Python functions.
        """
        res = {
            'results': dict(),
            'errors': [],
            'warnings': [],
        }

        api_actions = {
            'simpleCheck': predict.predict,
        }

        try:
            results, errors, warnings, status = api_actions[requested_action](request.args)
            res['results'].update(results)
            res['errors'].append(errors)
            res['warnings'].append(warnings)
        except KeyError:
            res['errors'].append(f"unknown API action: '{requested_action}'")
            status = 404
        except Exception as e:
            res['errors'].append(f"exception while calling API action '{requested_action}': {e}")
            status = 500
        return res, status

    return app


# run the Flask app
app = main()
if __name__ == "__main__":
    app.run()
