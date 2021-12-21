import json
import logging

import requests

API_URL = "http://api.hateflow.de/"


def warning_text(properties: list) -> str:
    """
    Generate a warning message for the given properties.
    """
    assert properties

    if "toxic" in properties and "severe_toxic" in properties:
        properties.remove("toxic")

    names = {
        'identity_hate': "identity hate",
        'insult': "insult",
        'obscene': "obscenity",
        'severe_toxic': "severe toxicity",
        'threat': "threat",
        'toxic': "toxicity",
    }
    properties = [names.get(i) or i for i in properties]

    detected_properties = " and ".join(properties[-2:])
    if len(properties) > 2:
        detected_properties = f"{', '.join(properties[:-2])}, {detected_properties}"

    text = f"I detected {detected_properties}.\n\n" \
           f"I am a bot developed by HateFlow. This action was performed automatically."
    return text


def check_comment(comment) -> (str, dict):
    """Check a comment using the HateFlow API."""
    response = requests.get(f"{API_URL}simpleCheck?text={comment}")
    if response.status_code != 200:
        return f"Connection to {API_URL} failed with status code {response.status_code}.", dict()
    else:
        data = json.loads(response.text)
        return "", data['results']


def process_message(text: str) -> str:
    """Runs every time a message is sent in chat."""
    print("---------new comment---------")
    print(text)

    error, result = check_comment(text)

    if error:
        logging.error(error)
    else:
        properties = []
        for key, value in result.items():
            if value == 1:
                properties.append(key)

        if properties:
            return warning_text(properties)

    return ""
