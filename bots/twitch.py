import json
import logging
import os

import requests
from dotenv import load_dotenv
from twitchio.ext import commands

API_URL = "http://api.hateflow.de/"

load_dotenv()
bot = commands.Bot(
    # set up the bot
    token=os.getenv('TMI_TOKEN'),
    client_id=os.getenv('CLIENT_ID'),
    nick=os.getenv('BOT_NICK'),
    prefix=os.getenv('BOT_PREFIX'),
    initial_channels=[os.getenv('CHANNEL')]
)


def warning_text(context, properties: list) -> str:
    assert properties

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

    text = f"Hey @{context.author.name}, I detected {detected_properties}.\n\n" \
           f"I am a bot developed by HateFlow. This action was performed automatically."
    return text


def check_comment(comment) -> (str, dict):
    response = requests.get(f"{API_URL}simpleCheck?text={comment}")
    if response.status_code != 200:
        return f"Connection to {API_URL} failed with status code {response.status_code}.", dict()
    else:
        data = json.loads(response.text)
        return "", data['results']


@bot.event
async def event_ready():
    print(f"{os.environ['BOT_NICK']} is online!")


@bot.event
async def event_message(context):
    """Runs every time a message is sent in chat."""
    print("---------new comment---------")
    print(context.content)

    # make sure the bot ignores itself and the streamer
    if context.author.name.lower() == os.getenv('BOT_NICK').lower():
        return
    error, result = check_comment(context.content)

    if error:
        logging.error(error)
    else:
        properties = []
        for key, value in result.items():
            if value == 1:
                properties.append(key)

        if properties:
            text = warning_text(context, properties)
            print(text)
            # await ctx.channel.send(text)


if __name__ == "__main__":
    bot.run()
