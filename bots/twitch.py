# twitch.py
import os # for importing env vars for the bot to use
from twitchio.ext import commands
from dotenv import load_dotenv
import time
import requests
import json
import logging


bot = commands.Bot(
    # set up the bot
    irc_token=os.environ['TMI_TOKEN'],
    client_id=os.environ['CLIENT_ID'],
    nick=os.environ['BOT_NICK'],
    prefix=os.environ['BOT_PREFIX'],
    initial_channels=[os.environ['CHANNEL']]
)

def comment_comment(ctx, properties):
    text = f"Hey @{ctx.author.name}, I detected "     #f"{os.environ['BOT_NICK']} is online!"
    propsleft = []
    for i in properties:
        propsleft.append(i)
    while propsleft != []:
        if len(propsleft) == 1 and len(properties) > 1:
            text = text + " and "
        elif len(propsleft) != len(properties):
            text = text + ", "
        if "identity_hate" in propsleft:
            text = text + "identity hate"
            propsleft.remove("identity_hate")
        elif "insult" in propsleft:
            text = text + "insult"
            propsleft.remove("insult")
        elif "obscene" in propsleft:
            text = text + "obscenity"
            propsleft.remove("obscene")
        elif "severe_toxic" in propsleft:
            text = text + "severe toxicity"
            propsleft.remove("severe_toxic")
            try:
                propsleft.remove("toxic")
            except:
                continue
        elif "threat" in propsleft:
            text = text + "threat"
            propsleft.remove("threat")
        elif "toxic" in propsleft:
            text = text + "toxicity"
            propsleft.remove("toxic")
    text = text + ".\n\nI am a bot developed by HateFlow. This action was performed automatically."
    return text

def check_comment(comment):
    API_URL = "http://api.hateflow.de/"
    response = requests.get(f"{API_URL}simpleCheck?text={comment}")
    if response.status_code != 200:
        return "Something went wrong!"
    else:
        data = json.loads(response.text)
        return data['results']

@bot.event
async def event_ready():
    'Called once when the bot goes online.'
    print(f"{os.environ['BOT_NICK']} is online!")
    ws = bot._ws  # this is only needed to send messages within event_ready
    #await ws.send_privmsg(os.environ['CHANNEL'], f"/me has landed!")

@bot.event
async def event_message(ctx):
    print("---------new-comment---------")
    print(ctx.content)
    'Runs every time a message is sent in chat.'
    # make sure the bot ignores itself and the streamer
    if ctx.author.name.lower() == os.environ['BOT_NICK'].lower():
        return
    result = check_comment(ctx.content)
    try:
        if 1 in result.values():
            properties = []
            i=0
            while i < len(list(result)):
                if list(result.values())[i] == 1:
                    properties.append(list(result.keys())[i])
                i+=1
            print(properties)
            text = comment_comment(ctx, properties)
            print(text)
            await ctx.channel.send(text)
    except Exception as e:
        logging.exception("No connection to the HateFlow-API.")

if __name__ == "__main__":
    bot.run()