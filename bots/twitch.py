import os

from dotenv import load_dotenv
from twitchio.ext import commands

from bots_base import process_message

load_dotenv()
bot = commands.Bot(
    # set up the bot
    token=os.getenv('T_TMI_TOKEN'),
    client_id=os.getenv('T_CLIENT_ID'),
    nick=os.getenv('T_BOT_NICK'),
    prefix=os.getenv('T_BOT_PREFIX'),
    initial_channels=[os.getenv('T_CHANNEL')]
)


@bot.event
async def event_ready():
    print(f"{os.environ['BOT_NICK']} is online!")


@bot.event
async def event_message(context):
    # make sure the bot ignores itself and the streamer
    if context.author.name.lower() == os.getenv('BOT_NICK').lower():
        return

    warning = process_message(context.content)
    if warning:
        response = f"Hey @{context.author.name}, {warning}"
        print(response)
        # await ctx.channel.send(response)


if __name__ == "__main__":
    bot.run()
