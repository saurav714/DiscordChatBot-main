Welcome to the repository for the DiscordChatBot project, a chatbot built using Python and powered by OpenAI's API to facilitate natural conversations on Discord. The bot leverages GPT-4.0/Deepseek R1 to create contextually fitting and human-like responses, making interactions with users more engaging and dynamic.

Table of Contents

Overview

Key Features

Demo

Technology Used

Prerequisites & API Keys

Installation

Running the Bot

Inviting the Bot to Your Server

Usage

Customization

Further Exploration

Disclaimer

Contributing

License

Hosting: Keeping the Bot Alive 24/7 (Replit + Keep-Alive)

Overview

The DiscordChatBot project demonstrates a Python-based Discord bot that interacts within a server, utilizing the OpenAI GPT-4.0 model for generating human-like responses. This bot is designed to maintain conversational context, ensuring dynamic and relevant interactions. By combining AI-driven responses with Discord's interactive platform, it opens up a world of engaging and lifelike conversations.

Key Features

Discord Bot Integration: The bot interacts within Discord servers, responding to mentions, facilitating conversations, and generating replies based on user input.

OpenAI GPT-4.0 Integration: Leverages the powerful text-davinci-003 model to generate coherent and contextually relevant responses, making interactions more natural.

Chat History Management: The bot stores the last 10 messages in a conversation to maintain context and deliver better replies. This helps to generate more relevant and consistent responses in ongoing conversations.

Message Similarity Handling: Uses Python’s difflib library to prevent repetition by comparing consecutive messages and ensuring a diverse range of responses.

Dynamic Prompt Generation: Customizes prompts by including chat history and user input to maintain the flow of conversation. This ensures the bot generates responses that are coherent and contextually aware.

Token Management: Limits responses to within OpenAI's token restriction (max_tokens=256) to ensure concise and meaningful interactions.

Asynchronous Processing: Built using Discord.py, the bot can handle multiple messages asynchronously, ensuring a smooth experience even in larger servers.

Demo

Here’s a quick look at the DiscordChatBot in action:

Technology Used

Python: Core language used to implement the bot and manage its functionalities.

Discord.py: A Python library that interfaces with the Discord API, allowing for seamless bot integration.

OpenAI API: Provides access to OpenAI’s GPT-3.5 model for generating responses.

difflib: A standard Python library that handles message similarity, ensuring the bot provides varied responses.

.env Files: Used to store sensitive bot credentials securely.

Prerequisites & API Keys

Before you begin, you'll need:

A Discord bot token from the Discord Developer Portal.

An OpenAI API key from the OpenAI website.

Installation

To get started with the DiscordChatbot project, follow these steps:

Clone the repository to your local machine:

git clone https://github.com/saurav714/DiscordChatBot-main.git

Navigate to the project directory:

cd DiscordChatBot

Create a new file in the root directory of the project and name it .env.

Inside the .env file, add your Discord bot token and OpenAI API key in the format given in .env.example

DISCORD_PASS_KEY=YOUR_DISCORD_BOT_TOKEN
OPENAI_API_KEY=YOUR_OPENAI_API_KEY

Save and close the .env file.

Run the bot using the provided code, and it will use the keys you've provided in the .env file to connect to Discord and use the OpenAI API.

Running the Bot

Install dependencies:

pip install discord.py openai python-dotenv

Then run your bot:

python main.py

Inviting the Bot to Your Server

Go to the Discord Developer Portal, select your bot.

Navigate to OAuth2 > URL Generator.

Select the bot scope and assign permissions like Read Messages, Send Messages, etc.

Copy the URL, open in your browser, and invite the bot to your server.

Usage

Mention the Bot: To initiate a conversation, mention the bot in a message.

Auto-Response: The bot will reply with a context-aware, AI-generated response.

Customization

OpenAI Model Parameters: Adjust temperature, max_tokens, etc., for different behaviors.

Message Similarity Threshold: Default 0.7 using difflib, adjust for creativity.

Chat History: Modify MAX_CHAT_HISTORY to change context depth.

Further Exploration

Enhance prompt formatting.

Add command support or features.

Multi-language or sentiment-aware responses.

Disclaimer

This bot is an example for educational purposes. OpenAI's model responses may vary and may not always reflect accurate or appropriate content.


Pull requests welcome! Fork, create a new branch, and submit your improvements.

Hosting: Keeping the Bot Alive 24/7 (Replit + Keep-Alive)

To run your bot 24/7, you can deploy it on Replit, a free browser-based Python IDE.

1. Add keep_alive.py

Create keep_alive.py:

from flask import Flask
from threading import Thread

app = Flask('')

@app.route('/')
def home():
    return "Bot is alive!"

def run():
    app.run(host='0.0.0.0', port=8080)

def keep_alive():
    t = Thread(target=run)
    t.start()

2. Modify main.py

Import and call keep_alive at the top:

from keep_alive import keep_alive
keep_alive()
client.run(my_secret)

3. Deploy to Replit

Create a new Python Repl.

Upload project files.

Add secrets from .env using the Replit Secrets tab.

Run your bot.

4. Ping with UptimeRobot

Go to UptimeRobot

Create HTTP(s) monitor.

Use your Replit URL (https://your-repl-name.username.repl.co/)

Set interval to 5 minutes.

License

MIT License.





