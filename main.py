import discord
import os
import requests
import json
import difflib
from dotenv import load_dotenv

load_dotenv()

# Configuration - easily switch between different API providers
API_CONFIG = {
    # OpenAI (default)
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-3.5-turbo",
        "api_key_env": "OPENAI_API_KEY"
    },
    # DeepSeek
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1", 
        "model": "deepseek-chat",
        "api_key_env": "DEEPSEEK_API_KEY"
    },
    # Together AI
    "together": {
        "base_url": "https://api.together.xyz/v1",
        "model": "meta-llama/Llama-2-7b-chat-hf",
        "api_key_env": "TOGETHER_API_KEY"
    },
    # Groq
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "model": "mixtral-8x7b-32768",
        "api_key_env": "GROQ_API_KEY"
    },
    # Local/Custom API
    "local": {
        "base_url": "http://localhost:1234/v1",  # Ollama, LM Studio, etc.
        "model": "llama2",
        "api_key_env": "LOCAL_API_KEY"  # May not be needed for local
    }
}

# Choose your provider here
CURRENT_PROVIDER = os.getenv("AI_PROVIDER", "openai")  # Default to OpenAI
config = API_CONFIG.get(CURRENT_PROVIDER, API_CONFIG["openai"])

my_secret = os.getenv("DISCORD_PASS_KEY")
api_key = os.getenv(config["api_key_env"])

chat_history = []
prev_response = ""

def call_ai_api(messages, temperature=0.7, max_tokens=256):
    """Generic function to call any OpenAI-compatible API"""
    headers = {
        "Content-Type": "application/json",
    }
    
    # Add authorization if API key exists
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    payload = {
        "model": config["model"],
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": 1,
        "frequency_penalty": 0.2,
        "presence_penalty": 0.5,
    }
    
    try:
        response = requests.post(
            f"{config['base_url']}/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
        return None

class MyClient(discord.Client):
    async def on_ready(self):
        print(f"Logged on as {self.user}!")
        print(f"Using AI provider: {CURRENT_PROVIDER}")
        print(f"Model: {config['model']}")

    async def on_message(self, message):
        global chat_history
        global prev_response

        if self.user != message.author:
            # Check if the bot is mentioned in the message
            if self.user in message.mentions:
                author = message.author.name
                content = message.content

                # Remove the bot mention from the content
                content = content.replace(f'<@{self.user.id}>', '').strip()

                # Update chat history with the latest message
                chat_history.append(f"Message from {author}: {content}")

                MAX_CHAT_HISTORY = 10
                if len(chat_history) > MAX_CHAT_HISTORY:
                    chat_history = chat_history[-MAX_CHAT_HISTORY:]

                # Create a context by joining the chat history messages
                context = "\n".join(chat_history)

                # Generate a response using the configured AI API
                response = None
                attempts = 0
                max_attempts = 3
                
                while (
                    response is None
                    or (prev_response and response.get("choices") and 
                        difflib.SequenceMatcher(
                            None, response["choices"][0]["message"]["content"], prev_response
                        ).ratio() > 0.7)
                ) and attempts < max_attempts:
                    
                    # Build messages for chat completion
                    messages = [
                        {"role": "system", "content": "You are ChatterBox, a helpful and friendly Discord bot. Keep responses conversational and engaging. Respond naturally without being overly formal."},
                        {"role": "user", "content": f"Context:\n{context}\n\nPlease respond to the latest message naturally."}
                    ]
                    
                    response = call_ai_api(messages)
                    attempts += 1
                    
                    if not response:
                        await message.reply("Sorry, I encountered an error while generating a response.", mention_author=False)
                        return

                if response and response.get("choices"):
                    message_to_send = response["choices"][0]["message"]["content"].strip()
                    prev_response = message_to_send
                    
                    # Limit message length for Discord
                    if len(message_to_send) > 2000:
                        message_to_send = message_to_send[:1997] + "..."
                    
                    await message.reply(message_to_send, mention_author=False)
                else:
                    await message.reply("I'm having trouble generating a unique response right now. Please try again!", mention_author=False)

intents = discord.Intents.default()
intents.message_content = True

client = MyClient(intents=intents)

try:
    if not my_secret:
        raise ValueError("Discord bot token not found. Please set DISCORD_PASS_KEY in your .env file.")
    
    print(f"Starting bot with {CURRENT_PROVIDER} API...")
    client.run(my_secret)
except Exception as e:
    print(f"Error running bot: {e}")
    print("\nMake sure you have set the following in your .env file:")
    print("- DISCORD_PASS_KEY=your_discord_bot_token")
    print(f"- {config['api_key_env']}=your_api_key")
    print("- AI_PROVIDER=openai|deepseek|together|groq|local (optional, defaults to openai)")