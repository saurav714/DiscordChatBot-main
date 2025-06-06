import discord
import os
import aiohttp
import json
import difflib
from dotenv import load_dotenv
from collections import defaultdict
import logging
import asyncio
import time
from datetime import datetime, timedelta
import random

# Configure logging with more detailed formatting
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("bot.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configuration for different API providers
API_CONFIG = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-3.5-turbo",
        "api_key_env": "OPENAI_API_KEY",
        "temperature": float(os.getenv("AI_TEMPERATURE", 0.7)),
        "max_tokens": int(os.getenv("AI_MAX_TOKENS", 512)),
        "retries": int(os.getenv("AI_RETRIES", 3))
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
        "api_key_env": "DEEPSEEK_API_KEY",
        "temperature": float(os.getenv("AI_TEMPERATURE", 0.7)),
        "max_tokens": int(os.getenv("AI_MAX_TOKENS", 512)),
        "retries": int(os.getenv("AI_RETRIES", 3))
    },
    "together": {
        "base_url": "https://api.together.xyz/v1",
        "model": "meta-llama/Llama-2-7b-chat-hf",
        "api_key_env": "TOGETHER_API_KEY",
        "temperature": float(os.getenv("AI_TEMPERATURE", 0.7)),
        "max_tokens": int(os.getenv("AI_MAX_TOKENS", 512)),
        "retries": int(os.getenv("AI_RETRIES", 3))
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "model": "mixtral-8x7b-32768",
        "api_key_env": "GROQ_API_KEY",
        "temperature": float(os.getenv("AI_TEMPERATURE", 0.7)),
        "max_tokens": int(os.getenv("AI_MAX_TOKENS", 512)),
        "retries": int(os.getenv("AI_RETRIES", 3))
    },
    "local": {
        "base_url": "http://localhost:1234/v1",
        "model": "llama2",
        "api_key_env": "LOCAL_API_KEY",
        "temperature": float(os.getenv("AI_TEMPERATURE", 0.7)),
        "max_tokens": int(os.getenv("AI_MAX_TOKENS", 512)),
        "retries": int(os.getenv("AI_RETRIES", 3))
    }
}

# Choose provider and validate configuration
CURRENT_PROVIDER = os.getenv("AI_PROVIDER", "openai")
config = API_CONFIG.get(CURRENT_PROVIDER, API_CONFIG["openai"])

# Enhanced configuration
MAX_CHAT_HISTORY = int(os.getenv("MAX_CHAT_HISTORY", 15))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.6))
MEMORY_CLEANUP_INTERVAL = int(os.getenv("MEMORY_CLEANUP_INTERVAL", 3600))  # 1 hour
INACTIVE_CHANNEL_TIMEOUT = int(os.getenv("INACTIVE_CHANNEL_TIMEOUT", 86400))  # 24 hours

def validate_config():
    """Validate configuration and API keys"""
    if CURRENT_PROVIDER not in API_CONFIG:
        raise ValueError(f"Invalid AI_PROVIDER '{CURRENT_PROVIDER}'. Choose from: {list(API_CONFIG.keys())}")
    api_key = os.getenv(config["api_key_env"])
    if not api_key and config["api_key_env"] != "LOCAL_API_KEY":
        raise ValueError(f"Missing API key. Set {config['api_key_env']} in your .env file.")
    logger.info(f"Configuration validated for provider: {CURRENT_PROVIDER}")

try:
    validate_config()
except ValueError as e:
    logger.error(f"Configuration error: {e}")
    exit(1)

my_secret = os.getenv("DISCORD_PASS_KEY")
api_key = os.getenv(config["api_key_env"])

class MemoryManager:
    """Manages chat history and memory cleanup"""
    
    def __init__(self):
        self.chat_history = defaultdict(list)
        self.prev_responses = defaultdict(str)
        self.channel_last_activity = defaultdict(lambda: datetime.now())
        self.user_preferences = defaultdict(dict)
    
    def add_message(self, channel_id, author, content):
        """Add a message to chat history"""
        timestamp = datetime.now().strftime("%H:%M")
        formatted_message = f"[{timestamp}] {author}: {content}"
        self.chat_history[channel_id].append(formatted_message)
        self.channel_last_activity[channel_id] = datetime.now()
        
        # Maintain history limit
        if len(self.chat_history[channel_id]) > MAX_CHAT_HISTORY:
            self.chat_history[channel_id] = self.chat_history[channel_id][-MAX_CHAT_HISTORY:]
    
    def get_context(self, channel_id):
        """Get formatted context for AI"""
        history = self.chat_history[channel_id]
        if not history:
            return "No previous conversation context."
        return "\n".join(history[-10:])  # Last 10 messages for context
    
    def cleanup_inactive_channels(self):
        """Remove data for inactive channels"""
        current_time = datetime.now()
        inactive_channels = []
        
        for channel_id, last_activity in self.channel_last_activity.items():
            if current_time - last_activity > timedelta(seconds=INACTIVE_CHANNEL_TIMEOUT):
                inactive_channels.append(channel_id)
        
        for channel_id in inactive_channels:
            self.chat_history.pop(channel_id, None)
            self.prev_responses.pop(channel_id, None)
            self.channel_last_activity.pop(channel_id, None)
            self.user_preferences.pop(channel_id, None)
            logger.info(f"Cleaned up inactive channel: {channel_id}")
    
    def get_stats(self):
        """Get memory usage statistics"""
        return {
            "active_channels": len(self.chat_history),
            "total_messages": sum(len(history) for history in self.chat_history.values()),
            "channels_with_responses": len(self.prev_responses)
        }

class RateLimiter:
    """Discord rate limiting handler"""
    
    def __init__(self):
        self.message_timestamps = defaultdict(list)
        self.typing_timestamps = defaultdict(list)
    
    async def can_send_message(self, channel_id):
        """Check if we can send a message (5 per 5 seconds per channel)"""
        now = time.time()
        self.message_timestamps[channel_id] = [
            ts for ts in self.message_timestamps[channel_id] 
            if now - ts < 5
        ]
        
        if len(self.message_timestamps[channel_id]) >= 5:
            return False
        
        self.message_timestamps[channel_id].append(now)
        return True
    
    async def can_start_typing(self, channel_id):
        """Check if we can start typing (1 per 5 seconds per channel)"""
        now = time.time()
        self.typing_timestamps[channel_id] = [
            ts for ts in self.typing_timestamps[channel_id]
            if now - ts < 5
        ]
        
        if self.typing_timestamps[channel_id]:
            return False
        
        self.typing_timestamps[channel_id].append(now)
        return True

# Initialize managers
memory_manager = MemoryManager()
rate_limiter = RateLimiter()

async def call_ai_api(messages, temperature=None, max_tokens=None, retries=None):
    """Enhanced AI API call with better error handling"""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    payload = {
        "model": config["model"],
        "messages": messages,
        "temperature": temperature if temperature is not None else config["temperature"],
        "max_tokens": max_tokens if max_tokens is not None else config["max_tokens"],
        "top_p": 0.9,
        "frequency_penalty": 0.3,
        "presence_penalty": 0.6,
    }
    
    max_retries = retries if retries is not None else config["retries"]
    
    for attempt in range(max_retries):
        try:
            timeout = aiohttp.ClientTimeout(total=45)  # Increased timeout
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    f"{config['base_url']}/chat/completions",
                    headers=headers,
                    json=payload
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    elif response.status == 429:
                        retry_after = int(response.headers.get('Retry-After', 2 ** attempt))
                        logger.warning(f"Rate limited, waiting {retry_after}s (attempt {attempt + 1}/{max_retries})")
                        await asyncio.sleep(retry_after)
                    elif response.status == 401:
                        logger.error("Authentication error: Invalid API key")
                        return None
                    elif response.status >= 500:
                        logger.warning(f"Server error {response.status}, retrying (attempt {attempt + 1}/{max_retries})")
                        await asyncio.sleep(2 ** attempt)
                    else:
                        logger.error(f"API request failed with status {response.status}")
                        return None
                        
        except asyncio.TimeoutError:
            logger.warning(f"Request timeout (attempt {attempt + 1}/{max_retries})")
            await asyncio.sleep(2 ** attempt)
        except Exception as e:
            logger.error(f"Unexpected error: {e} (attempt {attempt + 1}/{max_retries})")
            await asyncio.sleep(2 ** attempt)
    
    logger.error("Max retries exceeded for API call")
    return None

class EnhancedChatterBox(discord.Client):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cleanup_task = None
        self.start_time = datetime.now()
    
    async def on_ready(self):
        logger.info(f"🤖 {self.user} is online and ready!")
        logger.info(f"📡 Using AI provider: {CURRENT_PROVIDER}")
        logger.info(f"🧠 Model: {config['model']}")
        logger.info(f"🔧 Max history: {MAX_CHAT_HISTORY} messages")
        logger.info(f"🎯 Similarity threshold: {SIMILARITY_THRESHOLD}")
        
        # Start cleanup task
        self.cleanup_task = self.loop.create_task(self.periodic_cleanup())
    
    async def periodic_cleanup(self):
        """Periodic memory cleanup task"""
        while not self.is_closed():
            try:
                await asyncio.sleep(MEMORY_CLEANUP_INTERVAL)
                memory_manager.cleanup_inactive_channels()
                stats = memory_manager.get_stats()
                logger.info(f"Memory stats: {stats}")
            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")
    
    async def on_message(self, message):
        if self.user == message.author:
            return
        
        content = message.content.strip()
        channel_id = message.channel.id
        author = message.author.display_name
        
        # Check if message is for us
        is_mention = self.user in message.mentions
        is_command = content.startswith("!chat")
        
        if not (is_mention or is_command):
            return
        
        # Clean up the content
        if is_mention:
            content = content.replace(f'<@{self.user.id}>', '').strip()
        elif is_command:
            content = content[len("!chat"):].strip()
        
        # Handle special commands
        if content.lower() in ["help", "commands"]:
            await self.send_help(message)
            return
        elif content.lower() in ["stats", "status"]:
            await self.send_stats(message)
            return
        elif content.lower() == "clear":
            await self.clear_history(message, channel_id)
            return
        
        # Check rate limits
        if not await rate_limiter.can_send_message(channel_id):
            logger.warning(f"Rate limited in channel {channel_id}")
            return
        
        # Add message to history
        memory_manager.add_message(channel_id, author, content)
        
        # Start typing indicator
        if await rate_limiter.can_start_typing(channel_id):
            try:
                await message.channel.trigger_typing()
            except Exception as e:
                logger.warning(f"Could not trigger typing: {e}")
        
        # Generate and send response
        await self.generate_and_send_response(message, channel_id, content)
    
    async def send_help(self, message):
        """Send help message"""
        embed = discord.Embed(
            title="🤖 ChatterBox Help",
            description="I'm your friendly AI assistant!",
            color=0x00ff88
        )
        embed.add_field(
            name="💬 Chat Commands",
            value="• Mention me to chat: `@ChatterBox hello!`\n"
                  "• Use command: `!chat hello!`",
            inline=False
        )
        embed.add_field(
            name="🛠️ Utility Commands",
            value="• `!chat help` - Show this help\n"
                  "• `!chat stats` - Show bot statistics\n"
                  "• `!chat clear` - Clear chat history",
            inline=False
        )
        embed.add_field(
            name="ℹ️ Features",
            value=f"• Remembers {MAX_CHAT_HISTORY} messages per channel\n"
                  "• Smart response generation\n"
                  "• Multi-provider AI support",
            inline=False
        )
        embed.set_footer(text=f"Provider: {CURRENT_PROVIDER} | Model: {config['model']}")
        
        await message.reply(embed=embed, mention_author=False)
    
    async def send_stats(self, message):
        """Send bot statistics"""
        stats = memory_manager.get_stats()
        uptime = datetime.now() - self.start_time
        
        embed = discord.Embed(
            title="📊 ChatterBox Statistics",
            color=0x0099ff
        )
        embed.add_field(name="⏰ Uptime", value=str(uptime).split('.')[0], inline=True)
        embed.add_field(name="💬 Active Channels", value=stats["active_channels"], inline=True)
        embed.add_field(name="📝 Total Messages", value=stats["total_messages"], inline=True)
        embed.add_field(name="🤖 AI Provider", value=CURRENT_PROVIDER.title(), inline=True)
        embed.add_field(name="🧠 Model", value=config["model"], inline=True)
        embed.add_field(name="📚 Max History", value=f"{MAX_CHAT_HISTORY} msgs", inline=True)
        
        await message.reply(embed=embed, mention_author=False)
    
    async def clear_history(self, message, channel_id):
        """Clear chat history for current channel"""
        if channel_id in memory_manager.chat_history:
            memory_manager.chat_history[channel_id].clear()
            memory_manager.prev_responses.pop(channel_id, None)
            await message.reply("🗑️ Chat history cleared for this channel!", mention_author=False)
        else:
            await message.reply("No chat history to clear in this channel.", mention_author=False)
    
    async def generate_and_send_response(self, message, channel_id, content):
        """Generate and send AI response"""
        context = memory_manager.get_context(channel_id)
        max_attempts = config["retries"]
        
        # Enhanced system prompt
        system_prompt = (
            "You are ChatterBox, a helpful and engaging Discord bot. "
            "Keep responses conversational, friendly, and appropriately sized for Discord chat. "
            "Be natural and avoid being overly formal. Show personality and humor when appropriate. "
            "If the conversation seems to be ending, ask engaging follow-up questions."
        )
        
        for attempt in range(max_attempts):
            try:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Recent conversation:\n{context}\n\nRespond to: {content}"}
                ]
                
                # Add variety to temperature for more diverse responses
                temp_variation = random.uniform(-0.1, 0.1)
                adjusted_temp = max(0.1, min(1.0, config["temperature"] + temp_variation))
                
                response = await call_ai_api(messages, temperature=adjusted_temp)
                
                if not response or not response.get("choices"):
                    if attempt == max_attempts - 1:
                        await message.reply("❌ I'm having trouble generating a response right now. Please try again later!", mention_author=False)
                    continue
                
                ai_message = response["choices"][0]["message"]["content"].strip()
                
                # Check for similarity with previous response
                prev_response = memory_manager.prev_responses.get(channel_id, "")
                if prev_response and difflib.SequenceMatcher(None, ai_message, prev_response).ratio() > SIMILARITY_THRESHOLD:
                    logger.info(f"Response too similar, retrying (attempt {attempt + 1})")
                    continue
                
                # Store and send response
                memory_manager.prev_responses[channel_id] = ai_message
                
                # Split long messages
                if len(ai_message) > 2000:
                    chunks = [ai_message[i:i+1900] for i in range(0, len(ai_message), 1900)]
                    for i, chunk in enumerate(chunks):
                        if i == 0:
                            await message.reply(chunk, mention_author=False)
                        else:
                            await message.channel.send(chunk)
                        await asyncio.sleep(0.5)  # Prevent rate limiting
                else:
                    await message.reply(ai_message, mention_author=False)
                
                logger.info(f"Response sent to channel {channel_id} (attempt {attempt + 1})")
                return
                
            except Exception as e:
                logger.error(f"Error generating response: {e}")
                if attempt == max_attempts - 1:
                    await message.reply("❌ Sorry, I encountered an unexpected error. Please try again!", mention_author=False)
    
    async def close(self):
        """Clean shutdown"""
        if self.cleanup_task:
            self.cleanup_task.cancel()
        await super().close()

# Enhanced intents
intents = discord.Intents.default()
intents.message_content = True
intents.guild_messages = True

# Create and run bot
client = EnhancedChatterBox(intents=intents)

if __name__ == "__main__":
    try:
        if not my_secret:
            raise ValueError("Discord bot token not found. Please set DISCORD_PASS_KEY in your .env file.")
        
        logger.info(f"🚀 Starting Enhanced ChatterBox with {CURRENT_PROVIDER} API...")
        client.run(my_secret)
        
    except KeyboardInterrupt:
        logger.info("Bot shutdown requested by user")
    except Exception as e:
        logger.error(f"Error running bot: {e}")
        logger.info("Make sure you have set the following in your .env file:")
        logger.info("- DISCORD_PASS_KEY=your_discord_bot_token")
        logger.info(f"- {config['api_key_env']}=your_api_key")
        logger.info("- AI_PROVIDER=openai|deepseek|together|groq|local (optional, defaults to openai)")
    finally:
        logger.info("Bot shutdown complete")