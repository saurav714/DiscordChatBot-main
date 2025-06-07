import discord
from discord.ext import commands, tasks
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
import sqlite3
from typing import Optional, Dict, Any, List
import uuid
import hashlib
import re

# Configure logging with detailed formatting
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
        "model": "gpt-4o-mini",
        "api_key_env": "OPENAI_API_KEY",
        "temperature": float(os.getenv("AI_TEMPERATURE", 0.7)),
        "max_tokens": int(os.getenv("AI_MAX_TOKENS", 1024)),
        "retries": int(os.getenv("AI_RETRIES", 3))
    },
    "xai": {
        "base_url": "https://api.x.ai/v1",
        "model": "grok-beta",
        "api_key_env": "XAI_API_KEY",
        "temperature": float(os.getenv("AI_TEMPERATURE", 0.7)),
        "max_tokens": int(os.getenv("AI_MAX_TOKENS", 1024)),
        "retries": int(os.getenv("AI_RETRIES", 3))
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
        "api_key_env": "DEEPSEEK_API_KEY",
        "temperature": float(os.getenv("AI_TEMPERATURE", 0.7)),
        "max_tokens": int(os.getenv("AI_MAX_TOKENS", 1024)),
        "retries": int(os.getenv("AI_RETRIES", 3))
    },
    "anthropic": {
        "base_url": "https://api.anthropic.com/v1",
        "model": "claude-3-haiku-20240307",
        "api_key_env": "ANTHROPIC_API_KEY",
        "temperature": float(os.getenv("AI_TEMPERATURE", 0.7)),
        "max_tokens": int(os.getenv("AI_MAX_TOKENS", 1024)),
        "retries": int(os.getenv("AI_RETRIES", 3))
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "model": "llama-3.1-8b-instant",
        "api_key_env": "GROQ_API_KEY",
        "temperature": float(os.getenv("AI_TEMPERATURE", 0.7)),
        "max_tokens": int(os.getenv("AI_MAX_TOKENS", 1024)),
        "retries": int(os.getenv("AI_RETRIES", 3))
    }
}

# Choose provider and validate configuration
CURRENT_PROVIDER = os.getenv("AI_PROVIDER", "openai")  # Default to OpenAI
config = API_CONFIG.get(CURRENT_PROVIDER, API_CONFIG["openai"])

# Enhanced configuration
MAX_CHAT_HISTORY = int(os.getenv("MAX_CHAT_HISTORY", 20))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.65))
MEMORY_CLEANUP_INTERVAL = int(os.getenv("MEMORY_CLEANUP_INTERVAL", 3600))
INACTIVE_CHANNEL_TIMEOUT = int(os.getenv("INACTIVE_CHANNEL_TIMEOUT", 86400))
MAX_MESSAGE_LENGTH = int(os.getenv("MAX_MESSAGE_LENGTH", 1900))

def validate_config():
    """Validate configuration and API keys"""
    if CURRENT_PROVIDER not in API_CONFIG:
        raise ValueError(f"Invalid AI_PROVIDER '{CURRENT_PROVIDER}'. Choose from: {list(API_CONFIG.keys())}")
    api_key = os.getenv(config["api_key_env"])
    if not api_key:
        raise ValueError(f"Missing API key. Set {config['api_key_env']} in your .env file.")
    logger.info(f"Configuration validated for provider: {CURRENT_PROVIDER}")

try:
    validate_config()
except ValueError as e:
    logger.error(f"Configuration error: {e}")
    exit(1)

class ResponseCache:
    """Cache responses to improve performance and reduce API calls"""
    
    def __init__(self, max_size: int = 100, ttl: int = 3600):
        self.cache = {}
        self.timestamps = {}
        self.max_size = max_size
        self.ttl = ttl  # Time to live in seconds
    
    def get_cache_key(self, content: str, context: str) -> str:
        """Generate cache key from content and context"""
        combined = f"{content}_{context[:200]}"  # Limit context for key generation
        return hashlib.md5(combined.encode('utf-8')).hexdigest()[:12]
    
    def get(self, key: str) -> Optional[str]:
        """Get cached response if still valid"""
        if key in self.cache:
            if time.time() - self.timestamps[key] < self.ttl:
                return self.cache[key]
            else:
                # Remove expired entry
                del self.cache[key]
                del self.timestamps[key]
        return None
    
    def set(self, key: str, value: str):
        """Cache response with timestamp"""
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.timestamps.keys(), key=lambda k: self.timestamps[k], default=None)
            if oldest_key:
                del self.cache[oldest_key]
                del self.timestamps[oldest_key]
        
        self.cache[key] = value
        self.timestamps[key] = time.time()

class APIHealthChecker:
    """Monitor API health and manage failover"""
    
    def __init__(self, providers: List[str]):
        self.providers = providers
        self.health_status = {provider: True for provider in providers}
        self.last_check = {provider: datetime.now() for provider in providers}
        self.check_interval = timedelta(minutes=5)
    
    async def check_provider_health(self, provider: str) -> bool:
        """Check if a provider is responding"""
        try:
            provider_config = API_CONFIG[provider]
            api_key = os.getenv(provider_config["api_key_env"])
            if not api_key:
                return False
            
            test_messages = [{"role": "user", "content": "Test"}]
            response = await call_ai_api_direct(provider, test_messages, max_tokens=5)
            return response is not None
        except Exception as e:
            logger.warning(f"Health check failed for {provider}: {e}")
            return False
    
    async def get_best_provider(self) -> str:
        """Get the best available provider with health checking"""
        current_time = datetime.now()
        
        for provider in self.providers:
            if current_time - self.last_check[provider] > self.check_interval:
                self.health_status[provider] = await self.check_provider_health(provider)
                self.last_check[provider] = current_time
        
        for provider in self.providers:
            if self.health_status.get(provider, False):
                return provider
        
        primary = self.providers[0] if self.providers else CURRENT_PROVIDER
        if await self.check_provider_health(primary):
            self.health_status[primary] = True
            return primary
        
        return primary  # Fallback to primary

class SmartContextManager:
    """Intelligent context management with relevance scoring"""
    
    def __init__(self):
        self.context_weights = defaultdict(float)
        self.topic_keywords = defaultdict(set)
    
    def extract_keywords(self, text: str) -> set:
        """Extract key terms from text"""
        if not text or not isinstance(text, str):
            return set()
        
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        stopwords = {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'her', 
            'was', 'one', 'our', 'had', 'have', 'they', 'will', 'what', 'this', 
            'that', 'with', 'from', 'how', 'when', 'where', 'why', 'who', 'would',
            'could', 'should', 'there', 'here', 'said', 'get', 'got', 'make', 'made'
        }
        return set(word for word in words if word not in stopwords and len(word) > 2)
    
    def calculate_relevance(self, current_msg: str, historical_msg: str) -> float:
        """Calculate relevance score between messages"""
        current_keywords = self.extract_keywords(current_msg)
        historical_keywords = self.extract_keywords(historical_msg)
        
        if not current_keywords or not historical_keywords:
            return 0.0
        
        intersection = current_keywords.intersection(historical_keywords)
        union = current_keywords.union(historical_keywords)
        
        return len(intersection) / len(union) if union else 0.0
    
    def get_smart_context(self, channel_id: int, current_message: str, history: List[str]) -> str:
        """Get contextually relevant conversation history"""
        if not history:
            return "No previous context."
        
        scored_messages = []
        for i, msg in enumerate(history):
            relevance = self.calculate_relevance(current_message, msg)
            recency_bonus = (i / len(history)) * 0.3
            total_score = relevance + recency_bonus
            scored_messages.append((msg, total_score))
        
        scored_messages.sort(key=lambda x: x[1], reverse=True)
        
        relevant_msgs = [msg[0] for msg in scored_messages[:6]]
        recent_msgs = history[-3:]
        
        context_msgs = []
        seen = set()
        for msg in recent_msgs + relevant_msgs:
            if msg not in seen:
                context_msgs.append(msg)
                seen.add(msg)
        
        return "Relevant context:\n" + "\n".join(context_msgs[:8])

class ConversationManager:
    """Manage conversation threads and context"""
    
    def __init__(self):
        self.conversations = defaultdict(dict)
        self.thread_contexts = defaultdict(list)
    
    def create_thread(self, channel_id: int, user_id: int, topic: str) -> str:
        """Create a new conversation thread"""
        thread_id = str(uuid.uuid4())[:8]
        self.conversations[channel_id][thread_id] = {
            "user_id": user_id,
            "topic": topic,
            "created": datetime.now(),
            "messages": []
        }
        return thread_id
    
    def add_to_thread(self, channel_id: int, thread_id: str, message: str, is_user: bool = True):
        """Add message to specific thread"""
        if thread_id in self.conversations[channel_id]:
            self.conversations[channel_id][thread_id]["messages"].append({
                "content": message,
                "is_user": is_user,
                "timestamp": datetime.now()
            })
    
    def get_thread_context(self, channel_id: int, thread_id: str) -> str:
        """Get focused context for a specific thread"""
        if thread_id not in self.conversations[channel_id]:
            return ""
        
        thread = self.conversations[channel_id][thread_id]
        context_parts = [f"Thread Topic: {thread['topic']}", "Thread Conversation:"]
        
        for msg in thread["messages"][-10:]:
            prefix = "User" if msg["is_user"] else "Assistant"
            time_str = msg["timestamp"].strftime("%H:%M")
            context_parts.append(f"[{time_str}] {prefix}: {msg['content']}")
        
        return "\n".join(context_parts)
    
    def list_user_threads(self, channel_id: int, user_id: int) -> List[Dict]:
        """List all threads for a user in a channel"""
        user_threads = []
        for thread_id, thread_data in self.conversations[channel_id].items():
            if thread_data["user_id"] == user_id:
                user_threads.append({
                    "id": thread_id,
                    "topic": thread_data["topic"],
                    "created": thread_data["created"],
                    "message_count": len(thread_data["messages"])
                })
        return sorted(user_threads, key=lambda x: x["created"], reverse=True)

class UserAnalytics:
    """Track user interaction patterns and provide insights"""
    
    def __init__(self, db_manager):
        self.db = db_manager
        self.setup_analytics_tables()
    
    def setup_analytics_tables(self):
        """Setup analytics tables"""
        try:
            with sqlite3.connect(self.db.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS user_analytics (
                        user_id INTEGER,
                        channel_id INTEGER,
                        interaction_type TEXT,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        message_length INTEGER,
                        response_time REAL,
                        success BOOLEAN DEFAULT TRUE,
                        provider_used TEXT
                    )
                """)
                
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS server_config (
                        guild_id INTEGER PRIMARY KEY,
                        auto_respond BOOLEAN DEFAULT TRUE,
                        max_history INTEGER DEFAULT 20,
                        response_delay INTEGER DEFAULT 0,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.commit()
        except Exception as e:
            logger.error(f"Error setting up analytics tables: {e}")
    
    def log_interaction(self, user_id: int, channel_id: int, interaction_type: str, 
                       message_length: int = 0, response_time: float = 0.0, 
                       success: bool = True, provider_used: str = ""):
        """Log user interaction"""
        try:
            with sqlite3.connect(self.db.db_path) as conn:
                conn.execute("""
                    INSERT INTO user_analytics 
                    (user_id, channel_id, interaction_type, message_length, response_time, success, provider_used)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (user_id, channel_id, interaction_type, message_length, response_time, success, provider_used))
                conn.commit()
        except Exception as e:
            logger.error(f"Error logging interaction: {e}")
    
    def get_user_stats(self, user_id: int) -> Dict[str, Any]:
        """Get comprehensive user statistics"""
        try:
            with sqlite3.connect(self.db.db_path) as conn:
                cursor = conn.execute("""
                    SELECT 
                        COUNT(*) as total_interactions,
                        AVG(message_length) as avg_message_length,
                        AVG(response_time) as avg_response_time,
                        SUM(CASE WHEN success THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as success_rate,
                        MAX(timestamp) as last_interaction,
                        COUNT(DISTINCT channel_id) as channels_used
                    FROM user_analytics 
                    WHERE user_id = ?
                """, (user_id,))
                
                result = cursor.fetchone()
                if result and result[0] > 0:
                    return {
                        "total_interactions": result[0],
                        "avg_message_length": round(result[1] or 0, 1),
                        "avg_response_time": round(result[2] or 0, 2),
                        "success_rate": round(result[3] or 0, 1),
                        "last_interaction": result[4],
                        "channels_used": result[5]
                    }
        except Exception as e:
            logger.error(f"Error getting user stats: {e}")
        return {}
    
    def get_server_stats(self, guild_id: int) -> Dict[str, Any]:
        """Get server-wide statistics"""
        try:
            with sqlite3.connect(self.db.db_path) as conn:
                cursor = conn.execute("""
                    SELECT 
                        COUNT(DISTINCT user_id) as unique_users,
                        COUNT(*) as total_interactions,
                        AVG(response_time) as avg_response_time,
                        COUNT(DISTINCT channel_id) as active_channels
                    FROM user_analytics 
                    WHERE channel_id IN (
                        SELECT DISTINCT channel_id FROM user_analytics
                    )
                """)
                
                result = cursor.fetchone()
                if result:
                    return {
                        "unique_users": result[0] or 0,
                        "total_interactions": result[1] or 0,
                        "avg_response_time": round(result[2] or 0, 2),
                        "active_channels": result[3] or 0
                    }
        except Exception as e:
            logger.error(f"Error getting server stats: {e}")
        return {}

class DatabaseManager:
    """Enhanced database manager with additional features"""
    
    def __init__(self, db_path: str = "chatbot.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS user_preferences (
                        user_id INTEGER PRIMARY KEY,
                        username TEXT,
                        temperature REAL DEFAULT 0.7,
                        max_tokens INTEGER DEFAULT 1024,
                        personality TEXT DEFAULT 'friendly',
                        preferred_provider TEXT DEFAULT '',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS conversation_stats (
                        channel_id INTEGER PRIMARY KEY,
                        message_count INTEGER DEFAULT 0,
                        last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        total_tokens_used INTEGER DEFAULT 0
                    )
                """)
                
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS chat_sessions (
                        session_id TEXT PRIMARY KEY,
                        channel_id INTEGER,
                        user_id INTEGER,
                        messages TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS server_config (
                        guild_id INTEGER PRIMARY KEY,
                        auto_respond BOOLEAN DEFAULT TRUE,
                        max_history INTEGER DEFAULT 20,
                        response_delay INTEGER DEFAULT 0,
                        allowed_channels TEXT DEFAULT '',
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.commit()
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
    
    def get_user_preferences(self, user_id: int) -> Dict[str, Any]:
        """Get user preferences from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT temperature, max_tokens, personality, preferred_provider FROM user_preferences WHERE user_id = ?",
                    (user_id,)
                )
                result = cursor.fetchone()
                if result:
                    return {
                        "temperature": result[0],
                        "max_tokens": result[1],
                        "personality": result[2],
                        "preferred_provider": result[3] or ""
                    }
        except Exception as e:
            logger.error(f"Error getting user preferences: {e}")
        return {}
    
    def update_user_preferences(self, user_id: int, username: str, **preferences):
        """Update user preferences in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO user_preferences (user_id, username, temperature, max_tokens, personality, preferred_provider, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(user_id) DO UPDATE SET
                        username = excluded.username,
                        temperature = COALESCE(excluded.temperature, temperature),
                        max_tokens = COALESCE(excluded.max_tokens, max_tokens),
                        personality = COALESCE(excluded.personality, personality),
                        preferred_provider = COALESCE(excluded.preferred_provider, preferred_provider),
                        updated_at = CURRENT_TIMESTAMP
                """, (
                    user_id, username,
                    preferences.get('temperature'),
                    preferences.get('max_tokens'),
                    preferences.get('personality'),
                    preferences.get('preferred_provider')
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Error updating user preferences: {e}")
    
    def get_server_config(self, guild_id: int) -> Dict[str, Any]:
        """Get server configuration"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT auto_respond, max_history, response_delay, allowed_channels FROM server_config WHERE guild_id = ?",
                    (guild_id,)
                )
                result = cursor.fetchone()
                if result:
                    return {
                        "auto_respond": bool(result[0]),
                        "max_history": result[1],
                        "response_delay": result[2],
                        "allowed_channels": result[3].split(',') if result[3] else []
                    }
        except Exception as e:
            logger.error(f"Error getting server config: {e}")
        return {"auto_respond": True, "max_history": 20, "response_delay": 0, "allowed_channels": []}
    
    def update_server_config(self, guild_id: int, **config):
        """Update server configuration"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                allowed_channels = ','.join(map(str, config.get('allowed_channels', []))) if config.get('allowed_channels') else ''
                conn.execute("""
                    INSERT INTO server_config (guild_id, auto_respond, max_history, response_delay, allowed_channels, updated_at)
                    VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(guild_id) DO UPDATE SET
                        auto_respond = COALESCE(excluded.auto_respond, auto_respond),
                        max_history = COALESCE(excluded.max_history, max_history),
                        response_delay = COALESCE(excluded.response_delay, response_delay),
                        allowed_channels = COALESCE(excluded.allowed_channels, allowed_channels),
                        updated_at = CURRENT_TIMESTAMP
                """, (
                    guild_id,
                    config.get('auto_respond'),
                    config.get('max_history'),
                    config.get('response_delay'),
                    allowed_channels
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Error updating server config: {e}")

class EnhancedMemoryManager:
    """Enhanced memory management with persistence and user context"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.chat_history = defaultdict(list)
        self.prev_responses = defaultdict(str)
        self.channel_last_activity = defaultdict(lambda: datetime.now())
        self.user_contexts = defaultdict(dict)
        self.db = db_manager
        self.smart_context = SmartContextManager()
    
    def add_message(self, channel_id: int, author: str, content: str, user_id: int = None):
        """Add a message to chat history with enhanced context"""
        timestamp = datetime.now().strftime("%H:%M")
        formatted_message = f"[{timestamp}] {author}: {content}"
        self.chat_history[channel_id].append(formatted_message)
        self.channel_last_activity[channel_id] = datetime.now()
        
        if user_id:
            if user_id not in self.user_contexts:
                self.user_contexts[user_id] = {
                    "name": author,
                    "last_seen": datetime.now(),
                    "message_count": 0,
                    "favorite_topics": set()
                }
            self.user_contexts[user_id]["message_count"] += 1
            self.user_contexts[user_id]["last_seen"] = datetime.now()
            
            keywords = self.smart_context.extract_keywords(content)
            self.user_contexts[user_id]["favorite_topics"].update(keywords)
        
        server_config = self.db.get_server_config(channel_id)  # Assuming guild_id == channel_id for simplicity
        max_history = server_config.get("max_history", MAX_CHAT_HISTORY)
        
        if len(self.chat_history[channel_id]) > max_history:
            self.chat_history[channel_id] = self.chat_history[channel_id][-max_history:]
    
    def get_enhanced_context(self, channel_id: int, user_id: int = None, current_message: str = "") -> str:
        """Get enhanced context including user history and smart context"""
        history = self.chat_history[channel_id]
        
        if current_message and history:
            context = self.smart_context.get_smart_context(channel_id, current_message, history)
        else:
            if not history:
                context = "No previous conversation context."
            else:
                context = "Recent conversation:\n" + "\n".join(history[-8:])
        
        if user_id and user_id in self.user_contexts:
            user_ctx = self.user_contexts[user_id]
            topics = list(user_ctx["favorite_topics"])[:5]
            user_info = f"\nUser context: {user_ctx['name']} has sent {user_ctx['message_count']} messages."
            if topics:
                user_info += f" Interested in: {', '.join(topics)}"
            context += user_info
        
        return context
    
    def cleanup_inactive_channels(self):
        """Enhanced cleanup with database updates"""
        current_time = datetime.now()
        inactive_channels = []
        
        for channel_id, last_activity in self.channel_last_activity.items():
            if current_time - last_activity > timedelta(seconds=INACTIVE_CHANNEL_TIMEOUT):
                inactive_channels.append(channel_id)
        
        for channel_id in inactive_channels:
            self.chat_history.pop(channel_id, None)
            self.prev_responses.pop(channel_id, None)
            self.channel_last_activity.pop(channel_id, None)
            logger.info(f"Cleaned up inactive channel: {channel_id}")

class RateLimiter:
    """Enhanced rate limiting with user-specific limits"""
    
    def __init__(self):
        self.message_timestamps = defaultdict(list)
        self.typing_timestamps = defaultdict(list)
        self.user_limits = defaultdict(lambda: {"count": 0, "reset_time": time.time() + 60})
        self.channel_delays = defaultdict(float)
    
    async def can_user_send_request(self, user_id: int, limit: int = 15) -> bool:
        """Check if user can send request (per-user rate limiting)"""
        now = time.time()
        user_data = self.user_limits[user_id]
        
        if now > user_data["reset_time"]:
            user_data["count"] = 0
            user_data["reset_time"] = now + 60
        
        if user_data["count"] >= limit:
            return False
        
        user_data["count"] += 1
        return True
    
    async def can_send_message(self, channel_id: int) -> bool:
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
    
    async def apply_response_delay(self, channel_id: int, delay: int = 0):
        """Apply configured response delay"""
        if delay > 0:
            await asyncio.sleep(delay)

async def call_ai_api_direct(provider: str, messages: list, temperature: float = None, max_tokens: int = None, retries: int = None) -> Optional[dict]:
    """Direct AI API call for specific provider"""
    provider_config = API_CONFIG.get(provider, API_CONFIG["openai"])
    headers = {"Content-Type": "application/json"}
    api_key = os.getenv(provider_config["api_key_env"])
    
    if not api_key:
        logger.error(f"No API key for {provider}")
        return None
    
    if provider == "anthropic":
        headers["x-api-key"] = api_key
        headers["anthropic-version"] = "2023-06-01"
    else:
        headers["Authorization"] = f"Bearer {api_key}"
    
    if provider == "anthropic":
        system_msg = next((msg["content"] for msg in messages if msg["role"] == "system"), "")
        user_messages = [msg for msg in messages if msg["role"] != "system"]
        
        payload = {
            "model": provider_config["model"],
            "max_tokens": max_tokens or provider_config["max_tokens"],
            "temperature": temperature or provider_config["temperature"],
            "messages": user_messages
        }
        if system_msg:
            payload["system"] = system_msg
        
        endpoint = f"{provider_config['base_url']}/messages"
    else:
        payload = {
            "model": provider_config["model"],
            "messages": messages,
            "temperature": temperature or provider_config["temperature"],
            "max_tokens": max_tokens or provider_config["max_tokens"],
            "top_p": 0.9,
            "frequency_penalty": 0.3,
            "presence_penalty": 0.6,
        }
        endpoint = f"{provider_config['base_url']}/chat/completions"
    
    max_retries = retries or provider_config["retries"]
    
    for attempt in range(max_retries):
        try:
            timeout = aiohttp.ClientTimeout(total=60)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(endpoint, headers=headers, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data
                    elif response.status == 429:
                        retry_after = int(response.headers.get('Retry-After', 2 ** attempt))
                        logger.warning(f"Rate limited for {provider}, waiting {retry_after}s (attempt {attempt + 1}/{max_retries})")
                        await asyncio.sleep(retry_after)
                    elif response.status == 401:
                        logger.error(f"Authentication error for {provider}: Invalid API key")
                        return None
                    elif response.status >= 500:
                        logger.warning(f"Server error {response.status} for {provider}, retrying (attempt {attempt + 1}/{max_retries})")
                        await asyncio.sleep(2 ** attempt)
                    else:
                        error_text = await response.text()
                        logger.error(f"API request failed for {provider} with status {response.status}: {error_text}")
                        return None
        except asyncio.TimeoutError:
            logger.warning(f"Request timeout for {provider} (attempt {attempt + 1}/{max_retries})")
            await asyncio.sleep(2 ** attempt)
        except Exception as e:
            logger.error(f"Unexpected error for {provider}: {e} (attempt {attempt + 1}/{max_retries})")
            await asyncio.sleep(2 ** attempt)
    
    logger.error(f"Max retries exceeded for API call to {provider}")
    return None

async def call_ai_api(messages: list, temperature: float = None, max_tokens: int = None, retries: int = None) -> Optional[dict]:
    """Call AI API with provider failover"""
    health_checker = APIHealthChecker(list(API_CONFIG.keys()))
    provider = await health_checker.get_best_provider()
    logger.info(f"Using provider: {provider}")
    
    response = await call_ai_api_direct(provider, messages, temperature, max_tokens, retries)
    if response:
        return response
    
    # Try fallback providers
    for fallback in API_CONFIG.keys():
        if fallback != provider:
            logger.warning(f"Falling back to provider: {fallback}")
            response = await call_ai_api_direct(fallback, messages, temperature, max_tokens, retries)
            if response:
                return response
    
    logger.error("All API providers failed")
    return None

class ModernChatterBot(commands.Bot):
    """Modern Discord bot with slash commands and enhanced features"""
    
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        intents.guild_messages = True
        
        super().__init__(
            command_prefix='!',
            intents=intents,
            help_command=None,
            case_insensitive=True
        )
        
        self.db_manager = DatabaseManager()
        self.memory_manager = EnhancedMemoryManager(self.db_manager)
        self.rate_limiter = RateLimiter()
        self.start_time = datetime.now()
        self.conversation_manager = ConversationManager()
        self.analytics = UserAnalytics(self.db_manager)
    
    async def setup_hook(self):
        """Setup hook for initializing slash commands"""
        try:
            synced = await self.tree.sync()
            logger.info(f"Synced {len(synced)} slash commands")
        except Exception as e:
            logger.error(f"Failed to sync slash commands: {e}")
    
    async def on_ready(self):
        logger.info(f"🤖 {self.user} is online and ready!")
        logger.info(f"📡 Using AI provider: {CURRENT_PROVIDER}")
        logger.info(f"🧠 Model: {config['model']}")
        logger.info(f"🔧 Max history: {MAX_CHAT_HISTORY} messages")
        logger.info(f"🎯 Similarity threshold: {SIMILARITY_THRESHOLD}")
        logger.info(f"⚡ Slash commands enabled")
        
        if not self.cleanup_task.is_running():
            self.cleanup_task.start()
    
    @tasks.loop(seconds=MEMORY_CLEANUP_INTERVAL)
    async def cleanup_task(self):
        """Periodic cleanup task"""
        try:
            self.memory_manager.cleanup_inactive_channels()
            stats = self.get_stats()
            logger.info(f"Memory stats: {stats}")
        except Exception as e:
            logger.error(f"Error in cleanup task: {e}")
    
    @cleanup_task.before_loop
    async def before_cleanup_task(self):
        await self.wait_until_ready()
    
    def get_stats(self) -> dict:
        """Get bot statistics"""
        return {
            "active_channels": len(self.memory_manager.chat_history),
            "total_messages": sum(len(history) for history in self.memory_manager.chat_history.values()),
            "unique_users": len(self.memory_manager.user_contexts),
            "guilds": len(self.guilds),
            "uptime": str(datetime.now() - self.start_time).split('.')[0]
        }
    
    @discord.app_commands.command(name="chat", description="Chat with the AI assistant")
    @discord.app_commands.describe(message="Your message to the AI")
    async def slash_chat(self, interaction: discord.Interaction, message: str):
        """Modern slash command for chatting"""
        if not await self.rate_limiter.can_user_send_request(interaction.user.id):
            await interaction.response.send_message("⏰ You're sending messages too quickly. Please wait a moment!", ephemeral=True)
            return
        
        await interaction.response.defer(thinking=True)
        
        try:
            self.memory_manager.add_message(
                interaction.channel_id,
                interaction.user.display_name,
                message,
                interaction.user.id
            )
            
            user_prefs = self.db_manager.get_user_preferences(interaction.user.id)
            response = await self.generate_ai_response(
                interaction.channel_id,
                message,
                interaction.user.id,
                user_prefs
            )
            
            self.analytics.log_interaction(
                user_id=interaction.user.id,
                channel_id=interaction.channel_id,
                interaction_type="chat",
                message_length=len(message),
                response_time=0.0,  # Could be enhanced with actual timing
                success=bool(response),
                provider_used=CURRENT_PROVIDER
            )
            
            if response:
                await interaction.followup.send(response)
            else:
                await interaction.followup.send("❌ I'm having trouble generating a response right now. Please try again later!")
        except Exception as e:
            logger.error(f"Error in slash_chat: {e}")
            await interaction.followup.send("❌ An unexpected error occurred. Please try again!")
    
    @discord.app_commands.command(name="preferences", description="Set your AI chat preferences")
    @discord.app_commands.describe(
        temperature="Response creativity (0.1-1.0)",
        personality="AI personality style",
        max_tokens="Maximum response length"
    )
    @discord.app_commands.choices(personality=[
        discord.app_commands.Choice(name="Friendly", value="friendly"),
        discord.app_commands.Choice(name="Professional", value="professional"),
        discord.app_commands.Choice(name="Casual", value="casual"),
        discord.app_commands.Choice(name="Humorous", value="humorous"),
    ])
    async def set_preferences(
        self,
        interaction: discord.Interaction,
        temperature: Optional[float] = None,
        personality: Optional[str] = None,
        max_tokens: Optional[int] = None
    ):
        """Set user preferences for AI responses"""
        prefs = {}
        if temperature is not None:
            if not 0.1 <= temperature <= 1.0:
                await interaction.response.send_message("Temperature must be between 0.1 and 1.0!", ephemeral=True)
                return
            prefs['temperature'] = temperature
        
        if personality:
            prefs['personality'] = personality
        
        if max_tokens is not None:
            if not 100 <= max_tokens <= 2000:
                await interaction.response.send_message("Max tokens must be between 100 and 2000!", ephemeral=True)
                return
            prefs['max_tokens'] = max_tokens
        
        self.db_manager.update_user_preferences(
            interaction.user.id,
            interaction.user.display_name,
            **prefs
        )
        
        await interaction.response.send_message(f"✅ Preferences updated: {', '.join(f'{k}={v}' for k, v in prefs.items())}", ephemeral=True)
    
    @discord.app_commands.command(name="stats", description="View bot statistics")
    async def slash_stats(self, interaction: discord.Interaction):
        """Display bot statistics"""
        stats = self.get_stats()
        
        embed = discord.Embed(
            title="📊 ChatterBot Statistics",
            color=0x0099ff,
            timestamp=datetime.now()
        )
        
        embed.add_field(name="⏰ Uptime", value=stats["uptime"], inline=True)
        embed.add_field(name="🏰 Servers", value=stats["guilds"], inline=True)
        embed.add_field(name="💬 Active Channels", value=stats["active_channels"], inline=True)
        embed.add_field(name="👥 Unique Users", value=stats["unique_users"], inline=True)
        embed.add_field(name="📝 Total Messages", value=stats["total_messages"], inline=True)
        embed.add_field(name="🤖 AI Provider", value=CURRENT_PROVIDER.title(), inline=True)
        embed.add_field(name="🧠 Model", value=config["model"], inline=False)
        
        embed.set_footer(text=f"Bot Version 2.0 | {self.user.name}")
        
        await interaction.response.send_message(embed=embed)
    
    @discord.app_commands.command(name="clear", description="Clear your chat history in this channel")
    async def slash_clear(self, interaction: discord.Interaction):
        """Clear chat history for current channel"""
        channel_id = interaction.channel_id
        
        if channel_id in self.memory_manager.chat_history:
            self.memory_manager.chat_history[channel_id].clear()
            self.memory_manager.prev_responses.pop(channel_id, None)
            await interaction.response.send_message("🗑️ Chat history cleared for this channel!", ephemeral=True)
        else:
            await interaction.response.send_message("No chat history to clear in this channel.", ephemeral=True)
    
    @discord.app_commands.command(name="help", description="Show bot commands and usage")
    async def slash_help(self, interaction: discord.Interaction):
        """Display help information"""
        embed = discord.Embed(
            title="🤖 ChatterBot Help",
            description="A modern AI-powered Discord bot with multiple provider support",
            color=0x00ff00
        )
        
        embed.add_field(
            name="📝 Chat Commands",
            value="`/chat <message>` - Chat with the AI\n`@ChatterBot <message>` - Mention to chat\n`!chat <message>` - Legacy command",
            inline=False
        )
        
        embed.add_field(
            name="⚙️ Settings",
            value="`/preferences` - Set your AI preferences\n`/clear` - Clear channel chat history",
            inline=False
        )
        
        embed.add_field(
            name="📊 Information",
            value="`/stats` - View bot statistics\n`/help` - Show this help message",
            inline=False
        )
        
        embed.add_field(
            name="🎭 Personalities",
            value="**Friendly** - Warm and supportive\n**Professional** - Formal and informative\n**Casual** - Relaxed and laid-back\n**Humorous** - Witty and fun",
            inline=False
        )
        
        embed.add_field(
            name="🔧 Current Setup",
            value=f"**Provider:** {CURRENT_PROVIDER.title()}\n**Model:** {config['model']}\n**Max History:** {MAX_CHAT_HISTORY} messages",
            inline=False
        )
        
        embed.set_footer(text="ChatterBot v2.0 - Built with ❤️")
        
        await interaction.response.send_message(embed=embed)
    
    async def on_message(self, message: discord.Message):
        """Handle incoming messages"""
        if message.author == self.user:
            return
        
        if not message.guild:
            await message.channel.send("❌ This bot only works in server channels!")
            return
        
        server_config = self.db_manager.get_server_config(message.guild.id)
        if message.channel.id not in server_config.get("allowed_channels", []) and server_config.get("allowed_channels"):
            return
        
        await self.process_commands(message)
        
        content = message.content.strip()
        
        is_mention = self.user in message.mentions
        is_command = content.startswith("!chat")
        
        if not (is_mention or is_command):
            return
        
        if is_mention:
            content = content.replace(f'<@{self.user.id}>', '').strip()
        elif is_command:
            content = content[len("!chat"):].strip()
        
        if not content:
            return
        
        if not await self.rate_limiter.can_user_send_request(message.author.id):
            await message.reply("⏰ You're sending messages too quickly. Please wait a moment!", mention_author=False)
            return
        
        await self.process_chat_message(message, content)
    
    async def process_chat_message(self, message: discord.Message, content: str):
        """Process chat message with typing indicator"""
        try:
            if not await self.rate_limiter.can_send_message(message.channel.id):
                await message.reply("⏰ Bot is busy! Please wait a moment and try again.", mention_author=False)
                return
            
            async with message.channel.typing():
                self.memory_manager.add_message(
                    message.channel.id,
                    message.author.display_name,
                    content,
                    message.author.id
                )
                
                user_prefs = self.db_manager.get_user_preferences(message.author.id)
                
                start_time = time.time()
                response = await self.generate_ai_response(
                    message.channel.id,
                    content,
                    message.author.id,
                    user_prefs
                )
                response_time = time.time() - start_time
                
                self.analytics.log_interaction(
                    user_id=message.author.id,
                    channel_id=message.channel.id,
                    interaction_type="chat",
                    message_length=len(content),
                    response_time=response_time,
                    success=bool(response),
                    provider_used=CURRENT_PROVIDER
                )
                
                if response:
                    if len(response) > MAX_MESSAGE_LENGTH:
                        chunks = [response[i:i+MAX_MESSAGE_LENGTH] for i in range(0, len(response), MAX_MESSAGE_LENGTH)]
                        for i, chunk in enumerate(chunks):
                            if i == 0:
                                await message.reply(chunk, mention_author=False)
                            else:
                                await message.channel.send(chunk)
                            await asyncio.sleep(0.5)
                    else:
                        await message.reply(response, mention_author=False)
                else:
                    await message.reply("❌ I'm having trouble generating a response right now. Please try again later!", mention_author=False)
        
        except discord.errors.Forbidden:
            logger.error(f"Missing permissions to send message in channel {message.channel.id}")
            await message.channel.send("❌ I don't have permission to send messages here!")
        except Exception as e:
            logger.error(f"Error processing chat message: {e}")
            await message.reply("❌ An unexpected error occurred. Please try again!", mention_author=False)
    
    async def generate_ai_response(self, channel_id: int, content: str, user_id: int, user_prefs: dict) -> Optional[str]:
        """Generate AI response with user preferences"""
        try:
            context = self.memory_manager.get_enhanced_context(channel_id, user_id, content)
            
            personality_prompts = {
                "friendly": "You are ChatterBox, a warm and friendly Discord bot. Be conversational, supportive, and engaging. Keep responses concise but helpful.",
                "professional": "You are ChatterBox, a professional Discord assistant. Be helpful, concise, and informative with a formal tone.",
                "casual": "You are ChatterBox, a chill Discord bot. Keep things relaxed, use casual language, and be laid-back. Don't be too formal.",
                "humorous": "You are ChatterBox, a witty Discord bot. Add appropriate humor and personality to your responses. Be playful but not offensive."
            }
            
            personality = user_prefs.get("personality", "friendly")
            system_prompt = personality_prompts.get(personality, personality_prompts["friendly"])
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context}\n\nRespond to: {content}"}
            ]
            
            temperature = user_prefs.get("temperature", config["temperature"])
            max_tokens = user_prefs.get("max_tokens", config["max_tokens"])
            
            temp_variation = random.uniform(-0.05, 0.05)
            adjusted_temp = max(0.1, min(1.0, temperature + temp_variation))
            
            response = await call_ai_api(messages, temperature=adjusted_temp, max_tokens=max_tokens)
            
            if not response:
                return None
            
            if CURRENT_PROVIDER == "anthropic":
                if response.get("content") and len(response["content"]) > 0:
                    ai_message = response["content"][0]["text"].strip()
                else:
                    return None
            else:
                if response.get("choices") and len(response["choices"]) > 0:
                    ai_message = response["choices"][0]["message"]["content"].strip()
                else:
                    return None
            
            prev_response = self.memory_manager.prev_responses.get(channel_id, "")
            if prev_response and difflib.SequenceMatcher(None, ai_message, prev_response).ratio() > SIMILARITY_THRESHOLD:
                ai_message += f" {random.choice(['What do you think?', 'Any thoughts on that?', 'How does that sound?'])}"
            
            self.memory_manager.prev_responses[channel_id] = ai_message
            
            return ai_message
            
        except Exception as e:
            logger.error(f"Error generating AI response: {e}")
            return None

def main():
    """Main function to run the bot"""
    discord_token = os.getenv("DISCORD_PASS_KEY")
    
    if not discord_token:
        raise ValueError("Discord bot token not found. Please set DISCORD_PASS_KEY in your .env file.")
    
    bot = ModernChatterBot()
    
    try:
        logger.info(f"🚀 Starting Modern ChatterBot with {CURRENT_PROVIDER} API...")
        bot.run(discord_token)
    except KeyboardInterrupt:
        logger.info("Bot shutdown requested by user")
    except Exception as e:
        logger.error(f"Error running bot: {e}")
        logger.info("Make sure you have set the following in your .env file:")
        logger.info("- DISCORD_PASS_KEY=your_discord_bot_token")
        logger.info(f"- {config['api_key_env']}=your_api_key")
        logger.info("- AI_PROVIDER=openai|xai|deepseek|anthropic|groq (optional, defaults to openai)")
    finally:
        logger.info("Bot shutdown complete")

if __name__ == "__main__":
    main()