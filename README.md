ModernChatterBot
A modern, AI-powered Discord bot with advanced web search capabilities using the Tavily API, built with Python and discord.py. Ideal for interactive Discord communities, this bot supports slash commands, error handling, and configurable features for a seamless user experience.

🌟 Features
🔍 Web Search with Tavily
Use the /search slash command with configurable depth (basic for quick, advanced for deep analysis).

⚠️ Enhanced Error Handling
Smart retry logic for rate limits, timeouts, and API errors with detailed logging.

🧾 User-Friendly Results
Clean Discord embeds with titles, snippets, direct links, and AI-generated summaries.

⚙️ Flexible Configuration
Customize parameters such as search domains, image inclusion, memory, and chat length.

🛡️ Content Filtering
Ensures all queries and responses are safe and appropriate.

🧪 Debug Mode
Includes debug.py for testing your API integrations.

🔧 Prerequisites
Python 3.8+

Discord bot token (Discord Developer Portal)

Tavily API key (Tavily.com)

(Optional) AI Provider API key (Groq, OpenAI, Claude, etc.)

📦 Installation
1. Clone the Repository
bash
Copy
Edit
git clone https://github.com/saurav714/ModernChatBot-main.git
cd ModernChatBot-main
2. Install Dependencies
bash
Copy
Edit
pip install discord.py python-dotenv aiohttp langchain langchain-community sentence-transformers
3. Set Up Environment Variables
Create a .env file in the project root:

env
Copy
Edit
# Required
DISCORD_PASS_KEY=your_discord_bot_token_here
TAVILY_API_KEY=your_tavily_api_key
GROQ_API_KEY=your_groq_api_key  # or OPENAI_API_KEY / CLAUDE_API_KEY

# Optional
AI_PROVIDER=groq
AI_TEMPERATURE=0.7
AI_MAX_TOKENS=1024
AI_RETRIES=3
TAVILY_MAX_RESULTS=5
MAX_CHAT_HISTORY=20
SIMILARITY_THRESHOLD=0.65
MEMORY_CLEANUP_INTERVAL=3600
INACTIVE_CHANNEL_TIMEOUT=86400
MAX_MESSAGE_LENGTH=1900
CHROMA_PERSIST_DIR=./chroma_db
🤖 Invite the Bot to Your Discord Server
Go to the Discord Developer Portal

Select your bot → Bot tab → Copy token.

Go to OAuth2 → URL Generator:

Scopes: bot, applications.commands

Bot Permissions: Send Messages, Embed Links, Read Message History

Use the generated URL to invite the bot to your server.

🚀 Usage
Start the Bot
bash
Copy
Edit
python bot.py
Once started, the bot will log in, sync slash commands, and be ready for use.

Slash Commands
Command	Description
/search	Web search using Tavily API
/chat	Chat with AI (Groq/OpenAI/Claude/etc.)
/preferences	Customize your AI response preferences
/stats	View bot statistics
/clear	Clear chat history in the current channel
/help	Display help information

Example Search Command
bash
Copy
Edit
/search query:"latest AI news" depth:advanced
query: Your search topic.

depth: Choose between basic and advanced.

📈 Logging
Logs are stored in bot.log, including:

Timestamps

Search success/failure

API errors and retries

🤝 Contributing
Fork the repository.

Create a feature branch:

bash
Copy
Edit
git checkout -b feature/YourFeature
Make changes and commit:

bash
Copy
Edit
git commit -m "Add YourFeature"
Push and open a pull request.

📜 License
This project is licensed under the MIT License.

📬 Contact
For suggestions, issues, or collaborations, feel free to:

Open an issue here on GitHub

Reach out on Discord: saurav1099

Built with ❤️ by Saurav — Arise, Tarnished Warrior!
Created on June 08, 2025
