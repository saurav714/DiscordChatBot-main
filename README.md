ModernChatterBot

A modern, AI-powered Discord bot with advanced web search capabilities using the Tavily API, built with Python and Discord.py. This bot supports slash commands for web searches, robust error handling, and user-friendly features, making it perfect for interactive Discord communities.

Features



debug.py is used to check your api is working or not

Web Search with Tavily: Use the /search slash command to query the web with configurable depth ("basic" for quick results, "advanced" for deeper analysis).



Enhanced Error Handling: Handles rate limits, timeouts, and API errors with retries and detailed logging.



User-Friendly Results: Search results are presented in a clean Discord embed with titles, snippets, links, and a summary (if available).



Flexible Configuration: Supports customization of search parameters like domains and image inclusion.



Content Filtering: Ensures all inputs and outputs are safe and appropriate.

Prerequisites





Python 3.8 or higher



A Discord bot token from the Discord Developer Portal



A Tavily API key from Tavily



Required Python packages (see Installation)

Installation





Clone the Repository

git clone https://github.com/saurav714/ModernChatBot-main.git
cd DiscordChatBot-main



Install DependenciesInstall the required Python packages using pip:

pip install discord.py python-dotenv aiohttp langchain langchain-community sentence-transformers



Set Up Environment VariablesCreate a .env file in the project root and add the following:

# Required
DISCORD_PASS_KEY=your_discord_bot_token_here
GROQ_API_KEY=your_groq_api_key # other can be but just like OPENAI_API_KEY
TAVILY_API_KEY=your_tavily_api_key

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




Replace your_discord_bot_token with your Discord bot token.



Replace your_tavily_api_key with your Tavily API key.



Invite the Bot to Your Server





Go to the Discord Developer Portal.



Select your application, go to "Bot" tab, and copy the token.



Under "OAuth2" > "URL Generator", select bot and applications.commands scopes.



Grant permissions: Send Messages, Embed Links, Read Message History.



Use the generated URL to invite the bot to your server.

Usage





Run the BotStart the bot with:

python bot.py

The bot will log in, sync slash commands, and be ready to use.



Search the Web





Use the /search slash command in Discord:

/search query:"latest AI news" depth:advanced





query: Your search term (e.g., "latest AI news").



depth: Choose "Basic" (quick) or "Advanced" (deeper analysis).



The bot responds with an embed containing up to 3 results, a summary (if available), search depth, and response time.



Available Commands





/search <query> [depth]: Search the web using Tavily.



/chat <message>: Chat with the AI (if configured with an AI provider).



/preferences: Customize AI response settings.



/stats: View bot statistics.



/clear: Clear chat history in the current channel.



/help: Show help information.

Key Improvements





Configurable Search Depth: Choose "basic" or "advanced" search depth via the /search command.



Enhanced Error Handling: Robust retry logic for rate limits, timeouts, and server errors, with detailed logging.



Better Result Presentation: Search results in a Discord embed with depth and response time in the footer.



Flexibility: Supports future customization with domain inclusion/exclusion and image options.



Content Filtering: Filters inputs and outputs for safety and appropriateness.

Logging





Logs are saved to bot.log in the project directory.



Includes timestamps, success/failure of searches, and error details for debugging.

Contributing





Fork the repository.



Create a new branch (git checkout -b feature/YourFeature).



Make your changes and commit (git commit -m "Add YourFeature").



Push to the branch (git push origin feature/YourFeature).



Open a pull request.

License

This project is licensed under the MIT License. See the LICENSE file for details.

Contact

For issues or suggestions, open an issue on GitHub or contact Arise Tarnished Warrior
saurav1099(Discord).



Built with ❤️ by Saurav on June 08, 2025
