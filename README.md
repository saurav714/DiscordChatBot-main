🤖 Discord Multi-AI ChatBot
A powerful, flexible Discord chatbot that supports multiple AI providers including OpenAI GPT-4, DeepSeek R1, Anthropic Claude, and more. Switch between providers instantly without changing code!
Show Image
Show Image
Show Image
✨ Features

🔄 Multi-AI Provider Support - OpenAI, DeepSeek, Claude, Groq, local models
💬 Context-Aware Conversations - Maintains chat history for natural dialogue
🚫 Smart Response Filtering - Prevents repetitive responses using similarity detection
⚡ Asynchronous Processing - Handles multiple conversations simultaneously
🔧 Highly Configurable - Customize via environment variables
💰 Cost Optimization - Choose cheaper providers like DeepSeek or free local models
🛡️ Robust Error Handling - Graceful fallbacks and detailed error messages

🚀 Quick Start
Prerequisites

Python 3.8+
Discord Bot Token (Get one here)
AI API Key (choose your provider)

Installation

Clone the repository
bashgit clone https://github.com/yourusername/discord-multi-ai-chatbot.git
cd discord-multi-ai-chatbot

Install dependencies
bashpip install discord.py python-dotenv requests

Create environment file
bashcp .env.example .env

Configure your .env file
env# Required
DISCORD_PASS_KEY=your_discord_bot_token

# Choose your AI provider
AI_PROVIDER=deepseek  # Options: openai, deepseek, anthropic, together, groq, local

# Add your API key (only the one you're using)
DEEPSEEK_API_KEY=your_deepseek_api_key

Run the bot
bashpython main.py


🔧 Supported AI Providers
ProviderModelsCostSpeedNotesOpenAIGPT-4, GPT-3.5$$$FastIndustry standardDeepSeekDeepSeek R1$FastMost cost-effectiveAnthropicClaude Haiku/Sonnet$$FastGreat reasoningTogether AILlama, Mixtral$MediumOpen source modelsGroqLlama, MixtralFree tierUltra-fastBest for speedLocalOllama, LM StudioFreeVariesComplete privacy
⚙️ Configuration
Environment Variables
env# Required Settings
DISCORD_PASS_KEY=your_discord_bot_token
AI_PROVIDER=deepseek

# API Keys (add only what you need)
OPENAI_API_KEY=sk-...
DEEPSEEK_API_KEY=your_key
ANTHROPIC_API_KEY=your_key
TOGETHER_API_KEY=your_key
GROQ_API_KEY=your_key

# Optional Customization
MAX_CHAT_HISTORY=10          # Number of messages to remember
SIMILARITY_THRESHOLD=0.7     # Prevent repetitive responses (0.0-1.0)
AI_TEMPERATURE=0.7           # Response creativity (0.0-2.0)
AI_MAX_TOKENS=256           # Maximum response length
Provider-Specific Setup
<details>
<summary><b>🔵 OpenAI Setup</b></summary>
envAI_PROVIDER=openai
OPENAI_API_KEY=sk-your_openai_api_key
Get your API key: OpenAI Platform
</details>
<details>
<summary><b>🟢 DeepSeek Setup (Recommended - Cheapest)</b></summary>
envAI_PROVIDER=deepseek
DEEPSEEK_API_KEY=your_deepseek_api_key
Get your API key: DeepSeek Platform
</details>
<details>
<summary><b>🟠 Anthropic Claude Setup</b></summary>
envAI_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-your_key
Get your API key: Anthropic Console
</details>
<details>
<summary><b>🟡 Groq Setup (Fastest)</b></summary>
envAI_PROVIDER=groq
GROQ_API_KEY=your_groq_api_key
Get your API key: Groq Console
</details>
<details>
<summary><b>🔴 Local Setup (Free)</b></summary>
envAI_PROVIDER=local
# No API key needed
Install Ollama and run:
bashollama serve
ollama pull llama3.2
</details>
🎮 Usage

Invite the bot to your server

Go to Discord Developer Portal
Select your bot → OAuth2 → URL Generator
Select bot scope and required permissions
Use the generated URL to invite your bot


Start chatting
@YourBot Hello! How are you today?


The bot will respond naturally and maintain conversation context!
🏗️ Project Structure
discord-multi-ai-chatbot/
├── main.py              # Main bot code
├── keep_alive.py        # For 24/7 hosting (optional)
├── requirements.txt     # Python dependencies
├── .env.example        # Environment template
├── .env               # Your configuration (create this)
├── README.md          # This file
└── LICENSE           # MIT License
🌐 24/7 Hosting
Option 1: Replit (Free)

Create a new Python Repl on Replit
Upload your project files
Add environment variables in the Secrets tab
Add keep_alive.py for continuous running:

pythonfrom flask import Flask
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

Modify main.py to include:

pythonfrom keep_alive import keep_alive
keep_alive()
# ... rest of your code

Use UptimeRobot to ping your Repl every 5 minutes

Option 2: Railway/Render (Paid)
Deploy to Railway or Render for more reliable hosting.
🎨 Customization
Bot Personality
Edit the system message in main.py:
python{
    "role": "system", 
    "content": "You are MyBot, a helpful assistant who loves gaming and memes..."
}
Response Parameters
Adjust AI behavior:
envAI_TEMPERATURE=0.9      # More creative (0.0 = deterministic, 2.0 = very random)
AI_MAX_TOKENS=512      # Longer responses
SIMILARITY_THRESHOLD=0.5  # Allow more similar responses
Advanced Features

Add command support
Implement role-based permissions
Add multi-language support
Create custom conversation modes

🔍 Troubleshooting
Common Issues
Bot not responding?

Check if bot has Read Messages and Send Messages permissions
Ensure you're mentioning the bot (@YourBot)
Verify your API key is valid

API errors?

Check your API key and billing status
Try switching to a different provider
Check the console for detailed error messages

Rate limiting?

DeepSeek/Groq have generous free tiers
Consider using local models for unlimited usage

Debug Mode
Enable detailed logging by adding to your .env:
envDEBUG=true
📊 Cost Comparison
ProviderCost per 1M tokensFree tierBest forDeepSeek~$0.14YesBudget-consciousGroqFreeGenerousSpeed & free usageOpenAI~$3.00$5 creditPremium qualityClaude~$1.25LimitedReasoning tasksLocal$0UnlimitedPrivacy & control
🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

Fork the repository
Create your feature branch (git checkout -b feature/AmazingFeature)
Commit your changes (git commit -m 'Add some AmazingFeature')
Push to the branch (git push origin feature/AmazingFeature)
Open a Pull Request

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
⭐ Show Your Support
If this project helped you, please give it a ⭐ star on GitHub!
📞 Support

📧 Create an issue for bug reports
💬 Discussions for questions and ideas
📖 Check the wiki for detailed guides


Made with ❤️ for the Discord community
Remember to keep your API keys secure and never commit them to public repositories!
