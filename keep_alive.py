from flask import Flask
from threading import Thread
import logging

# Suppress Flask's default logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app = Flask('')

@app.route('/')
def home():
    return """
    <html>
        <head>
            <title>Discord Bot Status</title>
            <style>
                body { font-family: Arial, sans-serif; text-align: center; margin-top: 50px; }
                .status { color: #00ff00; font-size: 24px; }
            </style>
        </head>
        <body>
            <h1>🤖 Discord Multi-AI ChatBot</h1>
            <p class="status">✅ Bot is alive and running!</p>
            <p>This endpoint is used to keep the bot active 24/7.</p>
        </body>
    </html>
    """

@app.route('/health')
def health():
    return {"status": "healthy", "message": "Bot is running"}

def run():
    app.run(host='0.0.0.0', port=8080, debug=False)

def keep_alive():
    """Start the Flask server in a separate thread"""
    server_thread = Thread(target=run)
    server_thread.daemon = True
    server_thread.start()
    print("🌐 Keep-alive server started on port 8080")

if __name__ == "__main__":
    run()