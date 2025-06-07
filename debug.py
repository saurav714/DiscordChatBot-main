from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Check if the key exists
api_key = os.getenv('sk-e6bd13d062464369a87d29f05450079f')
if api_key:
    print(f"Key found, length: {len(api_key)}")
    print(f"Starts with: {api_key[:10]}...")
else:
    print("No API key found in environment variables")

# List all environment variables to debug
print("Available env vars:", list(os.environ.keys()))