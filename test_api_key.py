# Create a quick test script: test_api_key.py
from anthropic import Anthropic
import os
from dotenv import load_dotenv

load_dotenv()

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Test with the simplest possible call
try:
    message = client.messages.create(
        model="claude-3-haiku-20240307",  # Cheapest model to test
        max_tokens=50,
        messages=[
            {"role": "user", "content": "Say hello!"}
        ]
    )
    
    print("✅ API Key is working!")
    print(f"Response: {message.content[0].text}")
    print(f"\nAvailable model tested: claude-3-haiku-20240307")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("\nTroubleshooting:")
    print("1. Check your API key is correct in .env")
    print("2. Verify you have credits: https://console.anthropic.com/settings/billing")
    print("3. Try regenerating your API key")