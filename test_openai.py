#!/usr/bin/env python3
"""
Simple script to test OpenAI API key in Docker environment
"""
import os
from dotenv import load_dotenv

print("🧪 Testing OpenAI API Key in Docker")
print("=" * 40)

# Load environment
load_dotenv()

# Get API key
api_key = os.environ.get("OPENAI_API_KEY", "").strip()

if not api_key:
    print("❌ OPENAI_API_KEY not found in environment")
    print("Available env vars:", [k for k in os.environ.keys() if 'OPENAI' in k.upper()])
    exit(1)

print(f"✅ API Key found: {api_key[:10]}...{api_key[-4:]}")
print(f"   Length: {len(api_key)} characters")

# Basic format check
if not api_key.startswith('sk-'):
    print("❌ API key should start with 'sk-'")
    exit(1)

print("✅ API key format looks correct")

# Test with OpenAI
try:
    from openai import OpenAI
    
    client = OpenAI(api_key=api_key)
    
    print("🔄 Testing API connection...")
    
    # Simple test call
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello! Just testing the connection."}],
        max_tokens=10
    )
    
    print("✅ OpenAI API connection successful!")
    print(f"Response: {response.choices[0].message.content}")
    
except ImportError:
    print("⚠️ OpenAI package not installed, but API key format is valid")
except Exception as e:
    print(f"❌ OpenAI API test failed: {e}")
    if "401" in str(e):
        print("   This means the API key is invalid or expired")
    elif "quota" in str(e).lower():
        print("   This means you've exceeded your API quota")
    exit(1)

print("🎉 All tests passed!")