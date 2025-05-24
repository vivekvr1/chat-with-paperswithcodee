#!/usr/bin/env python3
"""
Test script to verify Google Cloud authentication for Vertex AI
"""

import os
from dotenv import load_dotenv

def test_vertex_ai_auth():
    load_dotenv()
    
    print("🧪 Testing Vertex AI Authentication...")
    print(f"📁 Project: {os.environ.get('GOOGLE_CLOUD_PROJECT', 'Not set')}")
    print(f"🔑 Credentials: {os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', 'Using default')}")
    
    try:
        from langchain_community.embeddings import VertexAIEmbeddings
        
        # Test embedding creation
        embeddings = VertexAIEmbeddings(model_name="textembedding-gecko@003")
        print("✅ VertexAI embeddings initialized successfully!")
        
        # Test actual embedding
        test_text = "This is a test for Vertex AI embeddings"
        result = embeddings.embed_query(test_text)
        
        print(f"✅ Embedding test successful! Vector dimension: {len(result)}")
        print("🎉 Authentication is working correctly!")
        
        return True
        
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        print("\n🔧 Troubleshooting steps:")
        print("1. Make sure you've enabled Vertex AI API:")
        print("   gcloud services enable aiplatform.googleapis.com")
        print("2. Check your credentials are set correctly")
        print("3. Verify your project ID is correct")
        
        return False

def test_basic_auth():
    """Test basic Google Cloud authentication"""
    try:
        from google.auth import default
        credentials, project = default()
        print(f"✅ Default credentials found for project: {project}")
        return True
    except Exception as e:
        print(f"❌ No default credentials found: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Google Cloud Authentication Test")
    print("=" * 50)
    
    if test_basic_auth():
        test_vertex_ai_auth()
    else:
        print("\n💡 Run one of these commands first:")
        print("   gcloud auth application-default login")
        print("   OR set GOOGLE_APPLICATION_CREDENTIALS environment variable")