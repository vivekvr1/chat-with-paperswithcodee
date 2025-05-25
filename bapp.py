import os
import sys
from dotenv import load_dotenv
import streamlit as st
from langchain_openai import OpenAIEmbeddings

# Setup
load_dotenv()
sys.path.append('src')

try:
    from src.brag import RAG
except:
    st.error("Can't import RAG. Check your files.")
    st.stop()

# Initialize once
if 'rag' not in st.session_state:
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        st.session_state.rag = RAG(None, embeddings)
    except Exception as e:
        st.error(f"Setup failed: {e}")
        st.stop()

# UI
st.title("💬 Ask Your Papers")

question = st.text_input("Question:")

if question:
    result = st.session_state.rag.predict(question)
    
    st.write("**Answer:**")
    st.write(result["answer"])
    
    if result["source_documents"]:
        st.write("**Sources:**")
        for i, (doc, score) in enumerate(result["source_documents"]):
            st.write(f"{i+1}. {doc.metadata.get('title', 'Unknown')} ({score:.2f})")