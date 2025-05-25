"""
Callback handlers for streaming responses in Streamlit
"""

from typing import Any, Dict, List
from langchain.callbacks.base import BaseCallbackHandler
import streamlit as st


class StreamHandler(BaseCallbackHandler):
    """
    Callback handler for streaming LLM responses to Streamlit
    """
    
    def __init__(self, container):
        self.container = container
        self.text = ""
        
    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Called when a new token is generated"""
        self.text += token
        self.container.markdown(self.text)
        
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        """Called when LLM starts generating"""
        self.text = ""
        
    def on_llm_end(self, response, **kwargs: Any) -> None:
        """Called when LLM finishes generating"""
        pass
        
    def on_llm_error(self, error: Exception, **kwargs: Any) -> None:
        """Called when LLM encounters an error"""
        self.container.error(f"Error: {str(error)}")


class SimpleStreamHandler(BaseCallbackHandler):
    """
    Simple callback handler that just prints tokens (for terminal use)
    """
    
    def __init__(self):
        self.text = ""
        
    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Called when a new token is generated"""
        print(token, end="", flush=True)
        
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        """Called when LLM starts generating"""
        pass
        
    def on_llm_end(self, response, **kwargs: Any) -> None:
        """Called when LLM finishes generating"""
        print()  # New line at the end
        
    def on_llm_error(self, error: Exception, **kwargs: Any) -> None:
        """Called when LLM encounters an error"""
        print(f"Error: {str(error)}")