import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores.upstash import UpstashVectorStore  # Official LangChain
from prompts import RAG_PROMPT_TEMPLATE
from src.callbacks import StreamHandler


load_dotenv()


class RAG:
    def __init__(self, chat_box, embeddings):
        self.chat_box = chat_box
        self.set_llm()
        self.embeddings = embeddings
        
        # Use official LangChain UpstashVectorStore (automatically reads env vars)
        self.vectorstore = UpstashVectorStore(embedding=embeddings)

    def set_llm(self):
        chat_box = self.chat_box.container().empty()
        stream_handler = StreamHandler(chat_box)
        llm = ChatOpenAI(
            max_tokens=400,
            streaming=True,
            callbacks=[stream_handler],
            model="gpt-3.5-turbo",
        )
        self.llm = llm

    def get_context(self, query, k=4):
        """Get relevant context from vector store"""
        try:
            # Use the official similarity_search_with_score method
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            context = ""

            for doc, score in results:
                context += f"[Score: {score:.3f}] {doc.page_content}\n===\n"
            
            return context, results
            
        except Exception as e:
            print(f"Error getting context: {e}")
            return "", []

    @staticmethod
    def get_prompt(question, context):
        prompt = RAG_PROMPT_TEMPLATE.format(question=question, context=context)
        return prompt

    def predict(self, query):
        try:
            context, source_documents = self.get_context(query)
            
            if not context.strip():
                context = "No relevant documents found in the knowledge base."
            
            prompt = self.get_prompt(query, context)
            answer = self.llm.predict(prompt)
            
            prediction = {
                "answer": answer,
                "source_documents": source_documents,
                "context_used": context,
            }
            return prediction
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            return {
                "answer": f"Sorry, I encountered an error: {str(e)}",
                "source_documents": [],
                "context_used": "",
            }

    def test_connection(self):
        """Test if the vector store connection is working"""
        try:
            results = self.vectorstore.similarity_search("test", k=1)
            print(f"✅ Vector store connection working. Found {len(results)} documents.")
            return True
        except Exception as e:
            print(f"❌ Vector store connection failed: {e}")
            return False