import os
import sys
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores.upstash import UpstashVectorStore

# Handle imports for both src/ and root directory usage
try:
    from bprompt import RAG_PROMPT_TEMPLATE
    from callbacks import StreamHandler
except ImportError:
    # If running from src/ directory, try parent directory
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        from bprompt import RAG_PROMPT_TEMPLATE
        from callbacks import StreamHandler
    except ImportError:
        # Fallback: define inline
        RAG_PROMPT_TEMPLATE = """You are an AI assistant helping users understand research papers. Use the provided context to answer the question accurately and concisely.

Context from research papers:
{context}

Question: {question}

Instructions:
- Answer based primarily on the provided context
- If the context doesn't contain enough information, say so clearly
- Cite specific papers or findings when relevant
- Be concise but comprehensive

Answer:"""
        
        # Simple fallback StreamHandler
        from langchain.callbacks.base import BaseCallbackHandler
        
        class StreamHandler(BaseCallbackHandler):
            def __init__(self, container):
                self.container = container
                self.text = ""
                
            def on_llm_new_token(self, token: str, **kwargs) -> None:
                self.text += token
                if hasattr(self.container, 'markdown'):
                    self.container.markdown(self.text)
                else:
                    print(token, end="", flush=True)

load_dotenv()


class RAG:
    def __init__(self, chat_box, embeddings):
        self.chat_box = chat_box
        self.embeddings = embeddings
        self.set_llm()
        
        # Use official LangChain UpstashVectorStore (automatically reads env vars)
        self.vectorstore = UpstashVectorStore(embedding=embeddings)

    def set_llm(self):
        """Initialize the language model with appropriate streaming settings"""
        if self.chat_box:
            # Streamlit mode with streaming
            try:
                chat_container = self.chat_box.container().empty()
                stream_handler = StreamHandler(chat_container)
                callbacks = [stream_handler]
                streaming = True
            except:
                # Fallback if Streamlit container setup fails
                callbacks = []
                streaming = False
        else:
            # Terminal mode without streaming
            callbacks = []
            streaming = False
            
        self.llm = ChatOpenAI(
            max_tokens=400,
            streaming=streaming,
            callbacks=callbacks,
            model="gpt-3.5-turbo",
            temperature=0.1,  # Lower temperature for more consistent answers
        )

    def get_context(self, query, k=4):
        """Get relevant context from vector store"""
        try:
            # Use the official similarity_search_with_score method
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            context = ""

            for doc, score in results:
                # Add score and content with separator
                context += f"[Relevance: {score:.3f}]\n{doc.page_content}\n{'='*50}\n"
            
            return context, results
            
        except Exception as e:
            print(f"Error getting context: {e}")
            return "", []

    @staticmethod
    def get_prompt(question, context):
        """Format the RAG prompt with question and context"""
        return RAG_PROMPT_TEMPLATE.format(question=question, context=context)

    def predict(self, query):
        """Main prediction method for RAG"""
        try:
            # Get relevant context from vector store
            context, source_documents = self.get_context(query)
            
            if not context.strip():
                context = "No relevant documents found in the knowledge base."
            
            # Format prompt with context
            prompt = self.get_prompt(query, context)
            
            # Generate answer using LLM
            answer = self.llm.invoke(prompt)
            
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
            return len(results) > 0
        except Exception as e:
            print(f"❌ Vector store connection failed: {e}")
            return False


# Functions for terminal testing
def test_retrieval_only(rag_system, query, k=3):
    """Test only the retrieval part without LLM generation"""
    print(f"🔍 Testing retrieval for: '{query}'")
    print("-" * 50)
    
    context, source_documents = rag_system.get_context(query, k=k)
    
    if not source_documents:
        print("❌ No documents found for this query.")
        return
    
    print(f"✅ Found {len(source_documents)} relevant documents:")
    print()
    
    for i, (doc, score) in enumerate(source_documents, 1):
        print(f"📄 Document {i} (Score: {score:.4f})")
        print(f"Title: {doc.metadata.get('title', 'No title')}")
        print(f"Authors: {doc.metadata.get('authors', 'Unknown')}")
        print(f"Content: {doc.page_content[:200]}...")
        print("-" * 30)


def test_full_rag(rag_system, query):
    """Test the full RAG pipeline (retrieval + generation)"""
    print(f"🤖 Testing full RAG for: '{query}'")
    print("=" * 60)
    
    prediction = rag_system.predict(query)
    
    print("📋 ANSWER:")
    print(prediction["answer"])
    print()
    
    print("📚 SOURCE DOCUMENTS:")
    source_docs = prediction["source_documents"]
    if source_docs:
        for i, (doc, score) in enumerate(source_docs, 1):
            print(f"{i}. {doc.metadata.get('title', 'No title')} (Score: {score:.3f})")
    else:
        print("No source documents found.")
    print()


def interactive_mode(rag_system):
    """Interactive mode for testing RAG system"""
    print("🚀 Interactive RAG Testing Mode")
    print("Commands:")
    print("  - Type your question to get an answer")
    print("  - Type 'search:<query>' to test retrieval only")
    print("  - Type 'quit' to exit")
    print("=" * 60)
    
    while True:
        try:
            user_input = input("\n💭 Your input: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if user_input.startswith('search:'):
                query = user_input[7:].strip()
                test_retrieval_only(rag_system, query)
            elif user_input:
                test_full_rag(rag_system, user_input)
            else:
                print("Please enter a question or command.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def main():
    """Main function to test RAG system in terminal"""
    print("🧪 RAG System Terminal Tester")
    print("=" * 60)
    
    # Check environment variables
    required_vars = ["OPENAI_API_KEY", "UPSTASH_VECTOR_REST_URL", "UPSTASH_VECTOR_REST_TOKEN"]
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    
    if missing_vars:
        print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        print("\nPlease add them to your .env file and run the indexing script first:")
        print("python index_papers.py index --query 'your topic' --max_papers 20")
        return
    
    try:
        # Initialize embeddings
        print("🔧 Initializing OpenAI embeddings...")
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=os.environ.get("OPENAI_API_KEY")
        )
        
        # Initialize RAG system (no chat_box for terminal)
        print("🔧 Initializing RAG system...")
        rag = RAG(chat_box=None, embeddings=embeddings)
        
        # Test connection
        print("🔧 Testing vector store connection...")
        if not rag.test_connection():
            print("❌ Vector store connection failed. Make sure you've indexed some papers first:")
            print("python index_papers.py index --query 'attention mechanism' --max_papers 20")
            return
        
        print("✅ RAG system initialized successfully!")
        print()
        
        # Quick test with sample queries
        sample_queries = [
            "What are attention mechanisms?",
            "How do transformers work?",
            "What is self-attention?"
        ]
        
        print("🧪 Quick test with sample queries:")
        for query in sample_queries:
            print(f"\n🔍 Testing: '{query}'")
            context, results = rag.get_context(query, k=2)
            if results:
                print(f"✅ Found {len(results)} relevant documents")
            else:
                print("❌ No relevant documents found")
        
        print("\n" + "=" * 60)
        
        # Start interactive mode
        interactive_mode(rag)
        
    except Exception as e:
        print(f"❌ Error initializing RAG system: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure your .env file has the correct credentials")
        print("2. Run the indexing script first to populate the vector database")
        print("3. Check that your OpenAI API key is valid")


if __name__ == "__main__":
    main()