import os
from dotenv import load_dotenv
import streamlit as st
from langchain_openai import OpenAIEmbeddings  # Changed from VertexAI to OpenAI
from src.rag import RAG

st.set_page_config(
    page_title="Chat with Papers",
    page_icon="📚", 
    layout="wide"
)
load_dotenv()

# Check required environment variables
required_vars = ["OPENAI_API_KEY", "UPSTASH_VECTOR_REST_URL", "UPSTASH_VECTOR_REST_TOKEN"]
missing_vars = [var for var in required_vars if not os.environ.get(var)]

if missing_vars:
    st.error(f"❌ Missing environment variables: {', '.join(missing_vars)}")
    st.info("Please add them to your .env file:")
    for var in missing_vars:
        if var == "OPENAI_API_KEY":
            st.code(f"{var}=sk-your-openai-api-key-here")
        else:
            st.code(f"{var}=your-{var.lower().replace('_', '-')}-value")
    st.stop()

@st.cache_resource
def get_embedding_model():
    """Get OpenAI embeddings model (cached for performance)"""
    try:
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",  # Changed from VertexAI model
            openai_api_key=os.environ.get("OPENAI_API_KEY")
        )
        return embeddings
    except Exception as e:
        st.error(f"Failed to initialize OpenAI embeddings: {e}")
        st.stop()

@st.cache_resource
def load_rag(_chat_box):
    """Load RAG system with OpenAI embeddings (cached for performance)"""
    embeddings = get_embedding_model()
    try:
        rag = RAG(_chat_box, embeddings)
        
        # Test the connection on startup
        if rag.test_connection():
            st.success("✅ Connected to vector database successfully!")
        else:
            st.warning("⚠️ Vector database connection test failed. You may need to index some papers first.")
        
        return rag
    except Exception as e:
        st.error(f"Failed to initialize RAG system: {e}")
        st.stop()

def display_source_documents(source_documents):
    """Display source documents with metadata"""
    if not source_documents:
        st.info("No source documents found.")
        return
        
    for i, (document, score) in enumerate(source_documents):
        metadata = document.metadata
        document_content = document.page_content

        # Extract metadata with fallbacks
        id_ = metadata.get("id", "Unknown")
        arxiv_id = metadata.get("arxiv_id", "N/A")
        url_pdf = metadata.get("url_pdf", "")
        title = metadata.get("title", "No title")
        authors = metadata.get("authors", [])
        published = metadata.get("published", "Unknown")

        with st.container(border=True):
            st.markdown(f"**📰 Document {i+1}**: {title}")
            st.markdown(f"**🎯 Relevance Score**: {score:.3f}")
            
            if arxiv_id and arxiv_id != "N/A":
                st.markdown(f"**🏷️ ArXiv ID**: `{arxiv_id}`")
            
            if authors:
                if isinstance(authors, list):
                    authors_str = ", ".join(authors)
                else:
                    authors_str = str(authors)
                st.markdown(f"**✍️ Authors**: {authors_str}")
            
            if published and published != "Unknown":
                st.markdown(f"**📅 Published**: {published}")
            
            if url_pdf:
                st.markdown(f"**🔗 PDF**: [Download Link]({url_pdf})")
            
            # Show content preview
            with st.expander(f"📄 Content Preview", expanded=False):
                st.write(document_content[:500] + "..." if len(document_content) > 500 else document_content)

# Main UI
st.title("📚 Chat with Research Papers")
st.markdown("Ask questions about research papers in your knowledge base!")

# Initialize RAG system
with st.spinner("Initializing RAG system..."):
    columns = st.columns(2)
    
    with columns[0]:
        chat_box = st.empty()
    
    rag = load_rag(chat_box)

# Input section
st.markdown("---")
input_question = st.text_input(
    "💭 Ask your question:", 
    placeholder="e.g., What are the main contributions of attention mechanisms in transformers?"
)

# Add some example questions
with st.expander("💡 Example Questions"):
    st.markdown("""
    - What are the main attention mechanisms used in transformers?
    - How do self-attention and cross-attention differ?
    - What are the computational complexities of different attention methods?
    - Which papers discuss multi-head attention?
    - What are the recent improvements to the transformer architecture?
    """)

# Process question
if input_question.strip() != "":
    with st.spinner("🔍 Searching knowledge base and generating answer..."):
        try:
            prediction = rag.predict(input_question)
            
            answer = prediction["answer"]
            source_documents = prediction["source_documents"]
            
            # Display results
            with columns[0]:
                st.markdown("### 🤖 Answer")
                st.markdown(answer)
                
                # Show context info
                if source_documents:
                    st.info(f"📊 Found {len(source_documents)} relevant source documents")
                else:
                    st.warning("No relevant documents found in the knowledge base")
            
            with columns[1]:
                st.markdown("### 📚 Source Documents")
                display_source_documents(source_documents)
                
        except Exception as e:
            st.error(f"❌ Error generating answer: {e}")
            st.info("Please check your configuration and try again.")

# Sidebar with info
with st.sidebar:
    st.markdown("### 📊 System Status")
    
    # Show environment status
    if os.environ.get("OPENAI_API_KEY"):
        st.success("✅ OpenAI API Key configured")
    else:
        st.error("❌ OpenAI API Key missing")
    
    if os.environ.get("UPSTASH_VECTOR_REST_URL") and os.environ.get("UPSTASH_VECTOR_REST_TOKEN"):
        st.success("✅ Upstash Vector DB configured")
    else:
        st.error("❌ Upstash Vector DB not configured")
    
    st.markdown("---")
    st.markdown("### 🔧 Setup Instructions")
    st.markdown("""
    1. **Index Papers**: Run the indexing script first
       ```bash
       python src/index_papers.py index --query "your topic" --max_papers 50
       ```
    
    2. **Required Environment Variables**:
       - `OPENAI_API_KEY`
       - `UPSTASH_VECTOR_REST_URL`
       - `UPSTASH_VECTOR_REST_TOKEN`
    
    3. **Start Chatting**: Ask questions about your indexed papers!
    """)