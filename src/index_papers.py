import os
import click
from dotenv import load_dotenv

print(f"Current working directory: {os.getcwd()}")
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings  # Changed from VertexAI
from langchain_community.vectorstores.upstash import UpstashVectorStore  # Import from community
from paperswithcode import extract_papers


@click.command()
@click.option("--query", type=str, required=True, help="Search query for papers")
@click.option("--batch_size", type=int, default=32, help="Batch size for indexing")
@click.option("--max_papers", type=int, default=50, help="Maximum number of papers to extract")
@click.option("--max_chunks", type=int, default=None, help="Maximum number of text chunks to index")
@click.option("--embedding_model", type=str, default="text-embedding-3-small", help="OpenAI embedding model")
def cli(query, batch_size, max_papers, max_chunks, embedding_model):
    load_dotenv()
    
    # Check required environment variables
    required_vars = ["UPSTASH_VECTOR_REST_URL", "UPSTASH_VECTOR_REST_TOKEN", "OPENAI_API_KEY"]
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    
    if missing_vars:
        click.echo(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        click.echo("\nPlease add them to your .env file:")
        for var in missing_vars:
            if var == "OPENAI_API_KEY":
                click.echo(f"  {var}=sk-your-openai-api-key-here")
            elif var == "UPSTASH_VECTOR_REST_URL":
                click.echo(f"  {var}=https://your-upstash-url.upstash.io")
            elif var == "UPSTASH_VECTOR_REST_TOKEN":
                click.echo(f"  {var}=your-upstash-token-here")
        return
    
    click.echo(f"Extracting papers matching query: '{query}'")
    click.echo(f"Maximum papers to fetch: {max_papers}")
    click.echo(f"Using OpenAI embedding model: {embedding_model}")
    
    # Extract papers with the specified limit
    papers = extract_papers(query, max_results=max_papers)
    click.echo(f"Extraction complete ✅: ({len(papers)} papers)")
    
    if not papers:
        click.echo("❌ No papers found. Try a different query.")
        return
    
    # Filter papers that have abstracts (required for indexing)
    papers_with_abstracts = [
        paper for paper in papers 
        if paper.get("abstract") and paper.get("abstract").strip()
    ]
    
    click.echo(f"Papers with abstracts: {len(papers_with_abstracts)}")
    
    if not papers_with_abstracts:
        click.echo("❌ No papers with abstracts found. Cannot proceed with indexing.")
        return
    
    # Create documents from papers
    documents = []
    for paper in papers_with_abstracts:
        # Handle missing fields gracefully
        doc = Document(
            page_content=paper.get("abstract", ""),
            metadata={
                "id": paper.get("id", ""),
                "arxiv_id": paper.get("arxiv_id", ""),
                "url_pdf": paper.get("url_pdf", ""),
                "title": paper.get("title", ""),
                "authors": paper.get("authors", []),
                "published": paper.get("published", ""),
                "url": paper.get("url", ""),
                "paper_url": paper.get("paper_url", ""),
            },
        )
        documents.append(doc)

    click.echo(f"Created {len(documents)} documents")

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", ".", " "],
    )
    
    click.echo("Splitting documents into chunks...")
    splits = text_splitter.split_documents(documents)
    click.echo(f"Created {len(splits)} text chunks")
    
    # Apply chunk limit if specified
    if max_chunks and max_chunks < len(splits):
        splits = splits[:max_chunks]
        click.echo(f"Limited to {len(splits)} chunks")

    if not splits:
        click.echo("❌ No text chunks created. Cannot proceed with indexing.")
        return

    # Initialize Upstash vector store with OpenAI embeddings
    try:
        # Use OpenAI embeddings
        embeddings = OpenAIEmbeddings(
            model=embedding_model,
            openai_api_key=os.environ.get("OPENAI_API_KEY")
        )
        
        # Create UpstashVectorStore with proper named parameters
        # Option 1: Using environment variables (recommended)
        upstash_vector_store = UpstashVectorStore(embedding=embeddings)
        
        # Option 2: Using explicit index (alternative)
        # index = Index(
        #     url=os.environ.get("UPSTASH_URL"),
        #     token=os.environ.get("UPSTASH_TOKEN"),
        # )
        # upstash_vector_store = UpstashVectorStore(embedding=embeddings, index=index)
        
        click.echo(f"Indexing {len(splits)} chunks to Upstash...")
        ids = upstash_vector_store.add_documents(splits, batch_size=batch_size)
        click.echo(f"✅ Successfully indexed {len(ids)} vectors to Upstash")
        
    except Exception as e:
        click.echo(f"❌ Error during indexing: {e}")
        raise


@click.command()
@click.option("--query", type=str, required=True)
@click.option("--max_papers", type=int, default=5)
def test_extraction(query, max_papers):
    """Test paper extraction without indexing"""
    load_dotenv()
    
    click.echo(f"Testing extraction for query: '{query}'")
    papers = extract_papers(query, max_results=max_papers)
    
    click.echo(f"Found {len(papers)} papers:")
    for i, paper in enumerate(papers, 1):
        title = paper.get("title", "No title")[:80]
        abstract_preview = paper.get("abstract", "No abstract")[:100]
        click.echo(f"{i}. {title}")
        click.echo(f"   Abstract: {abstract_preview}...")
        click.echo(f"   Authors: {paper.get('authors', 'Unknown')}")
        click.echo("")


@click.command()
def test_upstash():
    """Test Upstash vector store connection"""
    load_dotenv()
    
    required_vars = ["UPSTASH_VECTOR_REST_URL", "UPSTASH_VECTOR_REST_TOKEN", "OPENAI_API_KEY"]
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    
    if missing_vars:
        click.echo(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        return
    
    try:
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=os.environ.get("OPENAI_API_KEY")
        )
        
        # Test UpstashVectorStore initialization
        upstash_vector_store = UpstashVectorStore(embedding=embeddings)
        
        click.echo("✅ UpstashVectorStore initialized successfully!")
        
        # Test adding a simple document
        from langchain.docstore.document import Document
        test_doc = Document(
            page_content="This is a test document for Upstash vector store",
            metadata={"test": True}
        )
        
        ids = upstash_vector_store.add_documents([test_doc])
        click.echo(f"✅ Test document added with ID: {ids[0]}")
        
        # Test similarity search
        results = upstash_vector_store.similarity_search("test document", k=1)
        if results:
            click.echo(f"✅ Similarity search working! Found: '{results[0].page_content[:50]}...'")
        
    except Exception as e:
        click.echo(f"❌ Upstash test failed: {e}")
        if "API key" in str(e).lower():
            click.echo("💡 Check if your OpenAI API key is correct")
        elif "upstash" in str(e).lower():
            click.echo("💡 Check if your Upstash credentials are correct")


@click.group()
def main():
    """Papers with Code indexing tool with OpenAI embeddings and Upstash Vector"""
    pass


main.add_command(cli, name="index")
main.add_command(test_extraction, name="test")
main.add_command(test_upstash, name="test-upstash")


if __name__ == "__main__":
    main()