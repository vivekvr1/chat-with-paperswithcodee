import urllib.parse
import requests
from tqdm import tqdm
import time


def extract_papers(query: str, max_results: int = 50, max_retries: int = 3):
    """
    Extract papers using Semantic Scholar API (replacement for Papers with Code API)
    
    Args:
        query: Search query string
        max_results: Maximum number of papers to return
        max_retries: Number of retry attempts for failed requests
    
    Returns:
        List of paper dictionaries compatible with original Papers with Code format
    """
    
    # Semantic Scholar API endpoint
    base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
    
    # Headers for better API behavior
    headers = {
        "User-Agent": "PapersExtractor/1.0 (Academic Research Tool)"
    }
    
    # Parameters for Semantic Scholar API
    params = {
        "query": query,
        "fields": "paperId,title,abstract,authors,year,citationCount,url,openAccessPdf,publicationDate,venue,externalIds",
        "limit": min(100, max_results),  # API max is 100 per request
        "offset": 0
    }
    
    all_papers = []
    
    # Try with retries in case of temporary server issues
    for attempt in range(max_retries):
        try:
            print(f"Attempt {attempt + 1}/{max_retries}: Fetching papers from Semantic Scholar...")
            response = requests.get(base_url, params=params, headers=headers, timeout=30)
            
            # Check for specific status codes
            if response.status_code == 500:
                print(f"Server error (500) - Semantic Scholar API may be down")
                if attempt < max_retries - 1:
                    print(f"Retrying in {2 ** attempt} seconds...")
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                else:
                    print("Max retries reached. Try again later.")
                    return []
            elif response.status_code == 429:
                print("Rate limited. Waiting before retry...")
                time.sleep(5)
                continue
            elif response.status_code == 403:
                print("Access forbidden. You may need an API key for higher limits.")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                else:
                    return []
                
            response.raise_for_status()
            response_data = response.json()
            break  # Success, exit retry loop
            
        except (requests.exceptions.RequestException, ValueError) as e:
            print(f"Error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {2 ** attempt} seconds...")
                time.sleep(2 ** attempt)
            else:
                print("Max retries reached. The API may be temporarily unavailable.")
                return []
    
    # Extract papers from first response
    papers = response_data.get("data", [])
    all_papers.extend(papers)
    
    print(f"Found {len(papers)} papers in first batch")
    
    # Continue fetching if we need more papers and there are more available
    while len(all_papers) < max_results and len(papers) == 100:  # 100 means there might be more
        params["offset"] += 100
        params["limit"] = min(100, max_results - len(all_papers))
        
        print(f"Fetching more papers (total so far: {len(all_papers)})...")
        
        try:
            response = requests.get(base_url, params=params, headers=headers, timeout=30)
            response.raise_for_status()
            response_data = response.json()
            
            papers = response_data.get("data", [])
            all_papers.extend(papers)
            
            if not papers:  # No more papers available
                break
                
            # Be respectful to the API
            time.sleep(0.5)
            
        except (requests.exceptions.RequestException, ValueError) as e:
            print(f"Error fetching additional papers: {e}")
            break  # Continue with what we have
    
    # Convert Semantic Scholar format to Papers with Code compatible format
    results = []
    
    for paper in all_papers[:max_results]:
        # Extract arXiv ID if available
        arxiv_id = ""
        external_ids = paper.get("externalIds", {}) or {}
        if external_ids and "ArXiv" in external_ids:
            arxiv_id = external_ids["ArXiv"]
        
        # Extract PDF URL
        pdf_url = ""
        open_access_pdf = paper.get("openAccessPdf")
        if open_access_pdf and open_access_pdf.get("url"):
            pdf_url = open_access_pdf["url"]
        
        # Format authors
        authors_list = []
        authors = paper.get("authors", []) or []
        for author in authors:
            if isinstance(author, dict) and author.get("name"):
                authors_list.append(author["name"])
            elif isinstance(author, str):
                authors_list.append(author)
        
        # Convert to Papers with Code compatible format
        converted_paper = {
            "id": paper.get("paperId", ""),
            "arxiv_id": arxiv_id,
            "url_pdf": pdf_url,
            "title": paper.get("title", ""),
            "abstract": paper.get("abstract", ""),
            "authors": authors_list,
            "published": paper.get("publicationDate", "") or str(paper.get("year", "")),
            "url": paper.get("url", ""),
            "paper_url": paper.get("url", ""),  # Same as url for Semantic Scholar
            "citation_count": paper.get("citationCount", 0),
            "venue": paper.get("venue", ""),
            "year": paper.get("year", ""),
        }
        
        results.append(converted_paper)
    
    print(f"Successfully converted {len(results)} papers to compatible format")
    return results


def test_semantic_scholar_api():
    """Test function to verify Semantic Scholar API is working"""
    print("🧪 Testing Semantic Scholar API...")
    
    test_url = "https://api.semanticscholar.org/graph/v1/paper/search"
    test_params = {
        "query": "machine learning",
        "fields": "title,abstract,authors",
        "limit": 3
    }
    
    try:
        response = requests.get(test_url, params=test_params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        papers = data.get("data", [])
        if papers:
            print(f"✅ API working! Found {len(papers)} test papers")
            print(f"First paper: {papers[0].get('title', 'No title')}")
            return True
        else:
            print("⚠️ API responded but no papers found")
            return False
            
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False


if __name__ == "__main__":
    # Test the API first
    if test_semantic_scholar_api():
        print("\n" + "="*50)
        
        # Run the main extraction
        query = "attention mechanism"
        max_papers = 20  # Start with fewer papers for testing
        
        papers = extract_papers(query, max_results=max_papers)
        print(f"\nExtracted {len(papers)} papers")
        
        if papers:
            print(f"First paper title: {papers[0].get('title', 'No title')}")
            print(f"First paper abstract preview: {papers[0].get('abstract', 'No abstract')[:100]}...")
            print(f"Authors: {papers[0].get('authors', [])}")
            print(f"Year: {papers[0].get('year', 'Unknown')}")
            print(f"Citations: {papers[0].get('citation_count', 0)}")
            
            if len(papers) > 1:
                print(f"\nLast paper title: {papers[-1].get('title', 'No title')}")
        else:
            print("No papers found for the query.")
    else:
        print("❌ Semantic Scholar API test failed. Please check your internet connection.")