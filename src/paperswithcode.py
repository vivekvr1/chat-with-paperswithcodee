import urllib.parse
import requests
from tqdm import tqdm
import time


def extract_papers(query: str, max_results: int = 50, max_retries: int = 3):
    query = urllib.parse.quote(query)
    url = f"https://paperswithcode.com/api/v1/papers/?q={query}"
    
    # Try with retries in case of temporary server issues
    for attempt in range(max_retries):
        try:
            print(f"Attempt {attempt + 1}/{max_retries}: Fetching initial page...")
            response = requests.get(url, timeout=30)
            
            # Check for specific status codes
            if response.status_code == 500:
                print(f"Server error (500) - Papers with Code API may be down")
                if attempt < max_retries - 1:
                    print(f"Retrying in {2 ** attempt} seconds...")
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                else:
                    print("Max retries reached. Try again later or use an alternative API.")
                    return []
            elif response.status_code == 429:
                print("Rate limited. Waiting before retry...")
                time.sleep(5)
                continue
                
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
    
    count = response_data["count"]
    results = []
    results += response_data["results"]

    # If we already have enough results from the first page, return early
    if len(results) >= max_results:
        return results[:max_results][:max_results]

    # Calculate how many more pages we actually need
    pages_needed = ((max_results - len(results)) + 9) // 10  # Ceiling division
    max_page = min(2 + pages_needed, (count + 9) // 10)  # Don't exceed total pages
    
    print(f"Found {count} total papers. Fetching {max_results} papers from {max_page - 1} pages...")
    
    for page in tqdm(range(2, max_page + 1), desc="Fetching pages"):
        url = f"https://paperswithcode.com/api/v1/papers/?page={page}&q={query}"
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            page_data = response.json()
            results += page_data["results"]
            
            # Stop if we have enough results
            if len(results) >= max_results:
                break
            
            # Be respectful to the API - add a small delay
            time.sleep(0.1)
            
        except (requests.exceptions.RequestException, ValueError) as e:
            print(f"Error fetching page {page}: {e}")
            continue  # Skip this page and continue with others
    
    return results


if __name__ == "__main__":
    query = "attention mechanism"
    max_papers = 50  # Only fetch 50 papers for fast execution
    
    papers = extract_papers(query, max_results=max_papers)
    print(f"Extracted {len(papers)} papers")
    if papers:
        print(f"First paper title: {papers[0].get('title', 'No title')}")
        print(f"Last paper title: {papers[-1].get('title', 'No title')}")
    else:
        print("No papers found for the query.")