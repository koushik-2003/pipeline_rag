import requests
from bs4 import BeautifulSoup
import re

def scrape_website(url):
    """Fetch and clean content from a website."""
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch {url}. Status code: {response.status_code}")
    
    soup = BeautifulSoup(response.text, "html.parser")
    for script in soup(["script", "style"]):
        script.extract()  # Remove script and style tags
    
    text = soup.get_text()
    text = re.sub(r'\s+', ' ', text).strip()  # Clean text
    return text
