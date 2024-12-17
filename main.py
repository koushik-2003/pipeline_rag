from scraper import scrape_website
from embeddings import chunk_text, get_embeddings
from vector_store import store_embeddings
from query_handler import handle_query

def main():
    # Step 1: Data Ingestion
    url = input("Enter the website URL to scrape: ")
    content = scrape_website(url)
    print("Content scraped successfully.")
    
    # Step 2: Chunking and Embedding
    chunks = chunk_text(content)
    embeddings = get_embeddings(chunks)
    store_embeddings(chunks, embeddings)
    
    # Step 3: Query Handling
    print("\n--- Query the RAG Pipeline ---")
    while True:
        query = input("Ask a question (or type 'exit' to quit): ")
        if query.lower() == "exit":
            break
        response = handle_query(query)
        print(f"Response: {response}")

if __name__ == "__main__":
    main()
