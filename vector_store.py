import chromadb

def store_embeddings(chunks, embeddings, metadatas=None):
    """Store embeddings into a vector database."""
    client = chromadb.Client()
    collection = client.get_or_create_collection("rag_pipeline")

    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        collection.add(
            ids=[str(i)],
            embeddings=[embedding.tolist()],
            metadatas=[{"content": chunk}] if metadatas else [{}]
        )
    print("Embeddings stored successfully.")

def search_similar_vectors(query_embedding, top_k=3):
    """Search for similar vectors in the database."""
    client = chromadb.Client()
    collection = client.get_collection("rag_pipeline")
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k
    )
    return results
