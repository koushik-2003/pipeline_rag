from sentence_transformers import SentenceTransformer
from openai import OpenAI
from vector_store import search_similar_vectors

def handle_query(query):
    """Process user query and generate response."""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embedding = model.encode([query])[0]
    
    results = search_similar_vectors(embedding)
    retrieved_chunks = [meta['content'] for meta in results['metadatas'][0]]
    
    prompt = f"Context: {retrieved_chunks}\n\nQuestion: {query}\n\nAnswer:"
    client = OpenAI()
    response = client.Completion.create(
        engine="gpt-3.5-turbo",
        prompt=prompt,
        max_tokens=200
    )
    return response['choices'][0]['text']
