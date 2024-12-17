from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

def chunk_text(text, chunk_size=500, chunk_overlap=100):
    """Split text into manageable chunks."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

def get_embeddings(chunks):
    """Generate embeddings for text chunks using a pre-trained model."""
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Small and efficient embedding model
    embeddings = model.encode(chunks, show_progress_bar=True)
    return embeddings
