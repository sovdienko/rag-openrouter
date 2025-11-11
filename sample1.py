import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize OpenRouter client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
)

# Generate embeddings for documents
def embed_documents(texts, model="openai/text-embedding-3-large"):
    response = client.embeddings.create(
        model=model,
        input=texts  # Single string or list of strings
    )
    embeddings = [item.embedding for item in response.data]
    return embeddings

# Example usage
documents = [
    "Python is a high-level programming language.",
    "Machine learning enables computers to learn from data.",
    "RAG combines retrieval with generation for accurate responses."
]

doc_embeddings = embed_documents(documents)
print(f"Generated {len(doc_embeddings)} embeddings of dimension {len(doc_embeddings[0])}")