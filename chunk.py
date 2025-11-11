import os
from openai import OpenAI
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

# Load environment variables from .env file
load_dotenv()

# Initialize clients
openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
)

pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

# Create vector index (one-time setup)
index_name = "rag-documents"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=3072,  # For text-embedding-3-large
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)

# Step 1: Document ingestion with chunking
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Usage example
documents = [
    "The Python programming language was created by Guido van Rossum and first released in 1991. It emphasizes code readability with significant whitespace.",
    "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.",
    "Retrieval-Augmented Generation (RAG) combines information retrieval with large language model generation to provide accurate, grounded responses.",
]
"""Split documents and store with embeddings"""

# Split into chunks
splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
)
    
chunks = []
metadata = []
for i, doc in enumerate(documents):
    doc_chunks = splitter.split_text(doc)
    chunks.extend(doc_chunks)
    metadata.extend([{"source": f"doc_{i}", "chunk": j} for j in range(len(doc_chunks))])
    
print(f"Split into {len(chunks)} chunks.")
print(f"Chunks {chunks}")
print(f"Metadata: {metadata} ")
print("###############################")
 # Generate embeddings
response = openrouter_client.embeddings.create(
    model="openai/text-embedding-3-large",
    input=chunks
)
    
embeddings = [item.embedding for item in response.data]
    
# Prepare vectors for Pinecone
vectors = []
for i, (chunk, emb, meta) in enumerate(zip(chunks, embeddings, metadata)):
    vectors.append({
        "id": f"chunk_{i}",
        "values": emb,
        "metadata": {"text": chunk, **meta}
    })

print(f"Vectors:{vectors[0]}")

# Batch upsert to Pinecone
index.upsert(vectors=vectors, namespace="default")
print(f"Stored {len(vectors)} chunks with embeddings")