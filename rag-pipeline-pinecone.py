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
def chunk_and_store_documents(documents, chunk_size=1000, overlap=200):
    """Split documents and store with embeddings"""
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    
    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = []
    metadata = []
    for i, doc in enumerate(documents):
        doc_chunks = splitter.split_text(doc)
        chunks.extend(doc_chunks)
        metadata.extend([{"source": f"doc_{i}", "chunk": j} for j in range(len(doc_chunks))])
    
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
    
    # Batch upsert to Pinecone
    index.upsert(vectors=vectors, namespace="default")
    print(f"Stored {len(vectors)} chunks with embeddings")
    
    return len(vectors)

# Step 2: Retrieval function
def retrieve_context(query, top_k=3):
    """Retrieve most relevant document chunks"""
    # Embed query
    query_response = openrouter_client.embeddings.create(
        model="openai/text-embedding-3-large",
        input=query
    )
    query_embedding = query_response.data[0].embedding
    
    # Search vector database
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        namespace="default"
    )
    
    # Extract relevant text
    relevant_chunks = [match['metadata']['text'] for match in results['matches']]
    return relevant_chunks

# Step 3: Response generation
def generate_response(query, context, model="meta-llama/llama-3.3-70b-instruct"):
    """Generate answer using retrieved context"""
    prompt = f"""Answer the question based on the context below. If the context doesn't contain relevant information, say so.

Context:
{chr(10).join(context)}

Question: {query}

Answer:"""
    
    response = openrouter_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=512
    )
    
    return response.choices[0].message.content

# Complete RAG pipeline
def perform_rag(question):
    """Execute full RAG workflow"""
    # Retrieve relevant chunks
    context = retrieve_context(question, top_k=3)
    
    # Generate response
    answer = generate_response(question, context)
    
    return {
        "answer": answer,
        "sources": context,
        "model": "meta-llama/llama-3.3-70b-instruct"
    }

# Usage example
documents = [
    "The Python programming language was created by Guido van Rossum and first released in 1991. It emphasizes code readability with significant whitespace.",
    "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.",
    "Retrieval-Augmented Generation (RAG) combines information retrieval with large language model generation to provide accurate, grounded responses.",
]

# Store documents
chunk_and_store_documents(documents)

# Query the system
result = perform_rag("What is RAG and how does it work?")
print(f"Answer: {result['answer']}\n")
print(f"Based on {len(result['sources'])} source chunks")