# RAG OpenRouter

A complete Retrieval-Augmented Generation (RAG) pipeline using OpenRouter's API and Pinecone vector database.

## Description

This project demonstrates a full RAG implementation that combines document chunking, vector embeddings, semantic search, and LLM-based answer generation. It uses OpenRouter as a unified API gateway to access embedding models and large language models, with Pinecone providing scalable vector storage and similarity search.

## Features

- **Document Chunking**: Intelligent text splitting with configurable chunk sizes and overlap using LangChain
- **Vector Embeddings**: Generate high-quality embeddings using OpenAI's text-embedding-3-large (3072 dimensions)
- **Vector Storage**: Pinecone serverless vector database with cosine similarity search
- **Semantic Retrieval**: Find relevant document chunks based on query similarity
- **Hybrid Search**: Combine vector similarity with BM25 keyword matching for improved retrieval
- **Two-Stage Retrieval**: Optional reranking with cross-encoder models for improved precision
- **Document Classification**: Example-based categorization using embedding similarity
- **Document Clustering**: Automatic topic discovery and grouping using K-Means clustering
- **LangChain Integration**: High-level abstractions for rapid RAG development with FAISS
- **LLM Generation**: Answer questions using retrieved context with Meta's Llama 3.3 70B
- **Streaming Responses**: Real-time text generation for better user experience
- **Complete RAG Pipeline**: End-to-end workflow from document ingestion to answer generation
- **Modular Architecture**: Clean separation of concerns for easy customization
- Environment-based configuration for API keys

## Requirements

- Python >= 3.11
- OpenAI Python SDK >= 2.7.2
- python-dotenv >= 1.0.0
- pinecone-client >= 3.0.0
- langchain-text-splitters >= 1.0.0
- pypdf >= 6.0.0 (for PDF loading)
- reportlab >= 4.0.0 (for PDF generation)
- sentence-transformers >= 5.0.0 (optional, for reranking)
- rank-bm25 >= 0.2.2 (optional, for hybrid search)
- scikit-learn >= 1.5.0 (optional, for clustering)
- langchain >= 1.0.0 (optional, for LangChain integration)
- langchain-openai >= 1.0.0 (optional, for LangChain integration)
- langchain-community >= 0.4.0 (optional, for LangChain integration)
- faiss-cpu >= 1.12.0 (optional, for local vector storage)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/sovdienko/rag-openrouter.git
   cd rag-openrouter
   ```

2. Install dependencies using uv (recommended) or pip:
   ```bash
   uv sync
   ```
   or
   ```bash
   pip install -e .
   ```

3. Create a `.env` file in the project root and add your API keys:
   ```env
   OPENROUTER_API_KEY=your_openrouter_key_here
   PINECONE_API_KEY=your_pinecone_key_here
   ```

## Getting API Keys

### OpenRouter API Key
1. Visit [OpenRouter](https://openrouter.ai/)
2. Sign up or log in to your account
3. Navigate to the API Keys section
4. Generate a new API key
5. Add it to your `.env` file

### Pinecone API Key
1. Visit [Pinecone](https://www.pinecone.io/)
2. Sign up for a free account
3. Create a new project
4. Navigate to API Keys in your project dashboard
5. Copy your API key and add it to your `.env` file

## Usage

### Modular RAG Pipeline (Recommended)

The application now uses a clean modular architecture:

```bash
python example.py
```

This demonstrates the full pipeline:
1. Initialize the RAG pipeline
2. Ingest and chunk documents
3. Generate and store embeddings
4. Query the system with natural language
5. Generate grounded answers with sources

### Using the Pipeline in Your Code

```python
from rag_app.pipeline import RAGPipeline

# Initialize
pipeline = RAGPipeline()

# Ingest documents
documents = ["Your text here...", "More text..."]
result = pipeline.ingest_documents(documents)

# Query
answer = pipeline.query("Your question here?")
print(answer['answer'])
```

### Loading PDF Documents

Use the PDFLoader to load documents from PDF files:

```bash
python example_pdf_loader.py
```

Or use programmatically:

```python
from rag_app import PDFLoader, RAGPipeline

# Load PDFs from folder (text only)
loader = PDFLoader()
documents = loader.load_from_folder("rag-docs")

# Load with metadata (RECOMMENDED - includes filename tracking)
docs_with_metadata = loader.load_with_metadata("rag-docs")
for doc in docs_with_metadata:
    print(f"File: {doc['filename']}")
    print(f"Text: {doc['text'][:100]}...")

# Load single PDF
text = loader.load_single_file("document.pdf")

# Use with RAG pipeline - metadata is automatically stored in vector DB
pipeline = RAGPipeline()
pipeline.ingest_documents(docs_with_metadata)  # Filenames stored in Pinecone!
answer = pipeline.query("Your question?")

# Access source filenames in retrieval results
results = pipeline.retriever.retrieve("Your question?", top_k=3)
for result in results:
    print(f"From: {result['metadata']['source']}")  # Source contains filename
    print(f"Chunk: {result['metadata']['chunk'] + 1}/{result['metadata']['total_chunks']}")
    print(f"Text: {result['text']}")
```

**Filename Tracking:** When using `load_with_metadata()`, each chunk stored in Pinecone includes:
- `source`: PDF filename (e.g., "ai_topic_01.pdf") - directly contains the filename
- `chunk`: Chunk number within document (0-indexed)
- `total_chunks`: Total chunks in document
- `text`: The actual chunk content

### Hybrid Search (Vector + Keyword)

Combine vector similarity with BM25 keyword matching for improved retrieval:

```bash
python example_hybrid_search.py
```

Or use programmatically:

```python
from rag_app.pipeline import RAGPipeline

# Enable hybrid search during initialization
pipeline = RAGPipeline(use_hybrid=True, hybrid_alpha=0.5)
pipeline.ingest_documents(documents)

# Query with hybrid search (50% vector + 50% keyword)
result = pipeline.query(
    "What is BM25 ranking algorithm?",
    top_k=3,
    initial_k=20  # Retrieve more candidates for hybrid scoring
)
print(result['answer'])

# Access hybrid scores
results = pipeline.retriever.retrieve(
    "Your question?",
    top_k=3,
    use_hybrid=True,
    initial_k=20
)
for res in results:
    print(f"Hybrid: {res['hybrid_score']:.4f}")
    print(f"Vector: {res['vector_score']:.4f}")
    print(f"Keyword: {res['keyword_score']:.4f}")
```

**How It Works:**
- **Vector Search**: Captures semantic meaning and contextual understanding
- **Keyword Search (BM25)**: Ensures exact term matching and lexical relevance
- **Alpha Parameter**: Controls the balance (0-1)
  - `alpha=1.0`: Pure vector similarity search
  - `alpha=0.5`: Balanced hybrid (default)
  - `alpha=0.0`: Pure keyword search
- **Benefits**: Best of both worlds - semantic understanding + exact term matching

**When to Use Hybrid Search:**
- Queries with specific terms or proper nouns (e.g., "OpenAI GPT-4", "BERT model")
- Technical terminology (e.g., "BM25", "cosine similarity")
- Domain-specific content (medical terms, legal jargon, product names)
- Exact phrases or acronyms

**Recommended Alpha Values:**
- `alpha=0.7-0.9`: When semantic understanding is more important
- `alpha=0.5`: Balanced approach (default)
- `alpha=0.1-0.3`: When exact term matching is critical

### Two-Stage Retrieval with Reranking

Enable reranking for improved retrieval precision:

```bash
python example_reranking.py
```

Or use programmatically:

```python
from rag_app.pipeline import RAGPipeline

# Enable reranker during initialization
pipeline = RAGPipeline(use_reranker=True)
pipeline.ingest_documents(documents)

# Two-stage retrieval: retrieve 15 candidates, rerank to top 3
result = pipeline.query(
    "Your question?",
    top_k=3,
    use_reranking=True,
    initial_k=15
)
print(result['answer'])
```

**How It Works:**
- **Stage 1**: Fast bi-encoder retrieves 15-50 candidates using vector similarity
- **Stage 2**: Cross-encoder reranks candidates for precise relevance scoring
- **Benefits**: Combines speed of vector search with accuracy of cross-encoders
- **Performance**: ~100-200ms latency increase with significantly better precision

### Combining Hybrid Search + Reranking (Maximum Quality)

For the best retrieval quality, combine both techniques:

```python
from rag_app.pipeline import RAGPipeline

# Enable both hybrid search and reranking
pipeline = RAGPipeline(
    use_hybrid=True,
    use_reranker=True,
    hybrid_alpha=0.5
)
pipeline.ingest_documents(documents)

# Three-stage retrieval:
# 1. Vector similarity (retrieve 20 candidates)
# 2. Hybrid re-scoring (vector + keyword)
# 3. Cross-encoder reranking (final top 3)
result = pipeline.query(
    "What are transformers in deep learning?",
    top_k=3,
    initial_k=20
)
print(result['answer'])
```

**Retrieval Pipeline:**
1. **Stage 1**: Fast vector similarity retrieves 20+ candidates
2. **Stage 2**: Hybrid scoring combines vector + keyword scores
3. **Stage 3**: Cross-encoder reranking selects final top K

This approach provides maximum retrieval quality by leveraging semantic search, keyword matching, and precision reranking.

### Document Classification

Classify documents into categories using example-based embedding similarity:

```bash
python example_classification.py
```

Or use programmatically:

```python
from rag_app import DocumentClassifier

# Initialize classifier
classifier = DocumentClassifier()

# Define categories with examples
category_examples = {
    "technical": [
        "Python programming guide",
        "API documentation",
        "Software architecture patterns"
    ],
    "business": [
        "Quarterly earnings report",
        "Market research findings",
        "Sales strategy document"
    ],
    "support": [
        "How to reset password",
        "Installation troubleshooting",
        "FAQ about billing"
    ]
}

classifier.set_categories(category_examples)

# Classify a single document
doc = "The new authentication API endpoint supports OAuth 2.0"
category, confidence = classifier.classify(doc)
print(f"Category: {category}, Confidence: {confidence:.3f}")

# View all category scores
all_scores = classifier.classify(doc, return_all_scores=True)
for cat, score in sorted(all_scores.items(), key=lambda x: x[1], reverse=True):
    print(f"{cat}: {score:.3f}")

# Classify multiple documents efficiently
documents = ["API guide", "Sales report", "Bug fix tutorial"]
results = classifier.classify_batch(documents)
for doc, (cat, conf) in zip(documents, results):
    print(f"{doc} -> {cat} ({conf:.3f})")

# Add new categories dynamically
classifier.add_category("marketing", [
    "Email campaign metrics",
    "Brand awareness strategy",
    "SEO optimization guide"
])
```

**How It Works:**
- **Zero-Shot Classification**: No training required, just provide examples
- **Embedding Similarity**: Compares document embeddings to category examples
- **Confidence Scoring**: Returns scores for all categories
- **Dynamic Categories**: Add/remove categories anytime

**Use Cases:**
- Content categorization and organization
- Support ticket routing to teams
- Intent detection in customer communications
- Document filtering and quality control
- Automatic tagging and labeling

**Tips for Better Results:**
- Provide 3-10 diverse examples per category
- Use examples with domain-specific terminology
- Set confidence thresholds for uncertain cases
- Combine with RAG for context-aware classification

### Document Clustering

Automatically discover groups and topics in unlabeled document collections:

```bash
python example_clustering.py
```

Or use programmatically:

```python
from rag_app import DocumentClusterer

# Initialize clusterer
clusterer = DocumentClusterer()

# Cluster customer feedback into groups
feedback = [
    "Login takes too long",
    "Great customer support",
    "App crashes on iOS",
    "Love the dark mode",
    "Please add PDF export",
    # ... more items
]

# Cluster into 3 groups
clusters = clusterer.cluster(feedback, n_clusters=3)
for cluster_id, items in clusters.items():
    print(f"\nCluster {cluster_id}: {len(items)} items")
    for item in items[:3]:  # Show first 3
        print(f"  - {item}")

# Get cluster with metadata (sorted by relevance)
clusters_meta = clusterer.cluster_with_metadata(feedback, n_clusters=3)
for cluster_id, items in clusters_meta.items():
    print(f"\nCluster {cluster_id} (most representative):")
    for item in items[:2]:
        print(f"  - {item['text']} (distance: {item['distance']:.4f})")

# Find optimal number of clusters
scores = clusterer.find_optimal_clusters(feedback, min_clusters=2, max_clusters=6)
best_k = max(scores.items(), key=lambda x: x[1])[0]
print(f"Optimal clusters: {best_k}")

# Predict cluster for new documents
cluster_id, distance = clusterer.predict_cluster("New feedback text")
print(f"Belongs to cluster {cluster_id}")

# Get cluster summaries
summaries = clusterer.get_cluster_summaries(clusters)
for cluster_id, summary in summaries.items():
    print(f"Cluster {cluster_id}: {summary['size']} docs ({summary['percentage']:.1f}%)")
```

**How It Works:**
- **K-Means Clustering**: Groups documents by embedding similarity
- **Automatic Discovery**: No labels required - finds patterns automatically
- **Silhouette Analysis**: Helps find optimal number of clusters
- **Distance Metrics**: Identifies most representative documents per cluster

**Use Cases:**
- Customer feedback analysis and theme discovery
- Support ticket organization and routing
- Bug report grouping and deduplication
- Content organization and taxonomy creation
- Market research and sentiment clustering
- Email categorization and filtering

**Tips for Better Results:**
- Start with 3-7 clusters for most use cases
- Use `find_optimal_clusters()` to determine best cluster count
- Review cluster examples to verify coherence
- Combine with classification to label discovered clusters
- Re-cluster periodically as data evolves

### LangChain Integration

Use LangChain for rapid RAG prototyping with high-level abstractions:

```bash
python example_langchain.py
```

Or use programmatically:

```python
from rag_app import LangChainRAG

# Initialize (uses FAISS for local vector storage)
rag = LangChainRAG(
    model_name="meta-llama/llama-3.3-70b-instruct",
    temperature=0.7,
    chunk_size=1000,
    chunk_overlap=200
)

# Load PDFs from folder
docs = rag.load_from_folder("rag-docs")
print(f"Loaded {len(docs)} documents")

# Ingest documents
stats = rag.ingest_documents(docs, top_k=3)
print(f"Created {stats['num_chunks']} chunks")

# Query the system
result = rag.query("What is machine learning?")
print(result['answer'])
print(f"Sources: {result['num_sources']}")

# Similarity search without LLM
results = rag.similarity_search("neural networks", top_k=5)
for result in results:
    print(result['text'][:100])

# Save vector store for later
rag.save_vectorstore("my_vectorstore")

# Load vector store
rag2 = LangChainRAG()
rag2.load_vectorstore("my_vectorstore")
result = rag2.query("Explain deep learning")

# Work with text strings directly
texts = [
    "Machine learning enables systems to learn from data",
    "Neural networks are inspired by biological neurons",
    "Deep learning uses multiple layers for feature extraction"
]
rag3 = LangChainRAG()
rag3.ingest_documents(texts)
result = rag3.query("What is deep learning?")
```

**How It Works:**
- **LangChain Chains**: Pre-built RetrievalQA chain for question answering
- **FAISS Vector Store**: Local vector storage (no cloud database required)
- **Simplified API**: Fewer lines of code than custom pipeline
- **Persistence**: Save/load vector stores to disk

**Benefits:**
- Rapid prototyping and MVPs
- Local development without cloud dependencies
- Built-in prompt engineering
- Easy model experimentation
- Declarative, high-level code

**When to Use:**
- Building quick prototypes
- Experimenting with different approaches
- Local development and testing
- Learning RAG concepts
- Prefer simplicity over fine control

**Comparison with Custom Pipeline:**

```python
# LangChain (Simple & Fast)
rag = LangChainRAG()
docs = rag.load_from_folder("docs")
rag.ingest_documents(docs)
result = rag.query("What is AI?")

# Custom Pipeline (More Control)
pipeline = RAGPipeline(use_hybrid=True, use_reranker=True)
loader = PDFLoader()
docs = loader.load_with_metadata("docs")
pipeline.ingest_documents(docs)
result = pipeline.query("What is AI?", top_k=5, initial_k=20)
```

Choose LangChain for rapid development, custom pipeline for production optimization.

### Legacy Scripts

The original monolithic scripts are still available:

```bash
python rag-pipeline-pinecone.py  # Original complete pipeline
python chunk.py                   # Document chunking test
python sample1.py                 # Basic embeddings
```

## How It Works

### 1. Document Ingestion
Documents are split into overlapping chunks (default: 1000 characters, 200 overlap) to maintain context while staying within model limits.

### 2. Embedding Generation
Each chunk is converted to a 3072-dimensional vector using OpenAI's text-embedding-3-large model via OpenRouter.

### 3. Vector Storage
Embeddings are stored in Pinecone's serverless index with metadata including the original text and source information.

### 4. Semantic Search
User queries are embedded and compared against stored vectors using cosine similarity to retrieve the most relevant chunks.

### 5. Hybrid Search (Optional)
For improved retrieval, combine vector similarity with BM25 keyword matching:
1. **Vector Search**: Performs semantic similarity search
2. **Keyword Search**: Applies BM25 ranking algorithm for exact term matching
3. **Score Combination**: Weighted combination using alpha parameter (0-1)
4. Return top K candidates with combined scores

This approach balances semantic understanding with lexical matching, particularly effective for queries with specific terms or technical terminology.

### 6. Two-Stage Retrieval (Optional)
For improved precision, the system can:
1. **Stage 1**: Retrieve 15-50 candidates using fast vector similarity search (or hybrid search)
2. **Stage 2**: Rerank candidates with cross-encoder model for accurate relevance scoring
3. Return top K most relevant results

This approach combines the speed of bi-encoder embeddings with the precision of cross-encoder models, improving retrieval quality with minimal latency increase (~100-200ms).

### 7. Document Classification (Optional)
For content categorization without RAG:
1. **Define Categories**: Provide example documents for each category
2. **Embed Examples**: Generate embeddings for all category examples
3. **Classify Documents**: Compare new documents to category examples using cosine similarity
4. **Score Averaging**: Average similarity across all examples in each category

This zero-shot approach enables instant classification without model training, ideal for content routing and organization.

### 8. Document Clustering (Optional)
For automatic topic discovery in unlabeled data:
1. **Generate Embeddings**: Convert all documents to vector embeddings
2. **K-Means Clustering**: Group documents by embedding similarity
3. **Cluster Assignment**: Assign each document to nearest cluster center
4. **Optimization**: Use silhouette analysis to find optimal cluster count

This unsupervised approach automatically discovers hidden patterns and themes in document collections.

### 9. Answer Generation
Retrieved chunks are passed as context to Llama 3.3 70B, which generates a grounded answer based only on the provided information.

## Project Structure

```
rag-openrouter/
├── rag_app/                   # Main application package
│   ├── __init__.py           # Package initialization
│   ├── config.py             # Configuration management
│   ├── embeddings.py         # Embedding generation service
│   ├── vector_store.py       # Pinecone vector database operations
│   ├── document_processor.py # Document chunking logic
│   ├── retriever.py          # Semantic search retrieval
│   ├── hybrid_search.py      # BM25 keyword + vector hybrid search
│   ├── reranker.py           # Cross-encoder reranking
│   ├── classifier.py         # Example-based document classification
│   ├── clustering.py         # K-Means document clustering
│   ├── langchain_rag.py      # LangChain integration wrapper
│   ├── generator.py          # LLM answer generation
│   ├── pdf_loader.py         # PDF document loader
│   └── pipeline.py           # Main RAG pipeline orchestrator
├── rag-docs/                  # Sample PDF documents for testing
├── example.py                 # Example usage of modular pipeline
├── example_streaming.py       # Streaming responses example
├── example_hybrid_search.py   # Hybrid search (vector + keyword) example
├── example_reranking.py       # Two-stage retrieval example
├── example_classification.py  # Document classification example
├── example_clustering.py      # Document clustering example
├── example_langchain.py       # LangChain integration example
├── example_pdf_loader.py      # PDF loader usage example
├── generate_pdfs.py           # Script to generate sample PDFs
├── rag-pipeline-pinecone.py  # Legacy monolithic implementation
├── chunk.py                   # Legacy chunking script
├── sample1.py                 # Legacy embedding example
├── pyproject.toml             # Project configuration and dependencies
├── .env                       # Environment variables (not tracked in git)
└── README.md                  # This file
```

## Architecture

The modular design separates concerns:

- **config.py**: Centralized configuration with environment variable management
- **embeddings.py**: Handles all embedding generation via OpenRouter
- **vector_store.py**: Abstracts Pinecone operations (upsert, query, delete)
- **document_processor.py**: Text chunking with LangChain splitters
- **retriever.py**: Combines embeddings + vector search for semantic retrieval
- **hybrid_search.py**: BM25 keyword matching combined with vector similarity
- **reranker.py**: Cross-encoder models for precision reranking
- **classifier.py**: Example-based document classification using embedding similarity
- **clustering.py**: K-Means clustering for automatic topic discovery
- **langchain_rag.py**: LangChain integration with FAISS for rapid development
- **generator.py**: LLM-based answer generation with context and streaming
- **pdf_loader.py**: PDF document loading with metadata support
- **pipeline.py**: Orchestrates all components into a unified workflow

## Configuration

Configure via environment variables in `.env` or programmatically:

```python
from rag_app.config import Config

config = Config.from_env()
config.chunk_size = 1500  # Override defaults
config.top_k = 5
```

**Key parameters:**
- **chunk_size**: 1000 characters (adjustable for your documents)
- **chunk_overlap**: 200 characters (prevents context loss at boundaries)
- **embedding_model**: openai/text-embedding-3-large (3072 dimensions)
- **llm_model**: meta-llama/llama-3.3-70b-instruct
- **top_k**: 3 (number of chunks to retrieve)
- **vector_metric**: cosine similarity

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
