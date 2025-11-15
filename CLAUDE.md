# CLAUDE.md - AI Assistant Guide for RAG OpenRouter

## Project Overview

**Name:** RAG OpenRouter
**Version:** 1.0.0
**Type:** Retrieval-Augmented Generation (RAG) Pipeline
**Python Version:** >= 3.11
**License:** MIT

This is a production-ready RAG implementation that combines document processing, vector embeddings, semantic search, and LLM-based answer generation. The project uses OpenRouter as a unified API gateway for accessing embedding models and LLMs, with Pinecone providing scalable vector storage.

## Repository Structure

```
rag-openrouter/
├── rag_app/                      # Main application package
│   ├── __init__.py              # Package exports
│   ├── config.py                # Configuration management with dataclasses
│   ├── embeddings.py            # Embedding generation via OpenRouter
│   ├── cache.py                 # Embedding caching + cost tracking
│   ├── rate_limiting.py         # Rate limiting + retry with exponential backoff
│   ├── vector_store.py          # Pinecone vector database operations
│   ├── document_processor.py    # Document chunking with LangChain
│   ├── retriever.py             # Semantic search retrieval
│   ├── hybrid_search.py         # BM25 keyword + vector hybrid search
│   ├── reranker.py              # Cross-encoder reranking
│   ├── classifier.py            # Example-based document classification
│   ├── clustering.py            # K-Means document clustering
│   ├── langchain_rag.py         # LangChain integration with FAISS
│   ├── generator.py             # LLM answer generation (streaming support)
│   └── pipeline.py              # Main RAG pipeline orchestrator
├── rag-docs/                     # Sample PDF documents for testing
├── .cache/                       # Embedding cache (gitignored)
├── langchain_vectorstore/        # FAISS vector store for LangChain
├── example.py                    # Example: Modular pipeline usage
├── example_streaming.py          # Example: Streaming responses
├── example_hybrid_search.py      # Example: Vector + keyword search
├── example_reranking.py          # Example: Two-stage retrieval
├── example_classification.py     # Example: Document classification
├── example_clustering.py         # Example: Document clustering
├── example_langchain.py          # Example: LangChain integration
├── example_cost_optimization.py  # Example: Caching and cost tracking
├── example_batch_processing.py   # Example: Batch embedding generation
├── example_rate_limiting.py      # Example: Rate limiting and retry
├── example_with_filenames.py     # Example: PDF loading with metadata
├── generate_pdfs.py              # Generate sample PDF documents
├── clear_and_test.py            # Clear Pinecone index and test
├── test_metadata.py             # Test metadata handling
├── verify_metadata_structure.py  # Verify metadata structure
├── rag-pipeline-pinecone.py     # Legacy monolithic implementation
├── pyproject.toml               # Project configuration (uv/pip)
├── .env                         # Environment variables (gitignored)
├── .gitignore                   # Git ignore patterns
└── README.md                    # User-facing documentation
```

## Architecture & Design Principles

### Modular Architecture

The codebase follows a **clean, modular design** with clear separation of concerns:

1. **config.py** - Centralized configuration using dataclasses
2. **embeddings.py** - All embedding generation via OpenRouter
3. **cache.py** - Embedding caching + cost tracking for optimization
4. **rate_limiting.py** - Rate limiting + automatic retry with exponential backoff
5. **vector_store.py** - Abstracts Pinecone operations (upsert, query, delete)
6. **document_processor.py** - Text chunking with LangChain splitters
7. **retriever.py** - Combines embeddings + vector search for semantic retrieval
8. **hybrid_search.py** - BM25 keyword matching combined with vector similarity
9. **reranker.py** - Cross-encoder models for precision reranking
10. **classifier.py** - Example-based classification using embedding similarity
11. **clustering.py** - K-Means clustering for topic discovery
12. **langchain_rag.py** - LangChain integration wrapper with FAISS
13. **generator.py** - LLM-based answer generation with streaming support
14. **pdf_loader.py** - PDF document loading with metadata support
15. **pipeline.py** - Orchestrates all components into unified workflow

### Key Design Patterns

- **Dependency Injection**: Components receive dependencies via constructor
- **Dataclasses**: Used for configuration and data transfer objects
- **Type Hints**: All functions use type annotations
- **Factory Pattern**: `Config.from_env()` creates configuration from environment
- **Composition over Inheritance**: Components are composed in pipeline
- **Optional Features**: Hybrid search, reranking, caching are opt-in

## Development Workflows

### Setting Up Development Environment

```bash
# Clone repository
git clone https://github.com/sovdienko/rag-openrouter.git
cd rag-openrouter

# Install dependencies (using uv - recommended)
uv sync

# Or using pip
pip install -e .

# Create .env file with API keys
cat > .env << EOF
OPENROUTER_API_KEY=your_openrouter_key_here
PINECONE_API_KEY=your_pinecone_key_here
EOF
```

### Running Examples

All examples are self-contained and can be run directly:

```bash
# Basic pipeline
python example.py

# Streaming responses
python example_streaming.py

# Hybrid search (vector + keyword)
python example_hybrid_search.py

# Two-stage retrieval with reranking
python example_reranking.py

# Document classification
python example_classification.py

# Document clustering
python example_clustering.py

# LangChain integration
python example_langchain.py

# Cost optimization and caching
python example_cost_optimization.py

# Batch processing
python example_batch_processing.py

# Rate limiting and retry
python example_rate_limiting.py

# PDF loading with filenames
python example_with_filenames.py
```

### Testing Workflow

```bash
# Clear Pinecone index and run test
python clear_and_test.py

# Test metadata handling
python test_metadata.py

# Verify metadata structure
python verify_metadata_structure.py
```

### Generating Test Data

```bash
# Generate sample PDF documents
python generate_pdfs.py
```

## Key Conventions for AI Assistants

### 1. Configuration Management

**Always use environment variables for API keys:**

```python
# CORRECT: Load from environment
from rag_app import Config
config = Config.from_env()

# INCORRECT: Never hardcode keys
config = Config(
    openrouter_api_key="sk-xxx",  # NEVER DO THIS
    pinecone_api_key="pk-xxx"     # NEVER DO THIS
)
```

**Configuration is validated on pipeline initialization:**

```python
from rag_app import RAGPipeline

# This will raise ValueError if API keys are missing
pipeline = RAGPipeline()  # Calls config.validate()
```

### 2. Embedding Models

**Default embedding model:**
- `openai/text-embedding-3-large` (3072 dimensions)
- Cost: $0.13 per 1M tokens
- High accuracy for semantic search

**Alternative (cheaper):**
- `openai/text-embedding-3-small` (1536 dimensions)
- Cost: $0.02 per 1M tokens (6.5x cheaper)
- Good for most applications

**When to change embedding dimensions:**
- If changing embedding model, update `embedding_dimension` in config
- Pinecone index must match embedding dimensions
- Different dimensions require different Pinecone index

### 3. LLM Models

**Default LLM:**
- `meta-llama/llama-3.3-70b-instruct`
- Cost: $0.88 per 1M input tokens
- Production-quality responses

**For development/testing:**
- `meta-llama/llama-3.3-8b-instruct` (Free tier available)

### 4. Document Ingestion Patterns

**Simple text ingestion:**
```python
from rag_app import RAGPipeline

pipeline = RAGPipeline()
documents = ["Document 1 text", "Document 2 text"]
result = pipeline.ingest_documents(documents)
```

**PDF ingestion with metadata (RECOMMENDED):**
```python
from rag_app import RAGPipeline, PDFLoader

loader = PDFLoader()
docs_with_metadata = loader.load_with_metadata("rag-docs")
# Returns: [{"text": "...", "filename": "doc.pdf"}, ...]

pipeline = RAGPipeline()
pipeline.ingest_documents(docs_with_metadata)
# Filenames automatically stored in Pinecone metadata
```

**Important metadata conventions:**
- `source`: Contains filename (e.g., "ai_topic_01.pdf")
- `chunk`: Chunk number within document (0-indexed)
- `total_chunks`: Total chunks in document
- `text`: Actual chunk content

### 5. Query Patterns

**Standard query:**
```python
result = pipeline.query("What is machine learning?")
print(result['answer'])
```

**With hybrid search (vector + keyword):**
```python
pipeline = RAGPipeline(use_hybrid=True, hybrid_alpha=0.5)
result = pipeline.query(
    "What is BM25 algorithm?",
    top_k=3,
    initial_k=20  # Retrieve 20 candidates for hybrid scoring
)
```

**With reranking (two-stage retrieval):**
```python
pipeline = RAGPipeline(use_reranker=True)
result = pipeline.query(
    "Explain transformers",
    top_k=3,
    initial_k=15  # Retrieve 15 candidates, rerank to top 3
)
```

**Maximum quality (hybrid + reranking):**
```python
pipeline = RAGPipeline(use_hybrid=True, use_reranker=True, hybrid_alpha=0.5)
result = pipeline.query("Question?", top_k=3, initial_k=20)
```

**Streaming responses:**
```python
result = pipeline.query("Question?", stream=True)
for chunk in result['stream']:
    print(chunk, end="", flush=True)
```

### 6. Caching and Cost Optimization

**Always enable caching in production:**

```python
from rag_app import Config, EmbeddingCache, CostTracker
from rag_app.embeddings import EmbeddingService

# Create cache and cost tracker
cache = EmbeddingCache(cache_dir=".cache/embeddings", memory_only=False)
tracker = CostTracker()

# Create embedding service with optimization
config = Config.from_env()
embedding_service = EmbeddingService(config, cache=cache, cost_tracker=tracker)

# Use with pipeline
pipeline = RAGPipeline()
pipeline.embedding_service = embedding_service

# View statistics
stats = cache.get_stats()
print(f"Cache hit rate: {stats['hit_rate']:.1f}%")

report = tracker.get_report()
print(f"Total cost: ${report['total_cost']:.6f}")
```

**Cache strategies:**
- **Memory-only**: Fast, cleared on restart
- **Disk cache**: Persistent across restarts
- **Two-tier** (default): Memory + disk for best performance

### 7. Rate Limiting and Retry

**Production-ready configuration:**

```python
from rag_app import Config, RateLimiter
from rag_app.embeddings import EmbeddingService

config = Config.from_env()
rate_limiter = RateLimiter(requests_per_minute=50)

service = EmbeddingService(
    config,
    rate_limiter=rate_limiter,
    use_retry=True,
    max_retries=5
)

# Automatic retry with exponential backoff:
# 1s, 2s, 4s, 8s, 16s (capped at 60s)
```

**Rate limit recommendations:**
- Free tier: `requests_per_minute=10`
- Basic tier: `requests_per_minute=60`
- Pro tier: `requests_per_minute=300`

### 8. Batch Processing

**Process large collections efficiently:**

```python
from rag_app import Config, EmbeddingCache
from rag_app.embeddings import EmbeddingService

config = Config.from_env()
cache = EmbeddingCache(cache_dir=".cache/embeddings")
service = EmbeddingService(config, cache=cache)

# Batch process 1000 documents
documents = [f"Document {i} content..." for i in range(1000)]

embeddings = service.batch_embed_documents(
    texts=documents,
    batch_size=100,  # 100 texts per API request
    show_progress=True
)
```

**Batch size guidelines:**
- Short texts (<50 chars): `batch_size=200`
- Medium texts (50-500 chars): `batch_size=100`
- Long texts (>500 chars): `batch_size=50`
- Very long texts (>2000 chars): `batch_size=25`

### 9. Document Classification

**Example-based classification (zero-shot):**

```python
from rag_app import DocumentClassifier

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
        "Market research findings"
    ]
}

classifier.set_categories(category_examples)

# Classify
doc = "The new authentication API endpoint supports OAuth 2.0"
category, confidence = classifier.classify(doc)
print(f"{category}: {confidence:.3f}")

# Batch classify
results = classifier.classify_batch(documents)
```

### 10. Document Clustering

**Automatic topic discovery:**

```python
from rag_app import DocumentClusterer

clusterer = DocumentClusterer()

# Cluster into groups
feedback = ["Login slow", "Great support", "App crashes", ...]
clusters = clusterer.cluster(feedback, n_clusters=3)

# Find optimal number of clusters
scores = clusterer.find_optimal_clusters(feedback, min_clusters=2, max_clusters=6)
best_k = max(scores.items(), key=lambda x: x[1])[0]

# Predict cluster for new document
cluster_id, distance = clusterer.predict_cluster("New feedback")
```

## Git Workflow

### Current Branch

**Development branch:** `claude/claude-md-mi08jv6iqc86ivev-01VckAeN7aAi2bdTsoF4e5N1`

### Recent Commits

```
15c052f - rate-limiting
f8ae741 - batch processing
dda71d8 - caching
5997010 - langchain
c244abe - classification and clustering
b747149 - docs classifications
c32edc4 - hybrid search
8dd60c2 - clear and test
13f9294 - reranking
46e4e59 - refactoring
```

### Git Conventions

1. **Commit messages**: Short, descriptive (imperative mood)
2. **Branch naming**: `feature/description` or `claude/session-id`
3. **Always push to designated branch**
4. **Retry on network errors**: Up to 4 times with exponential backoff

## Common Pitfalls and How to Avoid Them

### 1. Missing Environment Variables

**Problem:** Pipeline fails with `ValueError: OPENROUTER_API_KEY not found`

**Solution:**
```bash
# Create .env file
echo "OPENROUTER_API_KEY=your_key" >> .env
echo "PINECONE_API_KEY=your_key" >> .env
```

### 2. Embedding Dimension Mismatch

**Problem:** Pinecone error about dimension mismatch

**Solution:**
- Ensure `embedding_dimension` in config matches model output
- `text-embedding-3-large`: 3072 dimensions
- `text-embedding-3-small`: 1536 dimensions
- Different models require different Pinecone indexes

### 3. Rate Limit Errors

**Problem:** API returns 429 Too Many Requests

**Solution:**
- Enable rate limiting: `RateLimiter(requests_per_minute=50)`
- Enable retry: `use_retry=True, max_retries=5`
- Use caching to reduce API calls

### 4. Out of Memory on Large Documents

**Problem:** Memory error when processing many documents

**Solution:**
- Use batch processing: `batch_embed_documents(texts, batch_size=100)`
- Reduce chunk size: `config.chunk_size = 500`
- Process in smaller batches

### 5. Poor Retrieval Quality

**Problem:** Retrieved chunks not relevant to query

**Solutions:**
- Enable hybrid search: `use_hybrid=True, hybrid_alpha=0.5`
- Enable reranking: `use_reranker=True, initial_k=15`
- Increase `top_k` to retrieve more chunks
- Adjust chunk size and overlap
- Use better embedding model

### 6. Slow Queries

**Problem:** Queries take too long

**Solutions:**
- Enable caching for repeated queries
- Reduce `top_k` (fewer chunks to retrieve)
- Disable reranking if not needed
- Use smaller embedding model (3-small vs 3-large)

## Dependencies and Version Requirements

### Core Dependencies

```toml
python = ">=3.11"
openai = ">=2.7.2"              # OpenRouter client
pinecone = ">=7.3.0"            # Vector database
langchain = ">=1.0.5"           # Document processing
langchain-text-splitters = ">=1.0.0"
python-dotenv = ">=1.0.0"       # Environment variables
pypdf = ">=6.2.0"               # PDF loading
```

### Optional Dependencies

```toml
sentence-transformers = ">=5.1.2"  # For reranking
rank-bm25 = ">=0.2.2"              # For hybrid search
scikit-learn = ">=1.7.2"           # For clustering
faiss-cpu = ">=1.12.0"             # For LangChain integration
langchain-openai = ">=1.0.2"       # LangChain OpenAI integration
langchain-community = ">=0.4.1"    # LangChain community integrations
reportlab = ">=4.4.4"              # PDF generation (tests)
```

## Performance Characteristics

### Latency Benchmarks

- **Embedding generation**: ~100-300ms per request
- **Batch embedding (100 texts)**: ~500-1000ms total (5-10ms per text)
- **Vector search**: ~50-100ms (Pinecone serverless)
- **Hybrid search**: +20-50ms overhead
- **Reranking**: +100-200ms overhead
- **LLM generation**: ~1-3 seconds (depends on output length)
- **Streaming**: First token in ~500ms

### Cache Performance

- **Memory cache lookup**: <1ms
- **Disk cache lookup**: ~5-10ms
- **Cache vs API**: 100-1000x speedup

### Cost Estimates (per 1M tokens)

| Service | Model | Cost |
|---------|-------|------|
| Embeddings | text-embedding-3-small | $0.02 |
| Embeddings | text-embedding-3-large | $0.13 |
| LLM | Llama 3.3 8B | Free |
| LLM | Llama 3.3 70B | $0.88 |

## Testing Strategy

### Unit Testing

Currently, the project uses example scripts for testing. For production:

```bash
# Test basic pipeline
python example.py

# Test hybrid search
python example_hybrid_search.py

# Test reranking
python example_reranking.py

# Clear index and test
python clear_and_test.py
```

### Integration Testing

```bash
# Test metadata handling
python test_metadata.py

# Verify metadata structure
python verify_metadata_structure.py
```

## API Reference Quick Guide

### RAGPipeline

**Main orchestrator for RAG workflow**

```python
from rag_app import RAGPipeline

# Initialize
pipeline = RAGPipeline(
    config=None,              # Uses Config.from_env() if None
    use_reranker=False,       # Enable two-stage retrieval
    use_hybrid=False,         # Enable hybrid search
    hybrid_alpha=0.5          # Vector vs keyword weight (0-1)
)

# Ingest documents
result = pipeline.ingest_documents(
    documents,                # List[str] or List[Dict[str, Any]]
    source_prefix="doc"       # Prefix for source identifiers
)

# Query
result = pipeline.query(
    question,                 # str
    top_k=3,                  # int (final chunk count)
    return_sources=True,      # bool
    stream=False,             # bool
    use_reranking=None,       # bool (default: True if reranker enabled)
    use_hybrid=None,          # bool (default: True if hybrid enabled)
    initial_k=None            # int (candidates for reranking/hybrid)
)

# Clear index
pipeline.clear_index()

# Get statistics
stats = pipeline.get_stats()
```

### PDFLoader

**Load PDF documents with metadata**

```python
from rag_app import PDFLoader

loader = PDFLoader()

# Load with metadata (RECOMMENDED)
docs = loader.load_with_metadata("rag-docs")
# Returns: [{"text": "...", "filename": "doc.pdf"}, ...]

# Load folder (text only)
texts = loader.load_from_folder("rag-docs")

# Load single file
text = loader.load_single_file("document.pdf")
```

### DocumentClassifier

**Example-based classification**

```python
from rag_app import DocumentClassifier

classifier = DocumentClassifier()

# Set categories
classifier.set_categories(category_examples)  # Dict[str, List[str]]

# Add category
classifier.add_category("name", examples)

# Classify
category, confidence = classifier.classify(doc)
all_scores = classifier.classify(doc, return_all_scores=True)

# Batch classify
results = classifier.classify_batch(documents)
```

### DocumentClusterer

**K-Means clustering**

```python
from rag_app import DocumentClusterer

clusterer = DocumentClusterer()

# Cluster
clusters = clusterer.cluster(documents, n_clusters=3)

# With metadata
clusters_meta = clusterer.cluster_with_metadata(documents, n_clusters=3)

# Find optimal clusters
scores = clusterer.find_optimal_clusters(documents, min_clusters=2, max_clusters=6)

# Predict cluster
cluster_id, distance = clusterer.predict_cluster(document)

# Get summaries
summaries = clusterer.get_cluster_summaries(clusters)
```

### LangChainRAG

**LangChain integration wrapper**

```python
from rag_app import LangChainRAG

rag = LangChainRAG(
    model_name="meta-llama/llama-3.3-70b-instruct",
    temperature=0.7,
    chunk_size=1000,
    chunk_overlap=200
)

# Load PDFs
docs = rag.load_from_folder("rag-docs")

# Ingest
stats = rag.ingest_documents(docs, top_k=3)

# Query
result = rag.query("Question?")

# Similarity search
results = rag.similarity_search("Query", top_k=5)

# Save/load vector store
rag.save_vectorstore("path")
rag.load_vectorstore("path")
```

### EmbeddingCache

**Embedding caching for cost optimization**

```python
from rag_app import EmbeddingCache

cache = EmbeddingCache(
    cache_dir=".cache/embeddings",
    memory_only=False
)

# Get/set (automatic via EmbeddingService)
embedding = cache.get(text)
cache.set(text, embedding)

# Statistics
stats = cache.get_stats()  # hits, misses, hit_rate

# Clear
cache.clear_memory()
cache.clear_disk()
cache.clear_all()
```

### CostTracker

**Track API usage and costs**

```python
from rag_app import CostTracker

tracker = CostTracker()

# Track embeddings
tracker.track_embedding(text, model="openai/text-embedding-3-small")

# Track LLM
tracker.track_llm(input_text, output_text, model="meta-llama/llama-3.3-70b-instruct")

# Get report
report = tracker.get_report()
# Returns: {embeddings: {calls, tokens, cost}, llm: {calls, ...}, total_cost}
```

### RateLimiter

**Rate limiting for API calls**

```python
from rag_app import RateLimiter, rate_limit_retry

# Create limiter
limiter = RateLimiter(
    requests_per_minute=60,
    requests_per_second=None
)

# Use as context manager
with limiter:
    result = api_call()

# Use as decorator
@rate_limit_retry(max_retries=3, initial_wait=1.0, backoff_factor=2.0)
def my_function():
    return api_call()
```

## Security Considerations

### 1. API Key Management

- **NEVER** commit `.env` files to git
- **ALWAYS** use environment variables for secrets
- **NEVER** hardcode API keys in code
- Use `.gitignore` to exclude `.env` files

### 2. Input Validation

- Validate user queries before processing
- Sanitize filenames when loading PDFs
- Validate configuration parameters

### 3. Rate Limiting

- Always enable rate limiting in production
- Prevent abuse and quota exhaustion
- Use retry logic for resilience

### 4. Cost Control

- Enable cost tracking to monitor expenses
- Set budget alerts in OpenRouter dashboard
- Use caching to reduce API calls
- Choose appropriate models for use case

## Troubleshooting Guide

### Issue: "OPENROUTER_API_KEY not found"

**Cause:** Missing environment variable

**Fix:**
```bash
echo "OPENROUTER_API_KEY=your_key" >> .env
```

### Issue: "Dimension mismatch" in Pinecone

**Cause:** Embedding model dimensions don't match Pinecone index

**Fix:**
- Check embedding model dimensions
- Recreate Pinecone index with correct dimensions
- Or change embedding model to match index

### Issue: Rate limit errors (429)

**Cause:** Too many API requests

**Fix:**
```python
service = EmbeddingService(
    config,
    rate_limiter=RateLimiter(requests_per_minute=50),
    use_retry=True,
    max_retries=5
)
```

### Issue: Slow query performance

**Cause:** No caching, too many chunks, reranking overhead

**Fix:**
- Enable caching
- Reduce `top_k`
- Disable reranking if not needed
- Use hybrid search for better relevance with fewer chunks

### Issue: Poor retrieval quality

**Cause:** Simple vector search not sufficient

**Fix:**
- Enable hybrid search: `use_hybrid=True`
- Enable reranking: `use_reranker=True`
- Increase `top_k`
- Adjust chunk size and overlap
- Use better embedding model

## Additional Resources

- **README.md**: User-facing documentation
- **Example scripts**: See all `example_*.py` files
- **OpenRouter docs**: https://openrouter.ai/docs
- **Pinecone docs**: https://docs.pinecone.io
- **LangChain docs**: https://python.langchain.com

## Changelog Summary

- **v1.0.0** - Initial release with full feature set
- Recent additions:
  - Rate limiting and retry logic
  - Batch processing
  - Embedding caching and cost tracking
  - LangChain integration
  - Document classification and clustering
  - Hybrid search (BM25 + vector)
  - Two-stage retrieval with reranking

## Contributing Guidelines

1. **Code Style**: Follow existing patterns
2. **Type Hints**: Always use type annotations
3. **Docstrings**: Document all public functions
4. **Modularity**: Keep components loosely coupled
5. **Testing**: Add example scripts for new features
6. **Documentation**: Update README.md and CLAUDE.md

## AI Assistant Best Practices

### When Adding New Features

1. **Maintain modularity**: Create new files in `rag_app/`
2. **Export in __init__.py**: Add to `__all__` list
3. **Add example script**: Create `example_feature_name.py`
4. **Update README.md**: Document usage and API
5. **Update CLAUDE.md**: Add conventions and patterns
6. **Use type hints**: Always annotate function signatures
7. **Add docstrings**: Document parameters and return values

### When Debugging

1. **Check .env file**: Verify API keys are set
2. **Check logs**: Look for error messages
3. **Test components individually**: Use example scripts
4. **Verify configurations**: Ensure models and dimensions match
5. **Check rate limits**: Monitor API usage
6. **Review costs**: Use CostTracker

### When Optimizing

1. **Enable caching**: Disk cache for persistence
2. **Use batch processing**: For large collections
3. **Enable rate limiting**: Prevent quota issues
4. **Monitor costs**: Use CostTracker
5. **Profile performance**: Identify bottlenecks
6. **Choose right model**: Balance cost vs quality

---

**Last Updated:** 2025-11-15
**Document Version:** 1.0.0
**Maintained by:** AI Assistants working with this codebase
