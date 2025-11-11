# RAG OpenRouter

A complete Retrieval-Augmented Generation (RAG) pipeline using OpenRouter's API and Pinecone vector database.

## Description

This project demonstrates a full RAG implementation that combines document chunking, vector embeddings, semantic search, and LLM-based answer generation. It uses OpenRouter as a unified API gateway to access embedding models and large language models, with Pinecone providing scalable vector storage and similarity search.

## Features

- **Document Chunking**: Intelligent text splitting with configurable chunk sizes and overlap using LangChain
- **Vector Embeddings**: Generate high-quality embeddings using OpenAI's text-embedding-3-large (3072 dimensions)
- **Vector Storage**: Pinecone serverless vector database with cosine similarity search
- **Semantic Retrieval**: Find relevant document chunks based on query similarity
- **LLM Generation**: Answer questions using retrieved context with Meta's Llama 3.3 70B
- **Complete RAG Pipeline**: End-to-end workflow from document ingestion to answer generation
- Environment-based configuration for API keys

## Requirements

- Python >= 3.11
- OpenAI Python SDK >= 2.7.2
- python-dotenv >= 1.0.0
- pinecone-client >= 3.0.0
- langchain-text-splitters >= 1.0.0

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

### 5. Answer Generation
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
│   ├── generator.py          # LLM answer generation
│   └── pipeline.py           # Main RAG pipeline orchestrator
├── example.py                 # Example usage of modular pipeline
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
- **generator.py**: LLM-based answer generation with context
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
