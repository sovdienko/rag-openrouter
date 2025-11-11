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

### Full RAG Pipeline

Run the complete RAG pipeline with Pinecone vector storage:

```bash
python rag-pipeline-pinecone.py
```

This script will:
1. Chunk sample documents into smaller pieces
2. Generate embeddings for each chunk
3. Store vectors in Pinecone
4. Retrieve relevant chunks for a query
5. Generate an answer using Llama 3.3 70B

### Document Chunking Only

To test document chunking independently:

```bash
python chunk.py
```

### Basic Embedding Generation

Run the basic embedding example:

```bash
python sample1.py
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
├── rag-pipeline-pinecone.py  # Complete RAG pipeline implementation
├── chunk.py                   # Document chunking and ingestion script
├── sample1.py                 # Basic embedding generation example
├── pyproject.toml             # Project configuration and dependencies
├── .env                       # Environment variables (not tracked in git)
└── README.md                  # This file
```

## Configuration

Key parameters in `rag-pipeline-pinecone.py`:

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
