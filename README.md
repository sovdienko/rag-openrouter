# RAG OpenRouter

A Python project demonstrating Retrieval-Augmented Generation (RAG) using OpenRouter's API for generating text embeddings.

## Description

This project provides a simple interface to generate embeddings for documents using OpenRouter's API, which serves as a gateway to various AI models including OpenAI's text-embedding models.

## Features

- Generate embeddings for single or multiple text documents
- Integration with OpenRouter API
- Support for OpenAI's text-embedding-3-large model
- Environment-based configuration for API keys

## Requirements

- Python >= 3.11
- OpenAI Python SDK >= 2.7.2
- python-dotenv >= 1.0.0

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

3. Create a `.env` file in the project root and add your OpenRouter API key:
   ```env
   OPENROUTER_API_KEY=your_api_key_here
   ```

## Getting Your OpenRouter API Key

1. Visit [OpenRouter](https://openrouter.ai/)
2. Sign up or log in to your account
3. Navigate to the API Keys section
4. Generate a new API key
5. Add it to your `.env` file

## Usage

Run the sample script to generate embeddings:

```bash
python sample1.py
```

## Project Structure

```
rag-openrouter/
├── sample1.py          # Example script demonstrating embedding generation
├── pyproject.toml      # Project configuration and dependencies
├── .env                # Environment variables (not tracked in git)
└── README.md           # This file
```

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
