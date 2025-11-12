"""Configuration module for RAG application"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class Config:
    """Application configuration"""

    # API Keys
    openrouter_api_key: str
    pinecone_api_key: str

    # OpenRouter settings
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    embedding_model: str = "openai/text-embedding-3-large"
    llm_model: str = "meta-llama/llama-3.3-70b-instruct"

    # Pinecone settings
    index_name: str = "rag-doc-classification"
    embedding_dimension: int = 3072
    similarity_metric: str = "cosine"
    cloud_provider: str = "aws"
    cloud_region: str = "us-east-1"
    namespace: str = "default"

    # Document processing settings
    chunk_size: int = 1000
    chunk_overlap: int = 200
    separators: list = None

    # Retrieval settings
    top_k: int = 3

    # Generation settings
    temperature: float = 0.7
    max_tokens: int = 512

    def __post_init__(self):
        if self.separators is None:
            self.separators = ["\n\n", "\n", " ", ""]

    @classmethod
    def from_env(cls):
        """Create configuration from environment variables"""
        return cls(
            openrouter_api_key=os.environ.get("OPENROUTER_API_KEY", ""),
            pinecone_api_key=os.environ.get("PINECONE_API_KEY", "")
        )

    def validate(self):
        """Validate required configuration"""
        if not self.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment variables")
        if not self.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY not found in environment variables")
