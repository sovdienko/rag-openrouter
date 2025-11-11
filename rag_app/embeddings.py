"""Embeddings module for generating vector representations"""

from typing import List
from openai import OpenAI
from .config import Config


class EmbeddingService:
    """Service for generating embeddings using OpenRouter"""

    def __init__(self, config: Config):
        """
        Initialize embedding service

        Args:
            config: Application configuration
        """
        self.config = config
        self.client = OpenAI(
            base_url=config.openrouter_base_url,
            api_key=config.openrouter_api_key
        )

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors
        """
        response = self.client.embeddings.create(
            model=self.config.embedding_model,
            input=texts
        )
        return [item.embedding for item in response.data]

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text

        Args:
            text: Text string to embed

        Returns:
            Embedding vector
        """
        response = self.client.embeddings.create(
            model=self.config.embedding_model,
            input=text
        )
        return response.data[0].embedding
