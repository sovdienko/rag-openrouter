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

    def embed_texts(self, texts: List[str] | str) -> List[List[float]]:
        """
        Generate embeddings for one or more texts

        Handles both single text (str) and multiple texts (List[str])

        Args:
            texts: Single text string or list of text strings

        Returns:
            List of embedding vectors (always returns a list, even for single text)

        Examples:
            # Single text
            embeddings = service.embed_texts("Hello world")
            # Returns: [[0.1, 0.2, ...]]

            # Multiple texts
            embeddings = service.embed_texts(["Text 1", "Text 2"])
            # Returns: [[0.1, 0.2, ...], [0.3, 0.4, ...]]
        """
        # Normalize input to list
        if isinstance(texts, str):
            texts = [texts]

        response = self.client.embeddings.create(
            model=self.config.embedding_model,
            input=texts
        )
        return [item.embedding for item in response.data]

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text

        Convenience method that returns a single vector instead of a list

        Args:
            text: Text string to embed

        Returns:
            Single embedding vector
        """
        return self.embed_texts(text)[0]
