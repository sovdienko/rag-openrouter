"""Embeddings module for generating vector representations"""

from typing import List, Optional
from openai import OpenAI
from .config import Config
from .cache import EmbeddingCache, CostTracker


class EmbeddingService:
    """Service for generating embeddings using OpenRouter"""

    def __init__(
        self,
        config: Config,
        cache: Optional[EmbeddingCache] = None,
        cost_tracker: Optional[CostTracker] = None
    ):
        """
        Initialize embedding service

        Args:
            config: Application configuration
            cache: Optional embedding cache for cost optimization
            cost_tracker: Optional cost tracker for monitoring API usage
        """
        self.config = config
        self.client = OpenAI(
            base_url=config.openrouter_base_url,
            api_key=config.openrouter_api_key
        )
        self.cache = cache
        self.cost_tracker = cost_tracker

    def embed_texts(self, texts: List[str] | str) -> List[List[float]]:
        """
        Generate embeddings for one or more texts

        Handles both single text (str) and multiple texts (List[str])
        Uses cache to avoid redundant API calls when available

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

        # Check cache for each text if cache is available
        results = []
        uncached_texts = []
        uncached_indices = []

        if self.cache:
            for i, text in enumerate(texts):
                cached = self.cache.get(text, self.config.embedding_model)
                if cached:
                    results.append((i, cached))
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
        else:
            uncached_texts = texts
            uncached_indices = list(range(len(texts)))

        # Make API call only for uncached texts
        if uncached_texts:
            response = self.client.embeddings.create(
                model=self.config.embedding_model,
                input=uncached_texts
            )

            # Store new embeddings in cache and results
            for i, item in enumerate(response.data):
                embedding = item.embedding
                original_index = uncached_indices[i]
                text = uncached_texts[i]

                results.append((original_index, embedding))

                # Cache the embedding
                if self.cache:
                    self.cache.set(text, self.config.embedding_model, embedding)

                # Track cost
                if self.cost_tracker:
                    self.cost_tracker.track_embedding(text, self.config.embedding_model)

        # Sort results by original index and return embeddings only
        results.sort(key=lambda x: x[0])
        return [embedding for _, embedding in results]

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
