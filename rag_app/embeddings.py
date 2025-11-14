"""Embeddings module for generating vector representations"""

from typing import List, Optional
from openai import OpenAI
from .config import Config
from .cache import EmbeddingCache, CostTracker
from .rate_limiting import rate_limit_retry, RateLimiter


class EmbeddingService:
    """Service for generating embeddings using OpenRouter"""

    def __init__(
        self,
        config: Config,
        cache: Optional[EmbeddingCache] = None,
        cost_tracker: Optional[CostTracker] = None,
        rate_limiter: Optional[RateLimiter] = None,
        use_retry: bool = True,
        max_retries: int = 3
    ):
        """
        Initialize embedding service

        Args:
            config: Application configuration
            cache: Optional embedding cache for cost optimization
            cost_tracker: Optional cost tracker for monitoring API usage
            rate_limiter: Optional rate limiter to control request frequency
            use_retry: Enable automatic retry with exponential backoff (default: True)
            max_retries: Maximum retry attempts for failed requests (default: 3)
        """
        self.config = config
        self.client = OpenAI(
            base_url=config.openrouter_base_url,
            api_key=config.openrouter_api_key
        )
        self.cache = cache
        self.cost_tracker = cost_tracker
        self.rate_limiter = rate_limiter
        self.use_retry = use_retry
        self.max_retries = max_retries

    def _call_api(self, texts: List[str]) -> List[List[float]]:
        """
        Internal method to call embedding API with optional rate limiting and retry

        Args:
            texts: List of texts to embed

        Returns:
            List of embeddings
        """
        # Apply rate limiting if configured
        if self.rate_limiter:
            self.rate_limiter.wait_if_needed()

        # Define the API call function
        def make_api_call():
            response = self.client.embeddings.create(
                model=self.config.embedding_model,
                input=texts
            )
            return [item.embedding for item in response.data]

        # Apply retry decorator if enabled
        if self.use_retry:
            decorated_call = rate_limit_retry(
                max_retries=self.max_retries,
                initial_wait=1.0,
                max_wait=60.0,
                backoff_factor=2.0
            )(make_api_call)
            return decorated_call()
        else:
            return make_api_call()

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
            # Use _call_api which handles rate limiting and retries
            embeddings = self._call_api(uncached_texts)

            # Store new embeddings in cache and results
            for i, embedding in enumerate(embeddings):
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

    def batch_embed_documents(
        self,
        texts: List[str],
        batch_size: int = 100,
        show_progress: bool = False
    ) -> List[List[float]]:
        """
        Process large document collections in batches for efficiency

        Reduces API overhead by embedding multiple texts per request.
        Automatically handles caching and deduplication.

        Args:
            texts: List of text strings to embed
            batch_size: Number of texts per API request (default: 100)
            show_progress: Print progress updates (default: False)

        Returns:
            List of embedding vectors in same order as input

        Examples:
            # Process 1000 documents in batches of 100
            embeddings = service.batch_embed_documents(
                texts=documents,
                batch_size=100,
                show_progress=True
            )

        Note:
            - OpenRouter accepts arrays of up to several hundred texts per request
            - Larger batch sizes reduce API overhead but may hit size limits
            - Recommended batch_size: 50-200 depending on text length
            - Caching happens automatically per text, not per batch
        """
        all_embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size

        for batch_idx in range(0, len(texts), batch_size):
            batch = texts[batch_idx:batch_idx + batch_size]

            if show_progress:
                current_batch = (batch_idx // batch_size) + 1
                print(f"Processing batch {current_batch}/{total_batches} "
                      f"({len(batch)} texts)...")

            # Use existing embed_texts which handles caching automatically
            batch_embeddings = self.embed_texts(batch)
            all_embeddings.extend(batch_embeddings)

        if show_progress:
            if self.cache:
                stats = self.cache.get_stats()
                print(f"\nBatch processing complete!")
                print(f"Total texts: {len(texts)}")
                print(f"Cache hits: {stats['hits']}")
                print(f"Cache misses: {stats['misses']}")
                print(f"Hit rate: {stats['hit_rate']:.1f}%")

        return all_embeddings
