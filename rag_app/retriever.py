"""Retriever module for semantic search"""

from typing import List, Dict, Any, Optional
from .config import Config
from .embeddings import EmbeddingService
from .vector_store import VectorStore
from .reranker import Reranker
from .hybrid_search import HybridSearcher


class Retriever:
    """Retriever for finding relevant document chunks"""

    def __init__(
        self,
        config: Config,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
        reranker: Optional[Reranker] = None,
        hybrid_searcher: Optional[HybridSearcher] = None
    ):
        """
        Initialize retriever

        Args:
            config: Application configuration
            embedding_service: Service for generating embeddings
            vector_store: Vector store for similarity search
            reranker: Optional reranker for two-stage retrieval
            hybrid_searcher: Optional hybrid searcher for keyword+vector search
        """
        self.config = config
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.reranker = reranker
        self.hybrid_searcher = hybrid_searcher

    def retrieve(
        self,
        query: str,
        top_k: int = None,
        use_reranking: bool = False,
        use_hybrid: bool = False,
        initial_k: int = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant document chunks for a query

        Supports multiple retrieval strategies:
        - Vector similarity search (default)
        - Two-stage retrieval with reranking
        - Hybrid search (vector + keyword)
        - Combination of hybrid + reranking

        Args:
            query: Search query
            top_k: Number of results to return (final count)
            use_reranking: Whether to use two-stage retrieval with reranking
            use_hybrid: Whether to use hybrid search (vector + keyword)
            initial_k: Number of candidates to retrieve before reranking/hybrid
                      (default: top_k * 5 if reranking/hybrid, else top_k)

        Returns:
            List of relevant chunks with metadata and scores

        Examples:
            # Standard vector retrieval
            results = retriever.retrieve(query, top_k=3)

            # Hybrid search (vector + keyword)
            results = retriever.retrieve(query, top_k=3, use_hybrid=True, initial_k=20)

            # Two-stage retrieval with reranking
            results = retriever.retrieve(query, top_k=3, use_reranking=True, initial_k=20)

            # Hybrid + reranking (best quality)
            results = retriever.retrieve(query, top_k=3, use_hybrid=True, use_reranking=True, initial_k=20)
        """
        # Determine retrieval count
        if use_reranking or use_hybrid:
            if initial_k is None:
                initial_k = (top_k or self.config.top_k) * 5
        else:
            initial_k = top_k

        # Stage 1: Vector retrieval
        query_embedding = self.embedding_service.embed_text(query)
        vector_results = self.vector_store.query(query_embedding, top_k=initial_k)

        # Stage 2: Hybrid search (if enabled)
        if use_hybrid and self.hybrid_searcher:
            vector_results = self.hybrid_searcher.search(query, vector_results, top_k=initial_k)

        # Stage 3: Reranking (if enabled)
        if use_reranking and self.reranker:
            vector_results = self.reranker.rerank(query, vector_results, top_k=top_k)
        elif top_k:
            # Limit to top_k if not using reranking
            vector_results = vector_results[:top_k]

        return vector_results

    def retrieve_texts(
        self,
        query: str,
        top_k: int = None,
        use_reranking: bool = False,
        use_hybrid: bool = False,
        initial_k: int = None
    ) -> List[str]:
        """
        Retrieve only the text of relevant chunks

        Args:
            query: Search query
            top_k: Number of results to return
            use_reranking: Whether to use two-stage retrieval
            use_hybrid: Whether to use hybrid search
            initial_k: Number of candidates for reranking/hybrid

        Returns:
            List of relevant text chunks
        """
        results = self.retrieve(
            query,
            top_k=top_k,
            use_reranking=use_reranking,
            use_hybrid=use_hybrid,
            initial_k=initial_k
        )
        return [result["text"] for result in results]
