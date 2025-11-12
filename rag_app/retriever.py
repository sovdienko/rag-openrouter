"""Retriever module for semantic search"""

from typing import List, Dict, Any, Optional
from .config import Config
from .embeddings import EmbeddingService
from .vector_store import VectorStore
from .reranker import Reranker


class Retriever:
    """Retriever for finding relevant document chunks"""

    def __init__(
        self,
        config: Config,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
        reranker: Optional[Reranker] = None
    ):
        """
        Initialize retriever

        Args:
            config: Application configuration
            embedding_service: Service for generating embeddings
            vector_store: Vector store for similarity search
            reranker: Optional reranker for two-stage retrieval
        """
        self.config = config
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.reranker = reranker

    def retrieve(
        self,
        query: str,
        top_k: int = None,
        use_reranking: bool = False,
        initial_k: int = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant document chunks for a query

        Supports two-stage retrieval: broad initial retrieval + reranking

        Args:
            query: Search query
            top_k: Number of results to return (final count)
            use_reranking: Whether to use two-stage retrieval with reranking
            initial_k: Number of candidates to retrieve before reranking
                      (default: top_k * 5 if reranking, else top_k)

        Returns:
            List of relevant chunks with metadata and scores

        Examples:
            # Standard retrieval
            results = retriever.retrieve(query, top_k=3)

            # Two-stage retrieval (retrieve 20, rerank to top 3)
            results = retriever.retrieve(query, top_k=3, use_reranking=True, initial_k=20)
        """
        # Determine retrieval count
        if use_reranking and self.reranker:
            # Stage 1: Broad retrieval
            if initial_k is None:
                initial_k = (top_k or self.config.top_k) * 5

            query_embedding = self.embedding_service.embed_text(query)
            initial_results = self.vector_store.query(query_embedding, top_k=initial_k)

            # Stage 2: Rerank
            reranked_results = self.reranker.rerank(query, initial_results, top_k=top_k)
            return reranked_results
        else:
            # Single-stage retrieval
            query_embedding = self.embedding_service.embed_text(query)
            results = self.vector_store.query(query_embedding, top_k=top_k)
            return results

    def retrieve_texts(
        self,
        query: str,
        top_k: int = None,
        use_reranking: bool = False,
        initial_k: int = None
    ) -> List[str]:
        """
        Retrieve only the text of relevant chunks

        Args:
            query: Search query
            top_k: Number of results to return
            use_reranking: Whether to use two-stage retrieval
            initial_k: Number of candidates for reranking

        Returns:
            List of relevant text chunks
        """
        results = self.retrieve(query, top_k=top_k, use_reranking=use_reranking, initial_k=initial_k)
        return [result["text"] for result in results]
