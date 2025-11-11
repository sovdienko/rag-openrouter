"""Retriever module for semantic search"""

from typing import List, Dict, Any
from .config import Config
from .embeddings import EmbeddingService
from .vector_store import VectorStore


class Retriever:
    """Retriever for finding relevant document chunks"""

    def __init__(
        self,
        config: Config,
        embedding_service: EmbeddingService,
        vector_store: VectorStore
    ):
        """
        Initialize retriever

        Args:
            config: Application configuration
            embedding_service: Service for generating embeddings
            vector_store: Vector store for similarity search
        """
        self.config = config
        self.embedding_service = embedding_service
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant document chunks for a query

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of relevant chunks with metadata and scores
        """
        # Generate query embedding
        query_embedding = self.embedding_service.embed_text(query)

        # Search vector store
        results = self.vector_store.query(query_embedding, top_k=top_k)

        return results

    def retrieve_texts(self, query: str, top_k: int = None) -> List[str]:
        """
        Retrieve only the text of relevant chunks

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of relevant text chunks
        """
        results = self.retrieve(query, top_k=top_k)
        return [result["text"] for result in results]
