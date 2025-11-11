"""Vector store module for Pinecone operations"""

from typing import List, Dict, Any
from pinecone import Pinecone, ServerlessSpec
from .config import Config


class VectorStore:
    """Vector store for managing embeddings in Pinecone"""

    def __init__(self, config: Config):
        """
        Initialize vector store

        Args:
            config: Application configuration
        """
        self.config = config
        self.pc = Pinecone(api_key=config.pinecone_api_key)
        self._ensure_index_exists()
        self.index = self.pc.Index(config.index_name)

    def _ensure_index_exists(self):
        """Create index if it doesn't exist"""
        if self.config.index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=self.config.index_name,
                dimension=self.config.embedding_dimension,
                metric=self.config.similarity_metric,
                spec=ServerlessSpec(
                    cloud=self.config.cloud_provider,
                    region=self.config.cloud_region
                )
            )

    def upsert_vectors(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadata: List[Dict[str, Any]]
    ) -> int:
        """
        Insert or update vectors in the store

        Args:
            texts: List of text chunks
            embeddings: List of embedding vectors
            metadata: List of metadata dictionaries

        Returns:
            Number of vectors upserted
        """
        vectors = []
        for i, (text, emb, meta) in enumerate(zip(texts, embeddings, metadata)):
            vectors.append({
                "id": f"chunk_{i}_{meta.get('source', 'unknown')}",
                "values": emb,
                "metadata": {"text": text, **meta}
            })

        self.index.upsert(vectors=vectors, namespace=self.config.namespace)
        return len(vectors)

    def query(
        self,
        query_embedding: List[float],
        top_k: int = None
    ) -> List[Dict[str, Any]]:
        """
        Query the vector store for similar vectors

        Args:
            query_embedding: Query vector
            top_k: Number of results to return

        Returns:
            List of matching results with metadata
        """
        if top_k is None:
            top_k = self.config.top_k

        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            namespace=self.config.namespace
        )

        return [
            {
                "text": match["metadata"]["text"],
                "score": match["score"],
                "metadata": {k: v for k, v in match["metadata"].items() if k != "text"}
            }
            for match in results["matches"]
        ]

    def delete_all(self):
        """Delete all vectors in the namespace"""
        self.index.delete(delete_all=True, namespace=self.config.namespace)

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        return self.index.describe_index_stats()
