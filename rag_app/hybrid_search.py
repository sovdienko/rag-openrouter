"""Hybrid search combining vector similarity with keyword matching (BM25)"""

from typing import List, Dict, Any
from rank_bm25 import BM25Okapi


class HybridSearcher:
    """
    Hybrid search that combines vector similarity with BM25 keyword matching

    Particularly effective for:
    - Queries with specific terms or proper nouns
    - Domain-specific terminology
    - Exact phrase matching
    - Technical documentation retrieval
    """

    def __init__(self, alpha: float = 0.5):
        """
        Initialize hybrid searcher

        Args:
            alpha: Weight for vector search (0-1)
                  - alpha=1.0: Pure vector search
                  - alpha=0.0: Pure keyword search
                  - alpha=0.5: Balanced hybrid (default)
        """
        self.alpha = alpha
        self.bm25 = None
        self.documents = None

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for BM25

        Args:
            text: Input text

        Returns:
            List of lowercase tokens
        """
        return text.lower().split()

    def index_documents(self, documents: List[Dict[str, Any]]):
        """
        Index documents for BM25 keyword search

        Args:
            documents: List of document dicts with 'text' key
        """
        self.documents = documents
        tokenized_docs = [self._tokenize(doc["text"]) for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)

    def search(
        self,
        query: str,
        vector_results: List[Dict[str, Any]],
        top_k: int = None
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining vector and keyword scores

        Args:
            query: Search query
            vector_results: Results from vector similarity search with scores
            top_k: Number of results to return (default: len(vector_results))

        Returns:
            Combined and re-ranked results

        Example:
            # Get vector results first
            vector_results = vector_store.query(embedding, top_k=20)

            # Apply hybrid search
            hybrid_results = hybrid_searcher.search(query, vector_results, top_k=5)
        """
        if not vector_results:
            return []

        if top_k is None:
            top_k = len(vector_results)

        # Index the vector results for BM25
        self.index_documents(vector_results)

        # Get BM25 keyword scores
        query_tokens = self._tokenize(query)
        keyword_scores = self.bm25.get_scores(query_tokens)

        # Normalize and combine scores
        combined_results = []
        max_keyword_score = max(keyword_scores) if max(keyword_scores) > 0 else 1.0

        for i, result in enumerate(vector_results):
            # Normalize vector score (0-1, higher is better)
            # Pinecone uses cosine similarity (higher is better)
            vec_score = result.get("score", 0.0)
            vec_score_norm = vec_score  # Already 0-1 range for cosine

            # Normalize keyword score (0-1)
            kw_score_norm = keyword_scores[i] / max_keyword_score

            # Combine scores with alpha weighting
            final_score = self.alpha * vec_score_norm + (1 - self.alpha) * kw_score_norm

            # Add hybrid score to result
            result_copy = result.copy()
            result_copy["hybrid_score"] = final_score
            result_copy["vector_score"] = vec_score
            result_copy["keyword_score"] = kw_score_norm
            combined_results.append(result_copy)

        # Sort by hybrid score (descending)
        combined_results.sort(key=lambda x: x["hybrid_score"], reverse=True)

        return combined_results[:top_k]

    def set_alpha(self, alpha: float):
        """
        Update the alpha parameter for search weighting

        Args:
            alpha: New alpha value (0-1)
        """
        if not 0 <= alpha <= 1:
            raise ValueError("Alpha must be between 0 and 1")
        self.alpha = alpha
