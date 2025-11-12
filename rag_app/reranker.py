"""Reranker module for improving retrieval precision"""

from typing import List, Dict, Any
from sentence_transformers import CrossEncoder
import numpy as np


class Reranker:
    """Reranker for improving retrieval quality using cross-encoder models"""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize reranker

        Args:
            model_name: HuggingFace model name for cross-encoder
                       Default: ms-marco-MiniLM-L-6-v2 (fast, good quality)
                       Alternatives:
                       - cross-encoder/ms-marco-MiniLM-L-12-v2 (better quality)
                       - cross-encoder/ms-marco-TinyBERT-L-2-v2 (faster)
        """
        self.model_name = model_name
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: int = None
    ) -> List[Dict[str, Any]]:
        """
        Rerank search results using cross-encoder

        Args:
            query: Search query
            results: List of search results with 'text' key
            top_k: Number of results to return (default: return all, reranked)

        Returns:
            Reranked list of results with added 'rerank_score' field
        """
        if not results:
            return results

        # Create query-document pairs
        pairs = [[query, result["text"]] for result in results]

        # Get reranking scores
        scores = self.model.predict(pairs)

        # Add scores to results
        for i, result in enumerate(results):
            result["rerank_score"] = float(scores[i])

        # Sort by rerank score (descending)
        reranked = sorted(results, key=lambda x: x["rerank_score"], reverse=True)

        # Return top K if specified
        if top_k is not None:
            reranked = reranked[:top_k]

        return reranked

    def rerank_texts(
        self,
        query: str,
        texts: List[str],
        top_k: int = None
    ) -> tuple[List[str], List[float]]:
        """
        Rerank texts and return sorted texts with scores

        Args:
            query: Search query
            texts: List of text strings
            top_k: Number of results to return

        Returns:
            Tuple of (reranked_texts, scores)
        """
        if not texts:
            return [], []

        # Create query-text pairs
        pairs = [[query, text] for text in texts]

        # Get reranking scores
        scores = self.model.predict(pairs)

        # Sort by score (descending)
        sorted_indices = np.argsort(scores)[::-1]

        if top_k is not None:
            sorted_indices = sorted_indices[:top_k]

        reranked_texts = [texts[i] for i in sorted_indices]
        reranked_scores = [float(scores[i]) for i in sorted_indices]

        return reranked_texts, reranked_scores
