"""Document classification using embedding-based similarity"""

from typing import Dict, List, Tuple, Any
import numpy as np
from .config import Config
from .embeddings import EmbeddingService


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Cosine similarity score (0-1, higher is more similar)
    """
    arr1 = np.array(vec1)
    arr2 = np.array(vec2)

    dot_product = np.dot(arr1, arr2)
    norm1 = np.linalg.norm(arr1)
    norm2 = np.linalg.norm(arr2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(dot_product / (norm1 * norm2))


class DocumentClassifier:
    """
    Example-based document classifier using embedding similarity

    Classifies documents by comparing their embeddings to example documents
    from each category, without requiring model training.

    Use cases:
    - Content categorization (technical, business, support, etc.)
    - Topic classification
    - Intent detection
    - Sentiment analysis (with positive/negative examples)
    - Language detection
    """

    def __init__(self, config: Config = None):
        """
        Initialize document classifier

        Args:
            config: Application configuration (uses env defaults if not provided)
        """
        if config is None:
            config = Config.from_env()

        self.config = config
        self.embedding_service = EmbeddingService(config)
        self.category_embeddings: Dict[str, List[List[float]]] = {}

    def add_category(self, category: str, examples: List[str]):
        """
        Add or update a category with example documents

        Args:
            category: Category name (e.g., "technical", "business")
            examples: List of example documents for this category

        Example:
            classifier.add_category("technical", [
                "Python programming guide",
                "API documentation",
                "Software architecture"
            ])
        """
        # Generate embeddings for all examples
        embeddings = self.embedding_service.embed_texts(examples)
        self.category_embeddings[category] = embeddings

    def set_categories(self, category_examples: Dict[str, List[str]]):
        """
        Set all categories at once

        Args:
            category_examples: Dictionary mapping category names to example lists

        Example:
            classifier.set_categories({
                "technical": ["API docs", "Code tutorial"],
                "business": ["Sales report", "Market analysis"],
                "support": ["How to guide", "FAQ answer"]
            })
        """
        self.category_embeddings.clear()
        for category, examples in category_examples.items():
            self.add_category(category, examples)

    def classify(
        self,
        text: str,
        return_all_scores: bool = False
    ) -> Tuple[str, float] | Dict[str, float]:
        """
        Classify a document based on similarity to category examples

        Args:
            text: Document text to classify
            return_all_scores: If True, return scores for all categories

        Returns:
            If return_all_scores=False: Tuple of (category_name, confidence_score)
            If return_all_scores=True: Dict mapping all categories to scores

        Example:
            category, confidence = classifier.classify("The new API endpoint...")
            print(f"Category: {category}, Confidence: {confidence:.3f}")

            # Or get all scores
            scores = classifier.classify("Text...", return_all_scores=True)
            for cat, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
                print(f"{cat}: {score:.3f}")
        """
        if not self.category_embeddings:
            raise ValueError("No categories defined. Use add_category() or set_categories() first.")

        # Embed the document
        doc_embedding = self.embedding_service.embed_text(text)

        # Calculate scores for each category
        category_scores = {}
        for category, example_embeddings in self.category_embeddings.items():
            # Calculate similarity to each example
            similarities = [
                cosine_similarity(doc_embedding, example_emb)
                for example_emb in example_embeddings
            ]

            # Average similarity across all examples
            category_scores[category] = sum(similarities) / len(similarities)

        if return_all_scores:
            return category_scores

        # Return highest scoring category
        best_category = max(category_scores.items(), key=lambda x: x[1])
        return best_category

    def classify_batch(
        self,
        texts: List[str],
        return_all_scores: bool = False
    ) -> List[Tuple[str, float]] | List[Dict[str, float]]:
        """
        Classify multiple documents efficiently

        Args:
            texts: List of document texts to classify
            return_all_scores: If True, return scores for all categories

        Returns:
            List of classification results (same format as classify())

        Example:
            documents = ["API guide", "Sales report", "Bug fix instructions"]
            results = classifier.classify_batch(documents)
            for doc, (category, conf) in zip(documents, results):
                print(f"{doc[:30]}... -> {category} ({conf:.3f})")
        """
        if not self.category_embeddings:
            raise ValueError("No categories defined. Use add_category() or set_categories() first.")

        # Embed all documents at once (more efficient)
        doc_embeddings = self.embedding_service.embed_texts(texts)

        results = []
        for doc_embedding in doc_embeddings:
            # Calculate scores for each category
            category_scores = {}
            for category, example_embeddings in self.category_embeddings.items():
                similarities = [
                    cosine_similarity(doc_embedding, example_emb)
                    for example_emb in example_embeddings
                ]
                category_scores[category] = sum(similarities) / len(similarities)

            if return_all_scores:
                results.append(category_scores)
            else:
                best_category = max(category_scores.items(), key=lambda x: x[1])
                results.append(best_category)

        return results

    def get_categories(self) -> List[str]:
        """Get list of defined categories"""
        return list(self.category_embeddings.keys())

    def remove_category(self, category: str):
        """Remove a category"""
        if category in self.category_embeddings:
            del self.category_embeddings[category]
