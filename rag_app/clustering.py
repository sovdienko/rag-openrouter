"""Document clustering using embedding-based similarity"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from .config import Config
from .embeddings import EmbeddingService


class DocumentClusterer:
    """
    Automatically discover groups in unlabeled documents using embeddings

    Use cases:
    - Identify emerging topics in customer feedback
    - Group similar bug reports or support tickets
    - Organize large document collections by theme
    - Discover patterns in unstructured text data
    - Content analysis and topic discovery
    """

    def __init__(self, config: Config = None):
        """
        Initialize document clusterer

        Args:
            config: Application configuration (uses env defaults if not provided)
        """
        if config is None:
            config = Config.from_env()

        self.config = config
        self.embedding_service = EmbeddingService(config)
        self.embeddings: Optional[np.ndarray] = None
        self.labels: Optional[np.ndarray] = None
        self.kmeans: Optional[KMeans] = None
        self.texts: Optional[List[str]] = None

    def cluster(
        self,
        texts: List[str],
        n_clusters: int = 5,
        random_state: int = 42
    ) -> Dict[int, List[str]]:
        """
        Cluster documents into groups using KMeans

        Args:
            texts: List of document texts to cluster
            n_clusters: Number of clusters to create
            random_state: Random seed for reproducibility

        Returns:
            Dictionary mapping cluster IDs to lists of documents

        Example:
            clusterer = DocumentClusterer()
            clusters = clusterer.cluster(feedback_texts, n_clusters=3)
            for cluster_id, items in clusters.items():
                print(f"Cluster {cluster_id}: {len(items)} items")
        """
        if len(texts) < n_clusters:
            raise ValueError(f"Number of texts ({len(texts)}) must be >= n_clusters ({n_clusters})")

        # Store texts for later use
        self.texts = texts

        # Generate embeddings
        embeddings_list = self.embedding_service.embed_texts(texts)
        self.embeddings = np.array(embeddings_list)

        # Perform clustering
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        self.labels = self.kmeans.fit_predict(self.embeddings)

        # Organize results into clusters
        clusters = {i: [] for i in range(n_clusters)}
        for text, label in zip(texts, self.labels):
            clusters[int(label)].append(text)

        return clusters

    def cluster_with_metadata(
        self,
        texts: List[str],
        n_clusters: int = 5,
        random_state: int = 42
    ) -> Dict[int, List[Dict[str, Any]]]:
        """
        Cluster documents and return with additional metadata

        Args:
            texts: List of document texts to cluster
            n_clusters: Number of clusters to create
            random_state: Random seed for reproducibility

        Returns:
            Dictionary mapping cluster IDs to lists of document dicts with metadata

        Example:
            clusters = clusterer.cluster_with_metadata(texts, n_clusters=3)
            for cluster_id, items in clusters.items():
                for item in items:
                    print(f"Text: {item['text'][:50]}")
                    print(f"Distance from center: {item['distance']:.4f}")
        """
        # Perform basic clustering first
        _ = self.cluster(texts, n_clusters, random_state)

        # Calculate distances from cluster centers
        distances = self.kmeans.transform(self.embeddings)

        # Organize results with metadata
        clusters = {i: [] for i in range(n_clusters)}
        for idx, (text, label) in enumerate(zip(texts, self.labels)):
            cluster_id = int(label)
            distance_to_center = distances[idx][cluster_id]

            clusters[cluster_id].append({
                'text': text,
                'index': idx,
                'distance': distance_to_center,
                'cluster': cluster_id
            })

        # Sort by distance from center (closest first)
        for cluster_id in clusters:
            clusters[cluster_id].sort(key=lambda x: x['distance'])

        return clusters

    def get_cluster_summaries(
        self,
        clusters: Dict[int, List[str]],
        top_n: int = 3
    ) -> Dict[int, Dict[str, Any]]:
        """
        Get summary statistics for each cluster

        Args:
            clusters: Clusters from cluster() or cluster_with_metadata()
            top_n: Number of representative examples to include

        Returns:
            Dictionary with cluster summaries including size and examples

        Example:
            clusters = clusterer.cluster(texts, n_clusters=3)
            summaries = clusterer.get_cluster_summaries(clusters)
            for cluster_id, summary in summaries.items():
                print(f"Cluster {cluster_id}: {summary['size']} items")
                print(f"Examples: {summary['examples']}")
        """
        summaries = {}

        for cluster_id, items in clusters.items():
            # Handle both string lists and dict lists
            if items and isinstance(items[0], dict):
                texts_in_cluster = [item['text'] for item in items]
                # Items already sorted by distance in cluster_with_metadata
                examples = texts_in_cluster[:top_n]
            else:
                texts_in_cluster = items
                examples = items[:top_n]

            summaries[cluster_id] = {
                'cluster_id': cluster_id,
                'size': len(items),
                'examples': examples,
                'percentage': len(items) / len(self.texts) * 100 if self.texts else 0
            }

        return summaries

    def predict_cluster(self, text: str) -> Tuple[int, float]:
        """
        Predict which cluster a new document belongs to

        Args:
            text: New document text

        Returns:
            Tuple of (cluster_id, distance_to_center)

        Example:
            # After clustering
            cluster_id, distance = clusterer.predict_cluster("New feedback text")
            print(f"Belongs to cluster {cluster_id} (distance: {distance:.4f})")
        """
        if self.kmeans is None:
            raise ValueError("Must call cluster() before predict_cluster()")

        # Embed the new text
        embedding = self.embedding_service.embed_text(text)
        embedding_array = np.array([embedding])

        # Predict cluster
        cluster_id = int(self.kmeans.predict(embedding_array)[0])

        # Calculate distance to cluster center
        distances = self.kmeans.transform(embedding_array)
        distance = float(distances[0][cluster_id])

        return cluster_id, distance

    def find_optimal_clusters(
        self,
        texts: List[str],
        min_clusters: int = 2,
        max_clusters: int = 10,
        method: str = 'silhouette'
    ) -> Dict[int, float]:
        """
        Find optimal number of clusters using silhouette analysis

        Args:
            texts: List of document texts
            min_clusters: Minimum number of clusters to try
            max_clusters: Maximum number of clusters to try
            method: Scoring method ('silhouette' or 'inertia')

        Returns:
            Dictionary mapping number of clusters to scores

        Example:
            scores = clusterer.find_optimal_clusters(texts, min_clusters=2, max_clusters=8)
            best_k = max(scores.items(), key=lambda x: x[1])[0]
            print(f"Optimal clusters: {best_k}")
        """
        if len(texts) < max_clusters:
            max_clusters = len(texts) - 1

        # Generate embeddings once
        embeddings_list = self.embedding_service.embed_texts(texts)
        embeddings_array = np.array(embeddings_list)

        scores = {}
        for k in range(min_clusters, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embeddings_array)

            if method == 'silhouette':
                # Higher is better
                score = silhouette_score(embeddings_array, labels)
            elif method == 'inertia':
                # Lower is better, so negate it
                score = -kmeans.inertia_
            else:
                raise ValueError(f"Unknown method: {method}")

            scores[k] = float(score)

        return scores

    def get_cluster_centers(self) -> Optional[np.ndarray]:
        """
        Get cluster center embeddings

        Returns:
            Array of cluster center vectors, or None if not clustered yet
        """
        if self.kmeans is None:
            return None
        return self.kmeans.cluster_centers_

    def get_cluster_labels(self) -> Optional[np.ndarray]:
        """Get cluster labels for all documents"""
        return self.labels

    def get_cluster_sizes(self) -> Optional[Dict[int, int]]:
        """
        Get size of each cluster

        Returns:
            Dictionary mapping cluster IDs to sizes
        """
        if self.labels is None:
            return None

        unique, counts = np.unique(self.labels, return_counts=True)
        return {int(k): int(v) for k, v in zip(unique, counts)}
