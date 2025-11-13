"""Embedding cache for cost optimization"""

import hashlib
import json
import os
from typing import List, Dict, Any, Optional
from pathlib import Path


class EmbeddingCache:
    """
    Cache embeddings to avoid redundant API calls

    Reduces costs by storing and reusing embeddings for
    frequently accessed content. Supports in-memory and
    disk-based caching strategies.

    Use cases:
    - Development and testing (avoid repeated API calls)
    - Production with frequently queried content
    - Cost optimization for large-scale applications
    - Offline operation after initial embedding generation
    """

    def __init__(self, cache_dir: Optional[str] = None, memory_only: bool = False):
        """
        Initialize embedding cache

        Args:
            cache_dir: Directory for disk-based cache (default: .cache/embeddings)
            memory_only: If True, only use in-memory cache (faster, not persistent)
        """
        self.memory_cache: Dict[str, List[float]] = {}
        self.memory_only = memory_only
        self.hit_count = 0
        self.miss_count = 0

        if not memory_only:
            self.cache_dir = Path(cache_dir or ".cache/embeddings")
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.cache_dir = None

    def _get_cache_key(self, text: str, model: str) -> str:
        """
        Generate cache key from text and model

        Args:
            text: Input text
            model: Model name

        Returns:
            MD5 hash as cache key
        """
        content = f"{model}:{text}"
        return hashlib.md5(content.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get file path for cached embedding"""
        # Use subdirectories to avoid too many files in one directory
        subdir = cache_key[:2]
        return self.cache_dir / subdir / f"{cache_key}.json"

    def get(self, text: str, model: str) -> Optional[List[float]]:
        """
        Get cached embedding if available

        Args:
            text: Input text
            model: Model name

        Returns:
            Cached embedding or None if not found
        """
        cache_key = self._get_cache_key(text, model)

        # Check memory cache first
        if cache_key in self.memory_cache:
            self.hit_count += 1
            return self.memory_cache[cache_key]

        # Check disk cache if enabled
        if not self.memory_only and self.cache_dir:
            cache_path = self._get_cache_path(cache_key)
            if cache_path.exists():
                try:
                    with open(cache_path, 'r') as f:
                        data = json.load(f)
                        embedding = data['embedding']
                        # Load into memory cache
                        self.memory_cache[cache_key] = embedding
                        self.hit_count += 1
                        return embedding
                except Exception as e:
                    print(f"Error loading cache: {e}")

        self.miss_count += 1
        return None

    def set(self, text: str, model: str, embedding: List[float]):
        """
        Store embedding in cache

        Args:
            text: Input text
            model: Model name
            embedding: Embedding vector
        """
        cache_key = self._get_cache_key(text, model)

        # Store in memory
        self.memory_cache[cache_key] = embedding

        # Store on disk if enabled
        if not self.memory_only and self.cache_dir:
            cache_path = self._get_cache_path(cache_key)
            cache_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                with open(cache_path, 'w') as f:
                    json.dump({
                        'text': text[:100],  # Store snippet for debugging
                        'model': model,
                        'embedding': embedding
                    }, f)
            except Exception as e:
                print(f"Error saving cache: {e}")

    def clear_memory(self):
        """Clear in-memory cache"""
        self.memory_cache.clear()

    def clear_disk(self):
        """Clear disk-based cache"""
        if self.cache_dir and self.cache_dir.exists():
            import shutil
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def clear_all(self):
        """Clear both memory and disk caches"""
        self.clear_memory()
        if not self.memory_only:
            self.clear_disk()
        self.hit_count = 0
        self.miss_count = 0

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics

        Returns:
            Dictionary with cache statistics
        """
        total = self.hit_count + self.miss_count
        hit_rate = (self.hit_count / total * 100) if total > 0 else 0

        stats = {
            'hits': self.hit_count,
            'misses': self.miss_count,
            'total_requests': total,
            'hit_rate': hit_rate,
            'memory_size': len(self.memory_cache)
        }

        if not self.memory_only and self.cache_dir:
            # Count disk cache files
            disk_count = sum(1 for _ in self.cache_dir.rglob('*.json'))
            stats['disk_size'] = disk_count

        return stats


class CostTracker:
    """
    Track API costs for embeddings and LLM calls

    Helps monitor and optimize spending on OpenRouter API.
    """

    # Cost per million tokens (as of 2025)
    EMBEDDING_COSTS = {
        "openai/text-embedding-3-small": 0.02,  # per 1M tokens
        "openai/text-embedding-3-large": 0.13,  # per 1M tokens
        "text-embedding-3-small": 0.02,
        "text-embedding-3-large": 0.13,
    }

    LLM_COSTS = {
        "meta-llama/llama-3.3-70b-instruct": {"input": 0.88, "output": 0.88},  # per 1M tokens
        "meta-llama/llama-3.3-8b-instruct:free": {"input": 0.0, "output": 0.0},  # Free tier
    }

    def __init__(self):
        """Initialize cost tracker"""
        self.embedding_tokens = 0
        self.llm_input_tokens = 0
        self.llm_output_tokens = 0
        self.embedding_calls = 0
        self.llm_calls = 0
        self.embedding_model = "openai/text-embedding-3-small"
        self.llm_model = "meta-llama/llama-3.3-70b-instruct"

    def track_embedding(self, text: str | List[str], model: str = None):
        """
        Track embedding API call

        Args:
            text: Input text or list of texts
            model: Embedding model used
        """
        if model:
            self.embedding_model = model

        # Estimate tokens (rough: 1 token ≈ 4 characters)
        if isinstance(text, str):
            tokens = len(text) // 4
        else:
            tokens = sum(len(t) // 4 for t in text)

        self.embedding_tokens += tokens
        self.embedding_calls += 1

    def track_llm(self, input_text: str, output_text: str, model: str = None):
        """
        Track LLM API call

        Args:
            input_text: Input prompt
            output_text: Generated output
            model: LLM model used
        """
        if model:
            self.llm_model = model

        # Estimate tokens
        self.llm_input_tokens += len(input_text) // 4
        self.llm_output_tokens += len(output_text) // 4
        self.llm_calls += 1

    def get_embedding_cost(self) -> float:
        """Calculate total embedding cost"""
        cost_per_million = self.EMBEDDING_COSTS.get(self.embedding_model, 0.02)
        return (self.embedding_tokens / 1_000_000) * cost_per_million

    def get_llm_cost(self) -> float:
        """Calculate total LLM cost"""
        costs = self.LLM_COSTS.get(self.llm_model, {"input": 0.88, "output": 0.88})
        input_cost = (self.llm_input_tokens / 1_000_000) * costs["input"]
        output_cost = (self.llm_output_tokens / 1_000_000) * costs["output"]
        return input_cost + output_cost

    def get_total_cost(self) -> float:
        """Calculate total cost (embeddings + LLM)"""
        return self.get_embedding_cost() + self.get_llm_cost()

    def get_report(self) -> Dict[str, Any]:
        """
        Get detailed cost report

        Returns:
            Dictionary with cost breakdown
        """
        return {
            'embeddings': {
                'model': self.embedding_model,
                'calls': self.embedding_calls,
                'tokens': self.embedding_tokens,
                'cost': round(self.get_embedding_cost(), 4)
            },
            'llm': {
                'model': self.llm_model,
                'calls': self.llm_calls,
                'input_tokens': self.llm_input_tokens,
                'output_tokens': self.llm_output_tokens,
                'cost': round(self.get_llm_cost(), 4)
            },
            'total_cost': round(self.get_total_cost(), 4)
        }

    def print_report(self):
        """Print formatted cost report"""
        report = self.get_report()

        print("\n" + "="*60)
        print("Cost Report")
        print("="*60)

        print(f"\nEmbeddings ({report['embeddings']['model']}):")
        print(f"  Calls: {report['embeddings']['calls']}")
        print(f"  Tokens: {report['embeddings']['tokens']:,}")
        print(f"  Cost: ${report['embeddings']['cost']:.4f}")

        print(f"\nLLM ({report['llm']['model']}):")
        print(f"  Calls: {report['llm']['calls']}")
        print(f"  Input tokens: {report['llm']['input_tokens']:,}")
        print(f"  Output tokens: {report['llm']['output_tokens']:,}")
        print(f"  Cost: ${report['llm']['cost']:.4f}")

        print(f"\nTotal Cost: ${report['total_cost']:.4f}")
        print("="*60)

    def reset(self):
        """Reset all counters"""
        self.embedding_tokens = 0
        self.llm_input_tokens = 0
        self.llm_output_tokens = 0
        self.embedding_calls = 0
        self.llm_calls = 0


# Global cache and tracker instances (optional)
_global_cache: Optional[EmbeddingCache] = None
_global_tracker: Optional[CostTracker] = None


def get_global_cache() -> EmbeddingCache:
    """Get or create global cache instance"""
    global _global_cache
    if _global_cache is None:
        _global_cache = EmbeddingCache()
    return _global_cache


def get_global_tracker() -> CostTracker:
    """Get or create global cost tracker"""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = CostTracker()
    return _global_tracker
