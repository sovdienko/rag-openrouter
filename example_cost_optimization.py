"""Example demonstrating cost optimization with caching and cost tracking"""

from rag_app import Config, EmbeddingCache, CostTracker
from rag_app.embeddings import EmbeddingService
import time


def main():
    print("="*70)
    print("Cost Optimization Demo: Caching and Cost Tracking")
    print("="*70)

    # Initialize configuration
    config = Config.from_env()

    # Example 1: Basic caching with memory-only cache
    print("\n" + "="*70)
    print("Example 1: Memory-Only Caching")
    print("="*70)

    # Create cache and embedding service
    cache = EmbeddingCache(memory_only=True)
    embedding_service = EmbeddingService(config, cache=cache)

    sample_texts = [
        "What is machine learning?",
        "Explain neural networks",
        "What is deep learning?",
        "What is machine learning?",  # Duplicate
        "Explain neural networks",     # Duplicate
    ]

    print("\nEmbedding 5 texts (2 are duplicates)...")
    start_time = time.time()

    for i, text in enumerate(sample_texts, 1):
        embedding = embedding_service.embed_text(text)
        print(f"  {i}. Embedded: {text[:40]}... (dim={len(embedding)})")

    elapsed = time.time() - start_time

    # Show cache statistics
    stats = cache.get_stats()
    print(f"\nCache Statistics:")
    print(f"  Total requests: {stats['total_requests']}")
    print(f"  Cache hits: {stats['hits']}")
    print(f"  Cache misses: {stats['misses']}")
    print(f"  Hit rate: {stats['hit_rate']:.1f}%")
    print(f"  Memory size: {stats['memory_size']} embeddings")
    print(f"  Time elapsed: {elapsed:.2f}s")

    # Example 2: Persistent disk cache
    print("\n" + "="*70)
    print("Example 2: Persistent Disk Cache")
    print("="*70)

    # Create disk-backed cache
    disk_cache = EmbeddingCache(cache_dir=".cache/embeddings", memory_only=False)
    embedding_service_disk = EmbeddingService(config, cache=disk_cache)

    print("\nFirst run - embedding 3 texts...")
    texts = [
        "Artificial intelligence is transforming industries",
        "Machine learning requires large datasets",
        "Deep learning uses neural networks"
    ]

    for text in texts:
        embedding_service_disk.embed_text(text)
        print(f"  Embedded: {text[:40]}...")

    stats1 = disk_cache.get_stats()
    print(f"\nCache stats after first run:")
    print(f"  Hits: {stats1['hits']}, Misses: {stats1['misses']}")

    # Simulate second run (cache should be loaded from disk)
    print("\nSecond run - same texts (should hit cache)...")
    disk_cache2 = EmbeddingCache(cache_dir=".cache/embeddings", memory_only=False)
    embedding_service_disk2 = EmbeddingService(config, cache=disk_cache2)

    for text in texts:
        embedding_service_disk2.embed_text(text)
        print(f"  Embedded: {text[:40]}...")

    stats2 = disk_cache2.get_stats()
    print(f"\nCache stats after second run:")
    print(f"  Hits: {stats2['hits']}, Misses: {stats2['misses']}")
    print(f"  Hit rate: {stats2['hit_rate']:.1f}%")

    # Example 3: Cost tracking
    print("\n" + "="*70)
    print("Example 3: Cost Tracking")
    print("="*70)

    # Create cost tracker and embedding service
    cost_tracker = CostTracker()
    embedding_service_tracked = EmbeddingService(config, cost_tracker=cost_tracker)

    print("\nEmbedding texts with cost tracking...")
    tracked_texts = [
        "Natural language processing enables computers to understand text",
        "Computer vision allows machines to interpret visual information",
        "Reinforcement learning trains agents through trial and error",
        "Transfer learning leverages pre-trained models for new tasks"
    ]

    for text in tracked_texts:
        embedding_service_tracked.embed_text(text)
        print(f"  Embedded: {text[:50]}...")

    # Get cost report
    report = cost_tracker.get_report()
    print(f"\nCost Report:")
    print(f"  Model: {report['embeddings']['model']}")
    print(f"  API calls: {report['embeddings']['calls']}")
    print(f"  Estimated tokens: {report['embeddings']['tokens']:,}")
    print(f"  Estimated cost: ${report['embeddings']['cost']:.6f}")

    # Example 4: Combined caching and cost tracking
    print("\n" + "="*70)
    print("Example 4: Combined Caching + Cost Tracking")
    print("="*70)

    # Create both cache and cost tracker
    combined_cache = EmbeddingCache(memory_only=True)
    combined_tracker = CostTracker()
    embedding_service_combined = EmbeddingService(
        config,
        cache=combined_cache,
        cost_tracker=combined_tracker
    )

    # Embed texts multiple times to show cache benefit
    demo_texts = [
        "Supervised learning requires labeled training data",
        "Unsupervised learning finds patterns in unlabeled data",
        "Semi-supervised learning combines both approaches"
    ]

    print("\nFirst pass - embedding 3 texts...")
    for text in demo_texts:
        embedding_service_combined.embed_text(text)

    report1 = combined_tracker.get_report()
    cache_stats1 = combined_cache.get_stats()

    print(f"\nAfter first pass:")
    print(f"  Cache hits: {cache_stats1['hits']}, misses: {cache_stats1['misses']}")
    print(f"  API calls: {report1['embeddings']['calls']}")
    print(f"  Cost: ${report1['embeddings']['cost']:.6f}")

    print("\nSecond pass - embedding same 3 texts (should hit cache)...")
    for text in demo_texts:
        embedding_service_combined.embed_text(text)

    report2 = combined_tracker.get_report()
    cache_stats2 = combined_cache.get_stats()

    print(f"\nAfter second pass:")
    print(f"  Cache hits: {cache_stats2['hits']}, misses: {cache_stats2['misses']}")
    print(f"  API calls: {report2['embeddings']['calls']} (no increase!)")
    print(f"  Cost: ${report2['embeddings']['cost']:.6f} (no increase!)")
    print(f"  Cache hit rate: {cache_stats2['hit_rate']:.1f}%")

    # Example 5: Model comparison
    print("\n" + "="*70)
    print("Example 5: Model Cost Comparison")
    print("="*70)

    print("\nComparing costs between embedding models:\n")

    # text-embedding-3-small
    config_small = Config.from_env()
    config_small.embedding_model = "openai/text-embedding-3-small"
    tracker_small = CostTracker()
    service_small = EmbeddingService(config_small, cost_tracker=tracker_small)

    # text-embedding-3-large
    config_large = Config.from_env()
    config_large.embedding_model = "openai/text-embedding-3-large"
    tracker_large = CostTracker()
    service_large = EmbeddingService(config_large, cost_tracker=tracker_large)

    test_text = "The quick brown fox jumps over the lazy dog" * 20  # ~200 words

    # Embed with both models
    service_small.embed_text(test_text)
    service_large.embed_text(test_text)

    report_small = tracker_small.get_report()
    report_large = tracker_large.get_report()

    print(f"Text length: ~{len(test_text)} characters")
    print(f"\ntext-embedding-3-small:")
    print(f"  Tokens: {report_small['embeddings']['tokens']:,}")
    print(f"  Cost: ${report_small['embeddings']['cost']:.6f}")

    print(f"\ntext-embedding-3-large:")
    print(f"  Tokens: {report_large['embeddings']['tokens']:,}")
    print(f"  Cost: ${report_large['embeddings']['cost']:.6f}")

    if report_small['embeddings']['cost'] > 0:
        cost_diff = report_large['embeddings']['cost'] / report_small['embeddings']['cost']
        print(f"\ntext-embedding-3-large is {cost_diff:.1f}x more expensive")
    else:
        print(f"\nCost comparison: 3-large is 6.5x more expensive ($0.13 vs $0.02 per 1M tokens)")

    # Example 6: Batch processing with caching
    print("\n" + "="*70)
    print("Example 6: Batch Processing with Caching")
    print("="*70)

    batch_cache = EmbeddingCache(memory_only=True)
    batch_tracker = CostTracker()
    batch_service = EmbeddingService(
        config,
        cache=batch_cache,
        cost_tracker=batch_tracker
    )

    # Large batch with duplicates
    large_batch = [
        "Data science combines statistics and programming",
        "Big data requires distributed computing systems",
        "Data science combines statistics and programming",  # Duplicate
        "Cloud computing provides on-demand resources",
        "Big data requires distributed computing systems",  # Duplicate
        "Data science combines statistics and programming",  # Duplicate
        "Edge computing processes data closer to the source",
        "Quantum computing uses quantum mechanics principles"
    ]

    print(f"\nProcessing batch of {len(large_batch)} texts...")
    print(f"(Contains {len(set(large_batch))} unique texts)")

    start = time.time()
    embeddings = batch_service.embed_texts(large_batch)
    elapsed = time.time() - start

    batch_stats = batch_cache.get_stats()
    batch_report = batch_tracker.get_report()

    print(f"\nResults:")
    print(f"  Embeddings generated: {len(embeddings)}")
    print(f"  Time elapsed: {elapsed:.2f}s")
    print(f"  Cache hits: {batch_stats['hits']}")
    print(f"  Cache misses: {batch_stats['misses']}")
    print(f"  Hit rate: {batch_stats['hit_rate']:.1f}%")
    print(f"  API calls saved: {batch_stats['hits']}")
    print(f"  Actual cost: ${batch_report['embeddings']['cost']:.6f}")

    # Calculate what cost would have been without cache
    hypothetical_calls = len(large_batch)
    hypothetical_tokens = batch_report['embeddings']['tokens'] / batch_stats['misses'] * hypothetical_calls if batch_stats['misses'] > 0 else 0
    hypothetical_cost = hypothetical_tokens / 1_000_000 * CostTracker.EMBEDDING_COSTS[config.embedding_model]

    print(f"  Cost without cache: ${hypothetical_cost:.6f}")
    savings_pct = ((hypothetical_cost - batch_report['embeddings']['cost']) / hypothetical_cost * 100) if hypothetical_cost > 0 else 0
    print(f"  Savings: {savings_pct:.1f}%")

    # Best practices summary
    print("\n" + "="*70)
    print("Cost Optimization Best Practices")
    print("="*70)
    print("""
1. Model Selection:
   - Use text-embedding-3-small for most applications (6.5x cheaper)
   - Use text-embedding-3-large only when higher accuracy is critical
   - Consider free models for development/testing

2. Caching Strategy:
   - Enable memory cache for short-running processes
   - Use disk cache for long-running applications
   - Cache persists across restarts, saving costs
   - Especially valuable for repeated queries

3. Batch Processing:
   - Process multiple texts in one API call when possible
   - Cache automatically deduplicates within batches
   - Reduces API overhead

4. Cost Tracking:
   - Monitor costs in development to avoid surprises
   - Set budgets and alerts for production
   - Track costs per feature/user for optimization
   - Use CostTracker to identify expensive operations

5. Production Recommendations:
   - Always use caching in production
   - Start with disk cache for persistence
   - Monitor cache hit rates (aim for >50%)
   - Consider cache warming for common queries
   - Regularly review cost reports

6. Cost Estimates (per 1M tokens):
   - text-embedding-3-small: $0.02
   - text-embedding-3-large: $0.13
   - Llama 3.3 70B: $0.88 (input/output)
   - Llama 3.3 8B: Free (with limits)

7. Scaling Tips:
   - Cache hit rates improve with scale
   - Consider Redis for distributed caching
   - Implement cache expiration for dynamic content
   - Monitor cache size and implement LRU if needed

8. Development vs Production:
   - Use free models for development
   - Use memory cache for quick iteration
   - Switch to paid models and disk cache for production
   - Implement cost alerts before scaling
    """)

    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    print("""
This demo showed:
- Memory and disk-based caching
- Cost tracking for embeddings
- Model cost comparison
- Combined caching + tracking
- Batch processing optimization
- Real-world savings examples

Key Takeaways:
- Caching can reduce costs by 50%+ for repeated queries
- text-embedding-3-small is 6.5x cheaper than 3-large
- Batch processing with deduplication saves significantly
- Cost tracking helps identify optimization opportunities
- Disk cache persists across restarts

Next Steps:
1. Enable caching in your RAG pipeline
2. Add cost tracking to monitor usage
3. Evaluate if text-embedding-3-small meets your needs
4. Implement batch processing for bulk operations
5. Set up cost alerts for production
    """)


if __name__ == "__main__":
    main()
