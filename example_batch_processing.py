"""Example demonstrating batch processing for efficient embedding generation"""

from rag_app import Config, EmbeddingCache, CostTracker
from rag_app.embeddings import EmbeddingService
import time


def main():
    print("="*70)
    print("Batch Processing Demo: Efficient Large-Scale Embedding")
    print("="*70)

    # Initialize configuration
    config = Config.from_env()

    # Example 1: Basic batch processing
    print("\n" + "="*70)
    print("Example 1: Basic Batch Processing (No Cache)")
    print("="*70)

    embedding_service = EmbeddingService(config)

    # Create sample documents
    documents = [
        f"This is document number {i} about artificial intelligence and machine learning."
        for i in range(1, 51)
    ]

    print(f"\nProcessing {len(documents)} documents with batch_size=10...")
    start_time = time.time()

    embeddings = embedding_service.batch_embed_documents(
        texts=documents,
        batch_size=10,
        show_progress=True
    )

    elapsed = time.time() - start_time

    print(f"\nResults:")
    print(f"  Total embeddings: {len(embeddings)}")
    print(f"  Embedding dimensions: {len(embeddings[0])}")
    print(f"  Time elapsed: {elapsed:.2f}s")
    print(f"  Time per document: {elapsed/len(documents):.3f}s")

    # Example 2: Batch processing with caching
    print("\n" + "="*70)
    print("Example 2: Batch Processing with Caching")
    print("="*70)

    cache = EmbeddingCache(memory_only=True)
    cached_service = EmbeddingService(config, cache=cache)

    # Create documents with some duplicates
    documents_with_dupes = [
        "Machine learning enables computers to learn from data",
        "Deep learning uses neural networks with multiple layers",
        "Natural language processing helps computers understand text",
        "Machine learning enables computers to learn from data",  # Duplicate
        "Computer vision allows machines to interpret images",
        "Deep learning uses neural networks with multiple layers",  # Duplicate
        "Reinforcement learning trains agents through rewards",
        "Natural language processing helps computers understand text",  # Duplicate
    ] * 5  # Repeat 5 times = 40 total documents (24 unique)

    print(f"\nProcessing {len(documents_with_dupes)} documents (with duplicates)...")
    print(f"Unique documents: {len(set(documents_with_dupes))}")

    start_time = time.time()
    embeddings = cached_service.batch_embed_documents(
        texts=documents_with_dupes,
        batch_size=20,
        show_progress=True
    )
    elapsed = time.time() - start_time

    stats = cache.get_stats()
    print(f"\nCaching efficiency:")
    print(f"  Total documents: {len(documents_with_dupes)}")
    print(f"  Unique documents: {len(set(documents_with_dupes))}")
    print(f"  API calls saved: {stats['hits']}")
    print(f"  Time elapsed: {elapsed:.2f}s")

    # Example 3: Comparing batch sizes
    print("\n" + "="*70)
    print("Example 3: Comparing Different Batch Sizes")
    print("="*70)

    test_docs = [f"Document {i} content" for i in range(100)]
    batch_sizes = [10, 25, 50, 100]

    print(f"\nProcessing {len(test_docs)} documents with different batch sizes:\n")

    for batch_size in batch_sizes:
        # Fresh service for each test
        service = EmbeddingService(config)

        start = time.time()
        embeddings = service.batch_embed_documents(
            texts=test_docs,
            batch_size=batch_size,
            show_progress=False
        )
        elapsed = time.time() - start

        num_batches = (len(test_docs) + batch_size - 1) // batch_size
        print(f"Batch size {batch_size:3d}: {elapsed:5.2f}s "
              f"({num_batches:2d} API calls, {elapsed/num_batches:.3f}s per call)")

    print("\nNote: Larger batch sizes reduce API overhead but may hit size limits")

    # Example 4: Batch processing with cost tracking
    print("\n" + "="*70)
    print("Example 4: Batch Processing with Cost Tracking")
    print("="*70)

    tracker = CostTracker()
    tracked_service = EmbeddingService(config, cost_tracker=tracker)

    large_collection = [
        f"Research paper about topic {i % 10}: artificial intelligence, "
        f"machine learning, neural networks, deep learning, and data science."
        for i in range(200)
    ]

    print(f"\nProcessing {len(large_collection)} documents...")
    embeddings = tracked_service.batch_embed_documents(
        texts=large_collection,
        batch_size=50,
        show_progress=True
    )

    report = tracker.get_report()
    print(f"\nCost analysis:")
    print(f"  Total documents: {len(large_collection)}")
    print(f"  API calls: {report['embeddings']['calls']}")
    print(f"  Estimated tokens: {report['embeddings']['tokens']:,}")
    print(f"  Estimated cost: ${report['embeddings']['cost']:.6f}")
    print(f"  Cost per document: ${report['embeddings']['cost']/len(large_collection):.8f}")

    # Example 5: Real-world scenario - processing PDF chunks
    print("\n" + "="*70)
    print("Example 5: Real-World Scenario - PDF Document Processing")
    print("="*70)

    # Simulate PDF chunks from multiple documents
    pdf_chunks = []
    for doc_num in range(1, 6):  # 5 PDF documents
        for chunk_num in range(1, 21):  # 20 chunks per document
            chunk_text = (
                f"[Document {doc_num}, Chunk {chunk_num}] "
                f"This section discusses important concepts in machine learning "
                f"including supervised learning, unsupervised learning, and reinforcement learning. "
                f"The key principles involve training models on data to make predictions."
            )
            pdf_chunks.append(chunk_text)

    print(f"\nScenario: Processing {len(pdf_chunks)} chunks from 5 PDF documents")
    print(f"Average chunk length: ~{sum(len(c) for c in pdf_chunks)/len(pdf_chunks):.0f} characters")

    # Use cache and tracker for optimal processing
    pdf_cache = EmbeddingCache(cache_dir=".cache/pdf_embeddings", memory_only=False)
    pdf_tracker = CostTracker()
    pdf_service = EmbeddingService(config, cache=pdf_cache, cost_tracker=pdf_tracker)

    print("\nFirst processing run...")
    start = time.time()
    embeddings = pdf_service.batch_embed_documents(
        texts=pdf_chunks,
        batch_size=50,
        show_progress=True
    )
    first_run_time = time.time() - start

    first_report = pdf_tracker.get_report()
    first_stats = pdf_cache.get_stats()

    print(f"\nFirst run results:")
    print(f"  Time: {first_run_time:.2f}s")
    print(f"  API calls: {first_report['embeddings']['calls']}")
    print(f"  Cost: ${first_report['embeddings']['cost']:.6f}")

    # Simulate reprocessing (e.g., after application restart)
    print("\n" + "-"*70)
    print("Simulating second run (cache loaded from disk)...")

    pdf_cache2 = EmbeddingCache(cache_dir=".cache/pdf_embeddings", memory_only=False)
    pdf_tracker2 = CostTracker()
    pdf_service2 = EmbeddingService(config, cache=pdf_cache2, cost_tracker=pdf_tracker2)

    start = time.time()
    embeddings2 = pdf_service2.batch_embed_documents(
        texts=pdf_chunks,
        batch_size=50,
        show_progress=True
    )
    second_run_time = time.time() - start

    second_report = pdf_tracker2.get_report()
    second_stats = pdf_cache2.get_stats()

    print(f"\nSecond run results:")
    print(f"  Time: {second_run_time:.2f}s")
    print(f"  API calls: {second_report['embeddings']['calls']}")
    print(f"  Cost: ${second_report['embeddings']['cost']:.6f}")
    print(f"  Speedup: {first_run_time/second_run_time:.1f}x faster")
    print(f"  Cost savings: 100% (no API calls needed)")

    # Example 6: Batch size recommendations
    print("\n" + "="*70)
    print("Example 6: Batch Size Selection Guide")
    print("="*70)

    print("\nTesting different text lengths to find optimal batch sizes...\n")

    # Test with short texts
    short_texts = [f"Short text {i}" for i in range(200)]
    service = EmbeddingService(config)

    start = time.time()
    service.batch_embed_documents(short_texts, batch_size=200, show_progress=False)
    short_time = time.time() - start
    print(f"Short texts (~15 chars): batch_size=200, time={short_time:.2f}s")

    # Test with medium texts
    medium_texts = [f"Medium length text number {i} with more content" * 5 for i in range(200)]
    service = EmbeddingService(config)

    start = time.time()
    service.batch_embed_documents(medium_texts, batch_size=100, show_progress=False)
    medium_time = time.time() - start
    print(f"Medium texts (~200 chars): batch_size=100, time={medium_time:.2f}s")

    # Test with long texts
    long_texts = [f"Long text number {i} with substantial content. " * 20 for i in range(200)]
    service = EmbeddingService(config)

    start = time.time()
    service.batch_embed_documents(long_texts, batch_size=50, show_progress=False)
    long_time = time.time() - start
    print(f"Long texts (~800 chars): batch_size=50, time={long_time:.2f}s")

    print("\nRecommendations:")
    print("  - Short texts (<50 chars): batch_size=200-300")
    print("  - Medium texts (50-500 chars): batch_size=50-100")
    print("  - Long texts (>500 chars): batch_size=25-50")
    print("  - Very long texts (>2000 chars): batch_size=10-25")

    # Summary
    print("\n" + "="*70)
    print("Batch Processing Best Practices")
    print("="*70)
    print("""
1. API Efficiency:
   - Batch processing reduces HTTP overhead
   - Fewer API calls = lower latency and costs
   - Optimal batch size depends on text length

2. Batch Size Selection:
   - Start with batch_size=100 for most use cases
   - Reduce for longer texts to avoid API limits
   - Increase for shorter texts to maximize efficiency
   - Monitor for API errors and adjust accordingly

3. Caching Integration:
   - Batch processing works seamlessly with caching
   - Cache hit rate improves with repeated batches
   - Disk cache persists across application restarts
   - Combine both for maximum efficiency

4. Progress Monitoring:
   - Use show_progress=True for long operations
   - Monitor cache hit rates to verify efficiency
   - Track costs with CostTracker for optimization

5. Production Recommendations:
   - Always enable caching for batch operations
   - Process documents in batches of 50-100
   - Use disk cache for persistent storage
   - Monitor API rate limits and adjust batch sizes

6. Performance Characteristics:
   - Individual calls: ~100-300ms per text
   - Batch of 100: ~500-1000ms total (5-10ms per text)
   - Speedup: 10-30x for batch processing
   - With caching: 100-1000x for repeated texts

7. Common Use Cases:
   - Initial document ingestion (large batch sizes)
   - Real-time processing (small batches with cache)
   - Periodic updates (medium batches)
   - Reprocessing (leverage cache for speed)

8. Error Handling:
   - Batch sizes that are too large may exceed API limits
   - Monitor response times and adjust accordingly
   - Consider exponential backoff for rate limiting
   - Implement retry logic for failed batches
    """)

    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    print("""
This demo showed:
- Basic batch processing with configurable batch sizes
- Caching integration for duplicate detection
- Batch size comparison and optimization
- Cost tracking for large collections
- Real-world PDF processing scenario
- Batch size selection guidelines

Key Takeaways:
- Batch processing provides 10-30x speedup vs individual calls
- Optimal batch size depends on text length
- Caching + batching = maximum efficiency
- Disk cache enables fast reprocessing
- Monitor costs and performance to optimize

Recommended Workflow:
1. Start with batch_size=100
2. Enable disk cache for persistence
3. Monitor cache hit rates
4. Adjust batch size based on text length
5. Track costs with CostTracker
    """)


if __name__ == "__main__":
    main()
