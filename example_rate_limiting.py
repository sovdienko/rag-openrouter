"""Example demonstrating rate limiting and retry logic for robust API calls"""

from rag_app import Config, RateLimiter, rate_limit_retry
from rag_app.embeddings import EmbeddingService
import time


def main():
    print("="*70)
    print("Rate Limiting and Retry Demo: Robust API Error Handling")
    print("="*70)

    # Initialize configuration
    config = Config.from_env()

    # Example 1: Basic retry with exponential backoff
    print("\n" + "="*70)
    print("Example 1: Automatic Retry with Exponential Backoff")
    print("="*70)

    print("\nEmbedding service with automatic retry enabled...")
    service_with_retry = EmbeddingService(
        config,
        use_retry=True,
        max_retries=3
    )

    texts = [
        "Artificial intelligence is transforming industries",
        "Machine learning enables predictive analytics",
        "Deep learning uses neural networks"
    ]

    print(f"\nProcessing {len(texts)} texts with retry enabled...")
    start = time.time()
    embeddings = service_with_retry.embed_texts(texts)
    elapsed = time.time() - start

    print(f"Success! Generated {len(embeddings)} embeddings in {elapsed:.2f}s")
    print("Note: If rate limits are hit, automatic retry with exponential backoff activates")

    # Example 2: Rate limiting to prevent hitting API limits
    print("\n" + "="*70)
    print("Example 2: Rate Limiting to Control Request Frequency")
    print("="*70)

    # Create rate limiter: maximum 60 requests per minute
    rate_limiter = RateLimiter(requests_per_minute=60)

    service_with_limiter = EmbeddingService(
        config,
        rate_limiter=rate_limiter
    )

    print("\nProcessing documents with rate limiting (60 req/min)...")
    documents = [f"Document {i} about AI and ML" for i in range(10)]

    start = time.time()
    for i, doc in enumerate(documents, 1):
        embedding = service_with_limiter.embed_text(doc)
        if i % 3 == 0:
            print(f"  Processed {i}/{len(documents)} documents...")

    elapsed = time.time() - start
    stats = rate_limiter.get_stats()

    print(f"\nRate limiter stats:")
    print(f"  Total requests: {stats['total_requests']}")
    print(f"  Requests per minute limit: {stats['requests_per_minute']}")
    print(f"  Min interval between requests: {stats['min_interval']:.3f}s")
    print(f"  Total time: {elapsed:.2f}s")
    print(f"  Average time per request: {elapsed/len(documents):.3f}s")

    # Example 3: Combining rate limiting with retry
    print("\n" + "="*70)
    print("Example 3: Rate Limiting + Automatic Retry (Production Setup)")
    print("="*70)

    # Production-ready configuration
    production_limiter = RateLimiter(requests_per_minute=50)
    production_service = EmbeddingService(
        config,
        rate_limiter=production_limiter,
        use_retry=True,
        max_retries=5  # More retries for production
    )

    print("\nProduction configuration:")
    print("  - Rate limit: 50 requests/minute")
    print("  - Automatic retry: enabled")
    print("  - Max retries: 5")
    print("  - Exponential backoff: 1s, 2s, 4s, 8s, 16s")

    batch_docs = [f"Production document {i}" for i in range(20)]
    print(f"\nProcessing {len(batch_docs)} documents...")

    start = time.time()
    embeddings = production_service.embed_texts(batch_docs)
    elapsed = time.time() - start

    print(f"\nProcessed {len(embeddings)} documents successfully")
    print(f"Time: {elapsed:.2f}s")
    print(f"Average: {elapsed/len(batch_docs):.3f}s per document")

    # Example 4: Using rate_limit_retry decorator directly
    print("\n" + "="*70)
    print("Example 4: Using rate_limit_retry Decorator")
    print("="*70)

    print("\nApplying retry decorator to custom function...")

    @rate_limit_retry(max_retries=3, initial_wait=1.0, backoff_factor=2.0)
    def custom_api_call(text):
        """Custom function with automatic retry"""
        service = EmbeddingService(config, use_retry=False)  # Decorator handles retry
        return service.embed_text(text)

    text = "Custom function call with retry decorator"
    print(f"\nCalling: {text[:50]}...")

    start = time.time()
    embedding = custom_api_call(text)
    elapsed = time.time() - start

    print(f"Success! Embedding dimension: {len(embedding)}")
    print(f"Time: {elapsed:.2f}s")
    print("\nIf rate limit is hit, decorator will retry with:")
    print("  - Attempt 1: wait 1.0s")
    print("  - Attempt 2: wait 2.0s")
    print("  - Attempt 3: wait 4.0s")

    # Example 5: Rate limiter as context manager
    print("\n" + "="*70)
    print("Example 5: Rate Limiter Context Manager")
    print("="*70)

    limiter = RateLimiter(requests_per_second=2)  # Max 2 requests per second
    service = EmbeddingService(config, use_retry=False)

    print("\nUsing rate limiter as context manager (2 req/sec)...")
    texts_to_process = [f"Text {i}" for i in range(5)]

    start = time.time()
    embeddings = []

    for i, text in enumerate(texts_to_process, 1):
        with limiter:  # Automatically waits if needed
            embedding = service.embed_text(text)
            embeddings.append(embedding)
            elapsed_so_far = time.time() - start
            print(f"  Request {i}: completed at {elapsed_so_far:.2f}s")

    total_time = time.time() - start
    print(f"\nTotal time: {total_time:.2f}s")
    print(f"Expected minimum time: {(len(texts_to_process)-1) * 0.5:.2f}s (2 req/sec)")

    # Example 6: Handling different error types
    print("\n" + "="*70)
    print("Example 6: Error Detection and Retry Logic")
    print("="*70)

    print("\nThe retry decorator automatically detects:")
    print("  - Rate limit errors (429, 'rate_limit', 'too many requests')")
    print("  - Temporary errors (500, 502, 503, 'timeout', 'connection')")
    print("  - Retries with exponential backoff")
    print("  - Non-retryable errors are raised immediately")

    print("\nError simulation example:")
    error_scenarios = [
        ("Rate Limit (429)", "Will retry 3 times with backoff"),
        ("Server Error (500)", "Will retry 3 times with backoff"),
        ("Connection Timeout", "Will retry 3 times with backoff"),
        ("Invalid API Key (401)", "Fails immediately (non-retryable)"),
        ("Invalid Request (400)", "Fails immediately (non-retryable)")
    ]

    for error_type, behavior in error_scenarios:
        print(f"  - {error_type}: {behavior}")

    # Example 7: Batch processing with rate limiting
    print("\n" + "="*70)
    print("Example 7: Batch Processing with Rate Limiting")
    print("="*70)

    batch_limiter = RateLimiter(requests_per_minute=30)
    batch_service = EmbeddingService(
        config,
        rate_limiter=batch_limiter,
        use_retry=True,
        max_retries=3
    )

    large_batch = [f"Batch document {i} with various content" for i in range(50)]

    print(f"\nProcessing {len(large_batch)} documents in batches...")
    print("Rate limit: 30 req/min")
    print("Batch size: 10")

    start = time.time()
    embeddings = batch_service.batch_embed_documents(
        texts=large_batch,
        batch_size=10,
        show_progress=True
    )
    elapsed = time.time() - start

    print(f"\nCompleted in {elapsed:.2f}s")
    print(f"Average: {elapsed/len(large_batch):.3f}s per document")

    # Best practices summary
    print("\n" + "="*70)
    print("Rate Limiting and Retry Best Practices")
    print("="*70)
    print("""
1. Always Enable Retry for Production:
   - Use use_retry=True for all production services
   - Set max_retries=3-5 depending on criticality
   - Automatic exponential backoff prevents overwhelming API

2. Implement Rate Limiting:
   - Set requests_per_minute based on your API tier
   - Prevents hitting rate limits before they happen
   - More efficient than retry after failure

3. Recommended Production Configuration:
   service = EmbeddingService(
       config,
       rate_limiter=RateLimiter(requests_per_minute=50),
       use_retry=True,
       max_retries=5
   )

4. Exponential Backoff Strategy:
   - Initial wait: 1 second
   - Backoff factor: 2x
   - Max wait: 60 seconds
   - Progression: 1s -> 2s -> 4s -> 8s -> 16s -> 32s -> 60s

5. Error Handling:
   - Rate limits (429): Automatically retried
   - Server errors (5xx): Automatically retried
   - Temporary failures: Automatically retried
   - Auth errors (401, 403): Fail immediately
   - Bad requests (400): Fail immediately

6. Batch Processing Considerations:
   - Combine rate limiting with batch processing
   - Rate limit applies per batch, not per document
   - Reduces total number of rate-limited requests
   - Example: 1000 docs in 10 batches = 10 rate checks

7. Monitoring and Logging:
   - Rate limiter provides statistics
   - Retry attempts are logged automatically
   - Monitor logs for frequent retries (may indicate need to adjust limits)
   - Track cache hit rates to reduce API calls

8. Cost Optimization:
   - Rate limiting prevents wasted quota on failed requests
   - Retry with backoff is cheaper than immediate retry
   - Combine with caching for maximum efficiency
   - Monitor retry frequency to adjust rate limits

9. Different API Tiers:
   - Free tier: RateLimiter(requests_per_minute=10)
   - Basic tier: RateLimiter(requests_per_minute=60)
   - Pro tier: RateLimiter(requests_per_minute=300)
   - Enterprise: RateLimiter(requests_per_minute=1000)

10. Testing and Development:
    - Use lower rate limits in development
    - Test retry logic with intentional failures
    - Monitor retry behavior in logs
    - Adjust parameters based on actual performance
    """)

    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    print("""
This demo showed:
- Automatic retry with exponential backoff
- Rate limiting to control request frequency
- Combined rate limiting + retry for production
- Using retry decorator directly
- Rate limiter as context manager
- Error detection and handling
- Batch processing with rate limiting

Key Takeaways:
- Always enable retry for production (prevents failures)
- Use rate limiting to prevent hitting limits
- Exponential backoff prevents overwhelming API
- Combine with caching for maximum efficiency
- Monitor logs to adjust parameters

Production Setup:
service = EmbeddingService(
    config,
    cache=EmbeddingCache(cache_dir=".cache"),
    rate_limiter=RateLimiter(requests_per_minute=50),
    use_retry=True,
    max_retries=5
)

This configuration provides:
- Caching to reduce API calls
- Rate limiting to prevent 429 errors
- Automatic retry for transient failures
- Exponential backoff for gradual recovery
- Maximum reliability and cost efficiency
    """)


if __name__ == "__main__":
    main()
