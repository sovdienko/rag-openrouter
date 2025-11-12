"""Example demonstrating two-stage retrieval with reranking"""

from rag_app.pipeline import RAGPipeline

# Sample documents
documents = [
    "The Python programming language was created by Guido van Rossum and first released in 1991. It emphasizes code readability with significant whitespace.",
    "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.",
    "Retrieval-Augmented Generation (RAG) combines information retrieval with large language model generation to provide accurate, grounded responses.",
    "Deep learning uses neural networks with multiple layers to progressively extract higher-level features from raw input.",
    "Natural language processing (NLP) is a field of AI focused on the interaction between computers and human language.",
    "Vector databases store high-dimensional embeddings and enable fast similarity search for semantic retrieval tasks.",
]

def main():
    print("="*70)
    print("Two-Stage Retrieval with Reranking Demo")
    print("="*70)

    # Example 1: Without Reranking
    print("\n" + "="*70)
    print("Example 1: Standard Retrieval (No Reranking)")
    print("="*70)

    pipeline_standard = RAGPipeline(use_reranker=False)

    print("\nIngesting documents...")
    ingest_result = pipeline_standard.ingest_documents(documents)
    print(f"✓ Ingested {ingest_result['num_documents']} documents")
    print(f"✓ Created {ingest_result['num_chunks']} chunks")

    question = "How does RAG work with vector databases?"
    print(f"\nQuestion: {question}")

    result = pipeline_standard.query(question, top_k=3)
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nSources used: {result['num_sources']}")

    # Example 2: With Reranking
    print("\n" + "="*70)
    print("Example 2: Two-Stage Retrieval (With Reranking)")
    print("="*70)

    pipeline_reranked = RAGPipeline(use_reranker=True)

    print("\nIngesting documents...")
    ingest_result = pipeline_reranked.ingest_documents(documents)
    print(f"✓ Ingested {ingest_result['num_documents']} documents")
    print(f"✓ Created {ingest_result['num_chunks']} chunks")
    print("✓ Reranker enabled (cross-encoder/ms-marco-MiniLM-L-6-v2)")

    print(f"\nQuestion: {question}")
    print("\nStage 1: Retrieving 15 candidates...")
    print("Stage 2: Reranking to top 3...")

    result = pipeline_reranked.query(
        question,
        top_k=3,
        use_reranking=True,
        initial_k=15
    )
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nSources used: {result['num_sources']}")

    # Example 3: Compare Results
    print("\n" + "="*70)
    print("Example 3: Comparing Retrieval Quality")
    print("="*70)

    question = "What is machine learning?"
    print(f"\nQuestion: {question}")

    # Standard retrieval
    print("\n--- Standard Retrieval (top 3) ---")
    result_standard = pipeline_standard.query(question, top_k=3, use_reranking=False)
    if result_standard.get('sources'):
        for i, source in enumerate(result_standard['sources'], 1):
            print(f"\n{i}. {source[:80]}...")

    # With reranking
    print("\n--- Two-Stage Retrieval (15 → rerank to 3) ---")
    result_reranked = pipeline_reranked.query(
        question,
        top_k=3,
        use_reranking=True,
        initial_k=15
    )
    if result_reranked.get('sources'):
        for i, source in enumerate(result_reranked['sources'], 1):
            print(f"\n{i}. {source[:80]}...")

    # Benefits explanation
    print("\n" + "="*70)
    print("Benefits of Two-Stage Retrieval")
    print("="*70)
    print("""
Two-stage retrieval improves precision:

1. Stage 1 (Broad Retrieval):
   - Fast vector similarity search
   - Retrieves 15-50 candidates
   - Uses efficient bi-encoder embeddings

2. Stage 2 (Reranking):
   - Cross-encoder evaluates query-document pairs
   - More accurate relevance scoring
   - Returns top K most relevant results

Why it works:
- Bi-encoders are fast but less precise
- Cross-encoders are slower but more accurate
- Combining both gives speed + quality
- Minimal latency increase (~100-200ms)
- Significantly better retrieval precision
    """)


if __name__ == "__main__":
    main()
