"""Example demonstrating hybrid search (vector + keyword)"""

from rag_app import PDFLoader, RAGPipeline


def main():
    print("="*70)
    print("Hybrid Search Demo: Vector Similarity + Keyword Matching")
    print("="*70)

    # Load documents
    print("\nLoading PDF documents...")
    loader = PDFLoader()
    documents = loader.load_with_metadata("rag-docs", verbose=False)

    print(f"Loaded {len(documents)} documents")

    # Example 1: Standard Vector Search
    print("\n" + "="*70)
    print("Example 1: Standard Vector Search (Baseline)")
    print("="*70)

    pipeline_vector = RAGPipeline(use_hybrid=False)

    print("\nIngesting documents...")
    result = pipeline_vector.ingest_documents(documents)
    print(f"Ingested {result['num_documents']} documents, {result['num_chunks']} chunks")

    # Query with specific term
    question = "What is BM25 ranking algorithm?"
    print(f"\nQuestion: {question}")
    print("Method: Pure vector similarity")

    answer_result = pipeline_vector.query(question, top_k=3)
    print(f"\nAnswer:\n{answer_result['answer']}")

    # Show sources
    print(f"\nTop {answer_result['num_sources']} sources (vector only):")
    vector_results = pipeline_vector.retriever.retrieve(question, top_k=3)
    for i, res in enumerate(vector_results, 1):
        print(f"\n{i}. {res['metadata']['source']}")
        print(f"   Score: {res.get('score', 0):.4f}")
        print(f"   Text: {res['text'][:100]}...")

    # Example 2: Hybrid Search
    print("\n" + "="*70)
    print("Example 2: Hybrid Search (Vector + Keyword)")
    print("="*70)

    # alpha=0.5 means 50% vector, 50% keyword
    pipeline_hybrid = RAGPipeline(use_hybrid=True, hybrid_alpha=0.5)

    print("\nIngesting documents...")
    result = pipeline_hybrid.ingest_documents(documents)
    print(f"Ingested {result['num_documents']} documents, {result['num_chunks']} chunks")

    print(f"\nQuestion: {question}")
    print("Method: Hybrid (50% vector + 50% keyword)")

    answer_result = pipeline_hybrid.query(question, top_k=3, initial_k=20)
    print(f"\nAnswer:\n{answer_result['answer']}")

    # Show sources with hybrid scores
    print(f"\nTop {answer_result['num_sources']} sources (hybrid):")
    hybrid_results = pipeline_hybrid.retriever.retrieve(question, top_k=3, use_hybrid=True, initial_k=20)
    for i, res in enumerate(hybrid_results, 1):
        print(f"\n{i}. {res['metadata']['source']}")
        print(f"   Hybrid score: {res.get('hybrid_score', 0):.4f}")
        print(f"   Vector score: {res.get('vector_score', 0):.4f}")
        print(f"   Keyword score: {res.get('keyword_score', 0):.4f}")
        print(f"   Text: {res['text'][:100]}...")

    # Example 3: Different Alpha Values
    print("\n" + "="*70)
    print("Example 3: Comparing Different Alpha Values")
    print("="*70)

    question2 = "machine learning neural networks"
    print(f"\nQuestion: {question2}")

    alphas = [0.0, 0.3, 0.5, 0.7, 1.0]
    print("\nAlpha values:")
    print("  0.0 = Pure keyword (BM25)")
    print("  0.5 = Balanced hybrid")
    print("  1.0 = Pure vector similarity")

    for alpha in alphas:
        pipeline_test = RAGPipeline(use_hybrid=True, hybrid_alpha=alpha)
        pipeline_test.ingest_documents(documents[:10])  # Use subset for speed

        results = pipeline_test.retriever.retrieve(question2, top_k=1, use_hybrid=True, initial_k=10)
        if results:
            result = results[0]
            print(f"\n  Alpha={alpha:.1f}: {result['metadata']['source']}")
            print(f"    Hybrid: {result.get('hybrid_score', 0):.4f}, "
                  f"Vector: {result.get('vector_score', 0):.4f}, "
                  f"Keyword: {result.get('keyword_score', 0):.4f}")

    # Example 4: Hybrid + Reranking (Best Quality)
    print("\n" + "="*70)
    print("Example 4: Hybrid Search + Reranking (Maximum Quality)")
    print("="*70)

    pipeline_best = RAGPipeline(use_hybrid=True, use_reranker=True, hybrid_alpha=0.5)

    print("\nIngesting documents...")
    result = pipeline_best.ingest_documents(documents)
    print(f"Ingested {result['num_documents']} documents")
    print("Enabled: Hybrid search (50% vector + 50% keyword) + Cross-encoder reranking")

    question3 = "What are transformers in deep learning?"
    print(f"\nQuestion: {question3}")
    print("\nRetrieval pipeline:")
    print("  Stage 1: Vector similarity search (retrieve 20 candidates)")
    print("  Stage 2: Hybrid re-scoring (vector + keyword)")
    print("  Stage 3: Cross-encoder reranking (final top 3)")

    answer_result = pipeline_best.query(question3, top_k=3, initial_k=20)
    print(f"\nAnswer:\n{answer_result['answer']}")

    # Benefits explanation
    print("\n" + "="*70)
    print("When to Use Hybrid Search")
    print("="*70)
    print("""
Hybrid search is particularly effective for:

1. Queries with Specific Terms:
   - Proper nouns (e.g., "OpenAI GPT-4", "BERT model")
   - Technical terminology (e.g., "BM25", "cosine similarity")
   - Exact phrases or acronyms

2. Domain-Specific Content:
   - Medical terms, legal jargon, scientific names
   - Product names, model numbers
   - Code identifiers, API names

3. Balancing Semantic and Lexical Matching:
   - Vector search: Captures semantic meaning and context
   - Keyword search (BM25): Ensures exact term matching
   - Hybrid: Best of both worlds!

Recommended alpha values:
- alpha=0.7-0.9: When semantic understanding is more important
- alpha=0.5: Balanced approach (default)
- alpha=0.1-0.3: When exact term matching is critical

Combine with reranking for maximum quality:
- Hybrid search provides diverse candidates
- Cross-encoder reranking ensures best relevance
    """)


if __name__ == "__main__":
    main()
