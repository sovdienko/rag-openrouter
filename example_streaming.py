"""Example usage of RAG pipeline with streaming responses"""

from rag_app.pipeline import RAGPipeline

# Sample documents
documents = [
    "The Python programming language was created by Guido van Rossum and first released in 1991. It emphasizes code readability with significant whitespace.",
    "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.",
    "Retrieval-Augmented Generation (RAG) combines information retrieval with large language model generation to provide accurate, grounded responses.",
]

def main():
    # Initialize pipeline
    print("Initializing RAG pipeline...")
    pipeline = RAGPipeline()

    # Ingest documents
    print("\nIngesting documents...")
    ingest_result = pipeline.ingest_documents(documents)
    print(f"✓ Ingested {ingest_result['num_documents']} documents")
    print(f"✓ Created {ingest_result['num_chunks']} chunks")
    print(f"✓ Stored {ingest_result['num_vectors']} vectors")

    # Example 1: Regular query (non-streaming)
    print("\n" + "="*60)
    print("Example 1: Regular Query (Non-Streaming)")
    print("="*60)
    question = "What is RAG?"
    print(f"\nQuestion: {question}")
    print("\nGenerating answer...")

    result = pipeline.query(question)
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nModel: {result['model']}")
    print(f"Sources used: {result['num_sources']}")

    # Example 2: Streaming query
    print("\n" + "="*60)
    print("Example 2: Streaming Query")
    print("="*60)
    question = "Tell me about Python programming language"
    print(f"\nQuestion: {question}")
    print("\nStreaming answer:")
    print("-" * 60)

    result = pipeline.query_stream(question)

    # Stream the response in real-time
    full_answer = ""
    for chunk in result['stream']:
        print(chunk, end="", flush=True)
        full_answer += chunk

    print("\n" + "-" * 60)
    print(f"\nModel: {result['model']}")
    print(f"Sources used: {result['num_sources']}")

    # Show sources if available
    if result.get('sources'):
        print("\nSource chunks:")
        for i, source in enumerate(result['sources'], 1):
            print(f"\n{i}. {source[:100]}...")

    # Example 3: Compare response times
    print("\n" + "="*60)
    print("Example 3: Benefits of Streaming")
    print("="*60)
    print("\nWith streaming:")
    print("- Users see text immediately as it's generated")
    print("- Better perceived performance")
    print("- More engaging user experience")
    print("- Can stop generation early if answer is sufficient")
    print("\nWithout streaming:")
    print("- Users wait for complete response")
    print("- May seem slower even if actual time is similar")
    print("- Less interactive")


if __name__ == "__main__":
    main()
