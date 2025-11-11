"""Example usage of the RAG pipeline"""

from rag_app.pipeline import RAGPipeline
from rag_app.config import Config

# Sample documents
documents = [
    "The Python programming language was created by Guido van Rossum and first released in 1991. It emphasizes code readability with significant whitespace.",
    "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.",
    "Retrieval-Augmented Generation (RAG) combines information retrieval with large language model generation to provide accurate, grounded responses.",
]

def main():
    # Initialize pipeline with default configuration
    print("Initializing RAG pipeline...")
    pipeline = RAGPipeline()

    # Ingest documents
    print("\nIngesting documents...")
    ingest_result = pipeline.ingest_documents(documents)
    print(f"✓ Ingested {ingest_result['num_documents']} documents")
    print(f"✓ Created {ingest_result['num_chunks']} chunks")
    print(f"✓ Stored {ingest_result['num_vectors']} vectors")

    # Query the system
    print("\n" + "="*60)
    question = "What is RAG and how does it work?"
    print(f"Question: {question}")
    print("="*60)

    print("\nRetrieving relevant information...")
    result = pipeline.query(question)

    print(f"\nAnswer:\n{result['answer']}")
    print(f"\n{'='*60}")
    print(f"Model: {result['model']}")
    print(f"Sources used: {result['num_sources']}")

    # Show source chunks
    if result.get('sources'):
        print(f"\nSource chunks:")
        for i, source in enumerate(result['sources'], 1):
            print(f"\n{i}. {source[:100]}...")

    # Get statistics
    print(f"\n{'='*60}")
    print("Index statistics:")
    stats = pipeline.get_stats()
    print(f"Total vectors: {stats.get('total_vector_count', 'N/A')}")


if __name__ == "__main__":
    main()
