"""Example demonstrating RAG with filename tracking"""

from rag_app import PDFLoader, RAGPipeline


def main():
    print("="*70)
    print("RAG Pipeline with Filename Tracking")
    print("="*70)

    # Load documents with metadata (includes filenames and paths)
    print("\nLoading PDF documents...")
    loader = PDFLoader()
    documents = loader.load_with_metadata("rag-docs")

    print(f"Loaded {len(documents)} PDF files")
    print("\nSample documents:")
    for i, doc in enumerate(documents[:5], 1):
        print(f"  {i}. {doc['filename']} ({len(doc['text'])} chars)")

    # Initialize RAG pipeline
    print("\n" + "="*70)
    print("Initializing RAG Pipeline")
    print("="*70)

    pipeline = RAGPipeline()

    # Ingest documents with metadata
    print("\nIngesting documents (this may take a moment)...")
    result = pipeline.ingest_documents(documents)

    print(f"\nIngestion complete:")
    print(f"  Documents: {result['num_documents']}")
    print(f"  Chunks: {result['num_chunks']}")
    print(f"  Vectors: {result['num_vectors']}")

    # Query examples
    questions = [
        "What is machine learning?",
        "Explain neural networks",
        "What is RAG and how does it work?"
    ]

    for question in questions:
        print("\n" + "="*70)
        print(f"Question: {question}")
        print("="*70)

        answer_result = pipeline.query(question, top_k=3)
        print(f"\nAnswer:\n{answer_result['answer']}")

        print(f"\nSources ({answer_result['num_sources']} chunks retrieved):")
        print("-"*70)

        # Get the retriever to show detailed metadata
        retriever_results = pipeline.retriever.retrieve(question, top_k=3)

        for i, result in enumerate(retriever_results, 1):
            print(f"\n{i}. Source: {result['metadata'].get('source', 'unknown')}")
            print(f"   Chunk {result['metadata'].get('chunk', 0) + 1}/{result['metadata'].get('total_chunks', 0)}")
            print(f"   Similarity score: {result.get('score', 0):.4f}")
            print(f"   Text preview: {result['text'][:150]}...")

    print("\n" + "="*70)
    print("Benefits of Filename Tracking")
    print("="*70)
    print("""
With filename stored as 'source' in the vector database, you can:

1. Source Attribution: Know exactly which PDF file each answer comes from
2. Document Provenance: Track the origin of retrieved information
3. Quality Control: Identify which documents provide the best answers
4. Debugging: Quickly locate source documents when verifying results
5. Citations: Generate proper citations with document names
6. Filtering: Potentially filter results by specific documents (future feature)

Example metadata structure in Pinecone:
{
    "source": "ai_topic_01.pdf",  ← Filename directly in source field
    "chunk": 0,
    "total_chunks": 5,
    "text": "The actual chunk text..."
}
    """)


if __name__ == "__main__":
    main()
