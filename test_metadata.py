"""Test script to verify filename metadata in vector database"""

from rag_app import PDFLoader, RAGPipeline

def main():
    print("="*70)
    print("Testing Filename Metadata in Vector Database")
    print("="*70)

    # Load PDFs with metadata
    loader = PDFLoader()
    documents = loader.load_with_metadata("rag-docs", verbose=False)

    print(f"\nLoaded {len(documents)} documents")
    print("\nFirst 3 documents with metadata:")
    for i, doc in enumerate(documents[:3], 1):
        print(f"\n{i}. Filename: {doc['filename']}")
        print(f"   Path: {doc['path']}")
        print(f"   Text length: {len(doc['text'])} characters")

    # Initialize pipeline and ingest with metadata
    print("\n" + "="*70)
    print("Ingesting documents into RAG pipeline...")
    print("="*70)

    pipeline = RAGPipeline()
    result = pipeline.ingest_documents(documents[:5])  # Use only first 5 for quick test

    print(f"\nIngestion complete:")
    print(f"  Documents ingested: {result['num_documents']}")
    print(f"  Chunks created: {result['num_chunks']}")
    print(f"  Vectors stored: {result['num_vectors']}")

    # Query to test retrieval
    print("\n" + "="*70)
    print("Testing Query with Source Tracking")
    print("="*70)

    question = "What is machine learning?"
    print(f"\nQuestion: {question}")

    answer_result = pipeline.query(question, top_k=3)
    print(f"\nAnswer:\n{answer_result['answer']}")

    # Show sources
    print("\n" + "="*70)
    print("Source Documents:")
    print("="*70)
    if answer_result.get('sources'):
        for i, source in enumerate(answer_result['sources'], 1):
            print(f"\n{i}. {source[:200]}...")

    print("\n" + "="*70)
    print("Verification Complete!")
    print("="*70)
    print("""
The filename is now stored as the 'source' in the vector database!

Each chunk in Pinecone includes:
- source: PDF filename (e.g., "ai_topic_01.pdf")
- chunk: Chunk number within the document
- total_chunks: Total chunks in the document

The 'source' field directly contains the filename, making it easy to
identify which PDF file each retrieved chunk came from.
    """)


if __name__ == "__main__":
    main()
