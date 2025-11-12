"""Clear index and test new metadata structure"""

from rag_app import PDFLoader, RAGPipeline


def main():
    print("="*70)
    print("Clear Index and Test New Metadata Structure")
    print("="*70)

    # Initialize pipeline
    pipeline = RAGPipeline()

    # Clear the index
    print("\nClearing Pinecone index...")
    pipeline.clear_index()
    print("Index cleared!")

    # Load PDFs with metadata
    print("\nLoading PDFs...")
    loader = PDFLoader()
    documents = loader.load_with_metadata("rag-docs", verbose=False)[:5]

    print(f"Loaded {len(documents)} documents:")
    for i, doc in enumerate(documents, 1):
        print(f"  {i}. {doc['filename']}")

    # Ingest with new structure
    print("\nIngesting documents with NEW metadata structure...")
    result = pipeline.ingest_documents(documents)
    print(f"Ingested {result['num_chunks']} chunks")

    # Test retrieval
    print("\n" + "="*70)
    print("Testing Retrieval with New Metadata")
    print("="*70)

    retrieval_results = pipeline.retriever.retrieve("machine learning", top_k=3)

    print(f"\nRetrieved {len(retrieval_results)} results:")
    for i, result in enumerate(retrieval_results, 1):
        metadata = result['metadata']
        print(f"\n{i}. Source: {metadata.get('source', 'N/A')}")
        print(f"   Chunk: {int(metadata.get('chunk', 0)) + 1}/{int(metadata.get('total_chunks', 0))}")
        print(f"   Score: {result.get('score', 0):.4f}")
        print(f"   Text: {result['text'][:100]}...")

        # Check metadata keys
        print(f"   Metadata keys: {list(metadata.keys())}")

    # Verification
    print("\n" + "="*70)
    print("Verification")
    print("="*70)

    all_sources = [r['metadata']['source'] for r in retrieval_results]
    print(f"\nSources: {all_sources}")

    has_filenames = all(s.endswith('.pdf') for s in all_sources)
    no_redundant = all('filename' not in r['metadata'] and 'path' not in r['metadata']
                       for r in retrieval_results)

    if has_filenames:
        print("\nSUCCESS: 'source' field contains PDF filenames!")
    else:
        print("\nWARNING: Some sources don't have .pdf extension")

    if no_redundant:
        print("SUCCESS: No redundant 'filename' or 'path' fields!")
    else:
        print("WARNING: Redundant fields found in metadata")

    print("\n" + "="*70)
    print("Final Metadata Structure")
    print("="*70)
    print("""
Each chunk in Pinecone now contains:
- source: PDF filename (e.g., "ai_topic_01.pdf")
- chunk: Chunk index (0-based integer)
- total_chunks: Total chunks in document
- text: The actual chunk content

Clean and efficient!
    """)


if __name__ == "__main__":
    main()
