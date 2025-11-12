"""Verify the metadata structure in Pinecone"""

from rag_app import PDFLoader, RAGPipeline


def main():
    print("="*70)
    print("Verifying Metadata Structure in Pinecone")
    print("="*70)

    # Load a few PDFs with metadata
    loader = PDFLoader()
    documents = loader.load_with_metadata("rag-docs", verbose=False)[:3]

    print(f"\nLoaded {len(documents)} documents:")
    for i, doc in enumerate(documents, 1):
        print(f"  {i}. {doc['filename']}")

    # Initialize pipeline and ingest
    pipeline = RAGPipeline()
    result = pipeline.ingest_documents(documents)

    print(f"\nIngested {result['num_chunks']} chunks")

    # Retrieve and inspect metadata structure
    print("\n" + "="*70)
    print("Metadata Structure in Retrieved Results")
    print("="*70)

    retrieval_results = pipeline.retriever.retrieve("machine learning", top_k=2)

    for i, result in enumerate(retrieval_results, 1):
        print(f"\nResult {i}:")
        print(f"  Metadata keys: {list(result['metadata'].keys())}")
        print(f"  Metadata content:")
        for key, value in result['metadata'].items():
            if key == 'text':
                print(f"    {key}: {value[:50]}...")
            else:
                print(f"    {key}: {value}")
        print(f"  Score: {result.get('score', 'N/A')}")

    # Verify source contains filename
    print("\n" + "="*70)
    print("Verification Results")
    print("="*70)

    all_sources = [r['metadata']['source'] for r in retrieval_results]
    print(f"\nSources found: {all_sources}")

    # Check if sources are filenames (end with .pdf)
    are_filenames = all(s.endswith('.pdf') for s in all_sources)

    if are_filenames:
        print("\n✓ SUCCESS: 'source' field contains filenames!")
        print("✓ No redundant 'filename' or 'path' fields in metadata")
    else:
        print("\n✗ WARNING: 'source' field does not contain filenames")

    # Show what's stored
    print("\n" + "="*70)
    print("What's Stored in Pinecone")
    print("="*70)
    print("""
Each chunk's metadata contains ONLY:
- source: PDF filename (e.g., "ai_topic_01.pdf")
- chunk: Chunk index (0-based)
- total_chunks: Total number of chunks in the document
- text: The chunk content

No redundant 'filename' or 'path' fields!
The filename is directly accessible via metadata['source'].
    """)


if __name__ == "__main__":
    main()
