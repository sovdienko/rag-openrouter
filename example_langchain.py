"""Example demonstrating LangChain RAG integration"""

from rag_app import LangChainRAG


def main():
    print("="*70)
    print("LangChain RAG Demo: Rapid Development with High-Level Abstractions")
    print("="*70)

    # Example 1: Quick start with PDF files
    print("\n" + "="*70)
    print("Example 1: Quick Start - Load PDFs and Query")
    print("="*70)

    # Initialize LangChain RAG
    print("\nInitializing LangChain RAG pipeline...")
    rag = LangChainRAG(
        model_name="meta-llama/llama-3.3-70b-instruct",
        temperature=0.7,
        chunk_size=1000,
        chunk_overlap=200
    )

    # Load documents from folder
    print("Loading PDF documents from 'rag-docs' folder...")
    documents = rag.load_from_folder("rag-docs")
    print(f"Loaded {len(documents)} PDF pages")

    # Ingest documents
    print("\nIngesting documents...")
    stats = rag.ingest_documents(documents, top_k=3)
    print(f"Ingested {stats['num_documents']} documents")
    print(f"Created {stats['num_chunks']} chunks")
    print(f"Status: {stats['status']}")

    # Query the system
    print("\n" + "="*70)
    print("Querying the System")
    print("="*70)

    questions = [
        "What is machine learning?",
        "Explain neural networks",
        "What are the main types of RAG systems?"
    ]

    for question in questions:
        print(f"\nQuestion: {question}")
        result = rag.query(question)
        print(f"\nAnswer: {result['answer']}")
        print(f"Sources used: {result['num_sources']}")

    # Example 2: Similarity search without LLM
    print("\n" + "="*70)
    print("Example 2: Similarity Search (No LLM Generation)")
    print("="*70)

    query = "deep learning transformers"
    print(f"\nSearching for: {query}")

    results = rag.similarity_search(query, top_k=3)
    print(f"\nFound {len(results)} relevant chunks:")

    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['text'][:150]}...")
        if result['metadata']:
            print(f"   Metadata: {result['metadata']}")

    # Example 3: Save and load vector store
    print("\n" + "="*70)
    print("Example 3: Persisting Vector Store")
    print("="*70)

    vectorstore_path = "langchain_vectorstore"

    print(f"\nSaving vector store to '{vectorstore_path}'...")
    rag.save_vectorstore(vectorstore_path)
    print("Vector store saved!")

    # Create new instance and load
    print("\nCreating new RAG instance and loading vector store...")
    rag2 = LangChainRAG()
    rag2.load_vectorstore(vectorstore_path, top_k=3)
    print("Vector store loaded successfully!")

    # Query the loaded system
    test_question = "What is artificial intelligence?"
    print(f"\nTesting loaded system with: {test_question}")
    result = rag2.query(test_question)
    print(f"Answer: {result['answer'][:200]}...")

    # Example 4: Working with text strings
    print("\n" + "="*70)
    print("Example 4: Ingesting Text Strings (No PDFs)")
    print("="*70)

    # Create new RAG instance
    rag_text = LangChainRAG()

    # Sample texts
    texts = [
        "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.",
        "Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes (neurons) organized in layers.",
        "Deep learning is a type of machine learning based on artificial neural networks with multiple layers. It's particularly effective for complex pattern recognition.",
        "Natural language processing (NLP) is a branch of AI that helps computers understand, interpret, and manipulate human language.",
        "Computer vision is a field of AI that trains computers to interpret and understand the visual world using digital images and videos."
    ]

    print(f"\nIngesting {len(texts)} text strings...")
    stats = rag_text.ingest_documents(texts, top_k=2)
    print(f"Created {stats['num_chunks']} chunks")

    question = "What is deep learning?"
    print(f"\nQuestion: {question}")
    result = rag_text.query(question)
    print(f"Answer: {result['answer']}")

    # Example 5: Adjusting retrieval parameters
    print("\n" + "="*70)
    print("Example 5: Dynamic Retrieval Configuration")
    print("="*70)

    print("\nUpdating retriever to fetch 5 documents instead of 3...")
    rag.update_retriever_k(top_k=5)

    question = "What are transformers in AI?"
    print(f"\nQuestion: {question}")
    result = rag.query(question)
    print(f"Answer: {result['answer'][:200]}...")
    print(f"Sources used: {result['num_sources']}")

    # Example 6: Statistics
    print("\n" + "="*70)
    print("Example 6: Pipeline Statistics")
    print("="*70)

    stats = rag.get_stats()
    print(f"\nVector store statistics:")
    print(f"  Number of vectors: {stats['num_vectors']}")
    print(f"  Status: {stats['status']}")

    # Benefits explanation
    print("\n" + "="*70)
    print("LangChain Integration Benefits")
    print("="*70)
    print("""
LangChain provides high-level abstractions for rapid RAG development:

1. Simplified API:
   - Fewer lines of code compared to custom implementation
   - Pre-built chains for common patterns
   - Automatic prompt engineering

2. Built-in Components:
   - Document loaders for PDFs, web pages, databases
   - Text splitters with multiple strategies
   - Vector stores (FAISS, Pinecone, Chroma, etc.)
   - LLM integrations (OpenAI, Anthropic, Hugging Face)

3. Quick Prototyping:
   - Rapid experimentation with different models
   - Easy switching between vector stores
   - Pre-configured retrieval chains

4. Local Development:
   - FAISS vector store runs locally (no cloud required)
   - Save/load vector stores for persistence
   - No external vector database needed

5. Flexibility:
   - Easy to customize chain components
   - Support for multiple retrieval strategies
   - Extensible with custom chains

When to Use LangChain vs Custom Pipeline:

Use LangChain when:
- Building MVPs or prototypes quickly
- Experimenting with different models/approaches
- Need local vector storage
- Want built-in prompt engineering
- Prefer declarative, high-level code

Use Custom Pipeline when:
- Need fine-grained control over components
- Optimizing for production performance
- Building domain-specific workflows
- Integrating with existing systems
- Want minimal dependencies

Example Comparison:

# LangChain (5 lines)
rag = LangChainRAG()
docs = rag.load_from_folder("docs")
rag.ingest_documents(docs)
result = rag.query("What is AI?")
print(result['answer'])

# Custom Pipeline (More control)
pipeline = RAGPipeline(use_hybrid=True, use_reranker=True)
loader = PDFLoader()
docs = loader.load_with_metadata("docs")
pipeline.ingest_documents(docs)
result = pipeline.query("What is AI?", top_k=5, initial_k=20)
print(result['answer'])

Both approaches are valid - choose based on your needs!
    """)


if __name__ == "__main__":
    main()
