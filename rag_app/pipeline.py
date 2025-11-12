"""Main RAG pipeline orchestrator"""

from typing import List, Dict, Any
from .config import Config
from .embeddings import EmbeddingService
from .vector_store import VectorStore
from .document_processor import DocumentProcessor
from .retriever import Retriever
from .generator import Generator
from .reranker import Reranker
from .hybrid_search import HybridSearcher


class RAGPipeline:
    """Complete RAG pipeline orchestrator"""

    def __init__(
        self,
        config: Config = None,
        use_reranker: bool = False,
        use_hybrid: bool = False,
        hybrid_alpha: float = 0.5
    ):
        """
        Initialize RAG pipeline

        Args:
            config: Application configuration (uses env defaults if not provided)
            use_reranker: Whether to enable two-stage retrieval with reranking
            use_hybrid: Whether to enable hybrid search (vector + keyword)
            hybrid_alpha: Weight for vector vs keyword in hybrid search (0-1)
                         alpha=1.0: pure vector, alpha=0.0: pure keyword, alpha=0.5: balanced
        """
        if config is None:
            config = Config.from_env()

        config.validate()
        self.config = config

        # Initialize components
        self.embedding_service = EmbeddingService(config)
        self.vector_store = VectorStore(config)
        self.document_processor = DocumentProcessor(config)

        # Initialize reranker if requested
        self.reranker = Reranker() if use_reranker else None

        # Initialize hybrid searcher if requested
        self.hybrid_searcher = HybridSearcher(alpha=hybrid_alpha) if use_hybrid else None

        self.retriever = Retriever(
            config,
            self.embedding_service,
            self.vector_store,
            self.reranker,
            self.hybrid_searcher
        )
        self.generator = Generator(config)

    def ingest_documents(
        self,
        documents: List[str] | List[Dict[str, Any]],
        source_prefix: str = "doc"
    ) -> Dict[str, Any]:
        """
        Ingest documents into the RAG system

        Args:
            documents: List of document strings OR list of dicts with 'text' and metadata
                      Examples:
                      - ["text1", "text2"]
                      - [{"text": "...", "filename": "doc1.pdf"}, {"text": "...", "filename": "doc2.pdf"}]
            source_prefix: Prefix for source identifiers

        Returns:
            Dictionary with ingestion statistics
        """
        # Extract texts and metadata if documents are dicts
        if documents and isinstance(documents[0], dict):
            texts = [doc["text"] for doc in documents]
            doc_metadata = [
                {k: v for k, v in doc.items() if k != "text"}
                for doc in documents
            ]
        else:
            texts = documents
            doc_metadata = None

        # Process documents into chunks
        chunks, metadata = self.document_processor.process_documents(
            texts, source_prefix, doc_metadata
        )

        # Generate embeddings
        embeddings = self.embedding_service.embed_texts(chunks)

        # Store in vector database
        num_vectors = self.vector_store.upsert_vectors(chunks, embeddings, metadata)

        return {
            "num_documents": len(texts),
            "num_chunks": len(chunks),
            "num_vectors": num_vectors,
            "status": "success"
        }

    def query(
        self,
        question: str,
        top_k: int = None,
        return_sources: bool = True,
        stream: bool = False,
        use_reranking: bool = None,
        use_hybrid: bool = None,
        initial_k: int = None
    ) -> Dict[str, Any]:
        """
        Query the RAG system

        Args:
            question: User question
            top_k: Number of chunks to retrieve (final count)
            return_sources: Whether to include source chunks in response
            stream: If True, returns streaming response
            use_reranking: Whether to use two-stage retrieval (default: True if reranker enabled)
            use_hybrid: Whether to use hybrid search (default: True if hybrid_searcher enabled)
            initial_k: Number of candidates for reranking/hybrid (default: top_k * 5)

        Returns:
            Dictionary with answer/stream and metadata

        Examples:
            # Standard retrieval
            result = pipeline.query("What is RAG?")
            print(result['answer'])

            # Hybrid search (vector + keyword)
            result = pipeline.query("What is RAG?", top_k=3, use_hybrid=True, initial_k=20)

            # With two-stage retrieval (retrieve 15, rerank to top 3)
            result = pipeline.query("What is RAG?", top_k=3, use_reranking=True, initial_k=15)

            # Hybrid + reranking (best quality)
            result = pipeline.query("What is RAG?", top_k=3, use_hybrid=True, use_reranking=True, initial_k=20)

            # Streaming
            result = pipeline.query("What is RAG?", stream=True)
            for chunk in result['stream']:
                print(chunk, end="", flush=True)
        """
        # Default: use reranking if reranker is available
        if use_reranking is None:
            use_reranking = self.reranker is not None

        # Default: use hybrid if hybrid_searcher is available
        if use_hybrid is None:
            use_hybrid = self.hybrid_searcher is not None

        # Retrieve relevant chunks
        context = self.retriever.retrieve_texts(
            question,
            top_k=top_k,
            use_reranking=use_reranking,
            use_hybrid=use_hybrid,
            initial_k=initial_k
        )

        # Generate answer
        result = self.generator.generate_with_metadata(question, context, stream=stream)

        if not return_sources:
            result.pop("sources", None)

        return result

    def query_stream(
        self,
        question: str,
        top_k: int = None,
        return_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Query the RAG system with streaming response

        Convenience wrapper for query(stream=True)

        Args:
            question: User question
            top_k: Number of chunks to retrieve
            return_sources: Whether to include source chunks in response

        Returns:
            Dictionary with stream iterator and metadata

        Example:
            result = pipeline.query_stream("What is RAG?")
            for chunk in result['stream']:
                print(chunk, end="", flush=True)
            print(f"\\nSources: {result['num_sources']}")
        """
        return self.query(question, top_k=top_k, return_sources=return_sources, stream=True)

    def clear_index(self):
        """Clear all vectors from the index"""
        self.vector_store.delete_all()

    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        return self.vector_store.get_stats()
