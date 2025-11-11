"""Main RAG pipeline orchestrator"""

from typing import List, Dict, Any
from .config import Config
from .embeddings import EmbeddingService
from .vector_store import VectorStore
from .document_processor import DocumentProcessor
from .retriever import Retriever
from .generator import Generator


class RAGPipeline:
    """Complete RAG pipeline orchestrator"""

    def __init__(self, config: Config = None):
        """
        Initialize RAG pipeline

        Args:
            config: Application configuration (uses env defaults if not provided)
        """
        if config is None:
            config = Config.from_env()

        config.validate()
        self.config = config

        # Initialize components
        self.embedding_service = EmbeddingService(config)
        self.vector_store = VectorStore(config)
        self.document_processor = DocumentProcessor(config)
        self.retriever = Retriever(config, self.embedding_service, self.vector_store)
        self.generator = Generator(config)

    def ingest_documents(
        self,
        documents: List[str],
        source_prefix: str = "doc"
    ) -> Dict[str, Any]:
        """
        Ingest documents into the RAG system

        Args:
            documents: List of document strings
            source_prefix: Prefix for source identifiers

        Returns:
            Dictionary with ingestion statistics
        """
        # Process documents into chunks
        chunks, metadata = self.document_processor.process_documents(
            documents, source_prefix
        )

        # Generate embeddings
        embeddings = self.embedding_service.embed_texts(chunks)

        # Store in vector database
        num_vectors = self.vector_store.upsert_vectors(chunks, embeddings, metadata)

        return {
            "num_documents": len(documents),
            "num_chunks": len(chunks),
            "num_vectors": num_vectors,
            "status": "success"
        }

    def query(
        self,
        question: str,
        top_k: int = None,
        return_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Query the RAG system

        Args:
            question: User question
            top_k: Number of chunks to retrieve
            return_sources: Whether to include source chunks in response

        Returns:
            Dictionary with answer and metadata
        """
        # Retrieve relevant chunks
        context = self.retriever.retrieve_texts(question, top_k=top_k)

        # Generate answer
        result = self.generator.generate_with_metadata(question, context)

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
        # Retrieve relevant chunks
        context = self.retriever.retrieve_texts(question, top_k=top_k)

        # Generate streaming answer
        result = self.generator.generate_stream_with_metadata(question, context)

        if not return_sources:
            result.pop("sources", None)

        return result

    def clear_index(self):
        """Clear all vectors from the index"""
        self.vector_store.delete_all()

    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        return self.vector_store.get_stats()
