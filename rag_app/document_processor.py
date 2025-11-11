"""Document processing module for chunking text"""

from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .config import Config


class DocumentProcessor:
    """Processor for splitting documents into chunks"""

    def __init__(self, config: Config):
        """
        Initialize document processor

        Args:
            config: Application configuration
        """
        self.config = config
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=config.separators
        )

    def process_documents(
        self,
        documents: List[str] | str,
        source_prefix: str = "doc"
    ) -> tuple[List[str], List[Dict[str, Any]]]:
        """
        Process documents into chunks with metadata

        Handles both single document (str) and multiple documents (List[str])

        Args:
            documents: Single document string or list of document strings
            source_prefix: Prefix for source identifiers (or exact ID for single doc)

        Returns:
            Tuple of (chunks, metadata)

        Examples:
            # Single document
            chunks, meta = processor.process_documents("Text here", source_prefix="my_doc")
            # Result: source = "my_doc_0"

            # Multiple documents
            chunks, meta = processor.process_documents(["Doc 1", "Doc 2"], source_prefix="doc")
            # Result: sources = "doc_0", "doc_1"
        """
        # Normalize input to list
        if isinstance(documents, str):
            documents = [documents]

        chunks = []
        metadata = []

        for i, doc in enumerate(documents):
            doc_chunks = self.splitter.split_text(doc)
            chunks.extend(doc_chunks)
            metadata.extend([
                {
                    "source": f"{source_prefix}_{i}",
                    "chunk": j,
                    "total_chunks": len(doc_chunks)
                }
                for j in range(len(doc_chunks))
            ])

        return chunks, metadata
