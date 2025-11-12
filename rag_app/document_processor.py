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
        source_prefix: str = "doc",
        doc_metadata: List[Dict[str, Any]] = None
    ) -> tuple[List[str], List[Dict[str, Any]]]:
        """
        Process documents into chunks with metadata

        Handles both single document (str) and multiple documents (List[str])

        Args:
            documents: Single document string or list of document strings
            source_prefix: Prefix for source identifiers (or exact ID for single doc)
            doc_metadata: Optional list of metadata dicts for each document
                         (e.g., [{"filename": "doc1.pdf", "path": "/path/to/doc1.pdf"}])

        Returns:
            Tuple of (chunks, metadata)

        Examples:
            # Single document
            chunks, meta = processor.process_documents("Text here", source_prefix="my_doc")
            # Result: source = "my_doc_0"

            # Multiple documents with metadata
            docs = ["Doc 1", "Doc 2"]
            metadata = [{"filename": "doc1.pdf"}, {"filename": "doc2.pdf"}]
            chunks, meta = processor.process_documents(docs, doc_metadata=metadata)
            # Result: metadata includes filename for each chunk
        """
        # Normalize input to list
        if isinstance(documents, str):
            documents = [documents]

        chunks = []
        metadata = []

        for i, doc in enumerate(documents):
            doc_chunks = self.splitter.split_text(doc)
            chunks.extend(doc_chunks)

            # Determine source identifier
            if doc_metadata and i < len(doc_metadata) and 'filename' in doc_metadata[i]:
                # Use filename as source if available
                source_id = doc_metadata[i]['filename']
            else:
                # Fallback to prefix_index format
                source_id = f"{source_prefix}_{i}"

            # Base metadata for each chunk
            for j in range(len(doc_chunks)):
                chunk_meta = {
                    "source": source_id,
                    "chunk": j,
                    "total_chunks": len(doc_chunks)
                }

                # Note: filename and path are NOT added to metadata
                # as 'source' now contains the filename

                metadata.append(chunk_meta)

        return chunks, metadata
