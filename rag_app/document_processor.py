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
        documents: List[str],
        source_prefix: str = "doc"
    ) -> tuple[List[str], List[Dict[str, Any]]]:
        """
        Process documents into chunks with metadata

        Args:
            documents: List of document strings
            source_prefix: Prefix for source identifiers

        Returns:
            Tuple of (chunks, metadata)
        """
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

    def process_single_document(
        self,
        document: str,
        source_id: str = "doc_0"
    ) -> tuple[List[str], List[Dict[str, Any]]]:
        """
        Process a single document

        Args:
            document: Document string
            source_id: Source identifier

        Returns:
            Tuple of (chunks, metadata)
        """
        chunks = self.splitter.split_text(document)
        metadata = [
            {
                "source": source_id,
                "chunk": i,
                "total_chunks": len(chunks)
            }
            for i in range(len(chunks))
        ]

        return chunks, metadata
