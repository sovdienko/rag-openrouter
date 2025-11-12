"""PDF document loader module"""

from pathlib import Path
from typing import List, Dict, Any
from pypdf import PdfReader


class PDFLoader:
    """Loader for PDF documents"""

    def __init__(self, folder_path: str = None):
        """
        Initialize PDF loader

        Args:
            folder_path: Optional default folder path for loading PDFs
        """
        self.folder_path = folder_path

    def load_from_folder(
        self,
        folder_path: str = None,
        pattern: str = "*.pdf",
        verbose: bool = True
    ) -> List[str]:
        """
        Load documents from all PDF files in a folder

        Args:
            folder_path: Path to folder containing PDF files (uses default if not provided)
            pattern: Glob pattern for PDF files (default: "*.pdf")
            verbose: Whether to print loading progress

        Returns:
            List of document texts

        Example:
            loader = PDFLoader()
            documents = loader.load_from_folder("rag-docs")
        """
        docs_with_metadata = self.load_with_metadata(
            folder_path=folder_path,
            pattern=pattern,
            verbose=verbose
        )
        return [doc["text"] for doc in docs_with_metadata]

    def load_single_file(self, file_path: str | Path) -> str:
        """
        Load text from a single PDF file

        Args:
            file_path: Path to PDF file

        Returns:
            Extracted text from all pages

        Example:
            loader = PDFLoader()
            text = loader.load_single_file("document.pdf")
        """
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text

    def _load_files_with_metadata(
        self,
        pdf_files: List[Path],
        verbose: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Internal helper to load multiple PDF files with metadata

        Args:
            pdf_files: List of Path objects to PDF files
            verbose: Whether to print loading progress

        Returns:
            List of dictionaries with 'text', 'filename', and 'path' keys
        """
        documents = []

        for pdf_file in pdf_files:
            try:
                text = self.load_single_file(pdf_file)
                if text.strip():
                    documents.append({
                        "text": text.strip(),
                        "filename": pdf_file.name,
                        "path": str(pdf_file)
                    })
                    if verbose:
                        print(f"  Loaded: {pdf_file.name}")
            except Exception as e:
                if verbose:
                    print(f"  Error loading {pdf_file.name}: {e}")

        return documents

    def load_with_metadata(
        self,
        folder_path: str = None,
        pattern: str = "*.pdf",
        verbose: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Load documents with metadata (filename, path, etc.)

        Args:
            folder_path: Path to folder containing PDF files
            pattern: Glob pattern for PDF files
            verbose: Whether to print loading progress

        Returns:
            List of dictionaries with 'text', 'filename', and 'path' keys

        Example:
            loader = PDFLoader()
            docs = loader.load_with_metadata("rag-docs")
            for doc in docs:
                print(f"File: {doc['filename']}")
                print(f"Text: {doc['text'][:100]}...")
        """
        path = folder_path or self.folder_path
        if not path:
            raise ValueError("folder_path must be provided")

        pdf_files = sorted(Path(path).glob(pattern))

        if verbose:
            print(f"Loading {len(pdf_files)} PDF files from '{path}'...")

        documents = self._load_files_with_metadata(pdf_files, verbose=verbose)

        if verbose:
            print(f"Successfully loaded {len(documents)} documents\n")

        return documents
