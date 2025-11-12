"""RAG Application Package"""

from .pipeline import RAGPipeline
from .pdf_loader import PDFLoader
from .config import Config
from .classifier import DocumentClassifier, cosine_similarity
from .clustering import DocumentClusterer

__version__ = "1.0.0"

__all__ = [
    "RAGPipeline",
    "PDFLoader",
    "Config",
    "DocumentClassifier",
    "DocumentClusterer",
    "cosine_similarity"
]
