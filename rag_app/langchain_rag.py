"""LangChain integration for rapid RAG development"""

from typing import List, Dict, Any, Optional
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from .config import Config


class LangChainRAG:
    """
    High-level LangChain-based RAG pipeline

    Provides simplified interface using LangChain's abstractions
    for rapid development and prototyping.

    Use cases:
    - Quick RAG prototypes and MVPs
    - Experimentation with different models
    - Local vector storage with FAISS
    - Simplified API compared to full pipeline
    """

    def __init__(
        self,
        config: Config = None,
        model_name: str = "meta-llama/llama-3.3-70b-instruct",
        temperature: float = 0.7,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """
        Initialize LangChain RAG pipeline

        Args:
            config: Application configuration (uses env defaults if not provided)
            model_name: LLM model to use via OpenRouter
            temperature: LLM temperature (0-1)
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        if config is None:
            config = Config.from_env()

        self.config = config

        # Configure LLM via OpenRouter
        self.llm = ChatOpenAI(
            model_name=model_name,
            openai_api_key=config.openrouter_api_key,
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=temperature,
            default_headers={
                "HTTP-Referer": "https://github.com/sovdienko/rag-openrouter",
                "X-Title": "RAG OpenRouter"
            }
        )

        # Configure embeddings (OpenRouter or direct OpenAI)
        self.embeddings = OpenAIEmbeddings(
            model=config.embedding_model.replace("openai/", ""),
            openai_api_key=config.openrouter_api_key,
            openai_api_base="https://openrouter.ai/api/v1"
        )

        # Text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True
        )

        self.vectorstore: Optional[FAISS] = None
        self.qa_chain: Optional[RetrievalQA] = None

    def load_pdfs(self, pdf_paths: List[str]) -> List[Document]:
        """
        Load documents from PDF files

        Args:
            pdf_paths: List of PDF file paths

        Returns:
            List of LangChain Document objects

        Example:
            docs = rag.load_pdfs(["doc1.pdf", "doc2.pdf"])
        """
        documents = []
        for pdf_path in pdf_paths:
            try:
                loader = PyPDFLoader(pdf_path)
                documents.extend(loader.load())
            except Exception as e:
                print(f"Error loading {pdf_path}: {e}")
                continue
        return documents

    def load_from_folder(self, folder_path: str) -> List[Document]:
        """
        Load all PDF files from a folder

        Args:
            folder_path: Path to folder containing PDFs

        Returns:
            List of LangChain Document objects

        Example:
            docs = rag.load_from_folder("rag-docs")
        """
        import glob
        pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))
        return self.load_pdfs(pdf_files)

    def ingest_documents(
        self,
        documents: List[Document] | List[str],
        top_k: int = 3
    ) -> Dict[str, Any]:
        """
        Ingest documents and create RAG chain

        Args:
            documents: List of Document objects or text strings
            top_k: Number of documents to retrieve per query

        Returns:
            Dictionary with ingestion statistics

        Example:
            # From PDF documents
            docs = rag.load_from_folder("rag-docs")
            stats = rag.ingest_documents(docs)

            # From text strings
            texts = ["Text 1", "Text 2"]
            stats = rag.ingest_documents(texts)
        """
        # Convert strings to Documents if needed
        if documents and isinstance(documents[0], str):
            documents = [Document(page_content=text) for text in documents]

        # Split documents into chunks
        chunks = self.text_splitter.split_documents(documents)

        # Create vector store
        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)

        # Create retrieval chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": top_k}),
            return_source_documents=True
        )

        return {
            "num_documents": len(documents),
            "num_chunks": len(chunks),
            "status": "success"
        }

    def query(
        self,
        question: str,
        return_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Query the RAG system

        Args:
            question: User question
            return_sources: Whether to include source documents

        Returns:
            Dictionary with answer and metadata

        Example:
            result = rag.query("What is machine learning?")
            print(result['answer'])
            print(f"Sources: {result['num_sources']}")
        """
        if self.qa_chain is None:
            raise ValueError("No documents ingested. Call ingest_documents() first.")

        # Query the chain
        result = self.qa_chain({"query": question})

        response = {
            "answer": result["result"],
            "num_sources": len(result.get("source_documents", []))
        }

        if return_sources:
            response["sources"] = [
                doc.page_content for doc in result.get("source_documents", [])
            ]

        return response

    def similarity_search(
        self,
        query: str,
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Perform similarity search without LLM generation

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of relevant documents with metadata

        Example:
            results = rag.similarity_search("neural networks", top_k=5)
            for result in results:
                print(result['text'][:100])
        """
        if self.vectorstore is None:
            raise ValueError("No documents ingested. Call ingest_documents() first.")

        docs = self.vectorstore.similarity_search(query, k=top_k)

        return [
            {
                "text": doc.page_content,
                "metadata": doc.metadata
            }
            for doc in docs
        ]

    def save_vectorstore(self, path: str):
        """
        Save vector store to disk

        Args:
            path: Directory path to save to

        Example:
            rag.save_vectorstore("vectorstore")
        """
        if self.vectorstore is None:
            raise ValueError("No vector store to save")

        self.vectorstore.save_local(path)

    def load_vectorstore(self, path: str, top_k: int = 3):
        """
        Load vector store from disk

        Args:
            path: Directory path to load from
            top_k: Number of documents to retrieve per query

        Example:
            rag.load_vectorstore("vectorstore")
        """
        self.vectorstore = FAISS.load_local(
            path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )

        # Recreate QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": top_k}),
            return_source_documents=True
        )

    def update_retriever_k(self, top_k: int):
        """
        Update the number of documents retrieved

        Args:
            top_k: New number of documents to retrieve

        Example:
            rag.update_retriever_k(5)  # Retrieve 5 docs instead of 3
        """
        if self.vectorstore is None:
            raise ValueError("No vector store loaded")

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": top_k}),
            return_source_documents=True
        )

    def get_stats(self) -> Dict[str, Any]:
        """
        Get pipeline statistics

        Returns:
            Dictionary with statistics

        Example:
            stats = rag.get_stats()
            print(f"Vector store size: {stats['num_vectors']}")
        """
        if self.vectorstore is None:
            return {
                "num_vectors": 0,
                "status": "no_vectorstore"
            }

        # FAISS doesn't expose count directly, so we need to check index
        return {
            "num_vectors": self.vectorstore.index.ntotal,
            "status": "ready"
        }
