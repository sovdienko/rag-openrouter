"""Generator module for LLM-based answer generation"""

from typing import List, Iterator
from openai import OpenAI
from .config import Config


class Generator:
    """Generator for creating answers using LLM"""

    def __init__(self, config: Config):
        """
        Initialize generator

        Args:
            config: Application configuration
        """
        self.config = config
        self.client = OpenAI(
            base_url=config.openrouter_base_url,
            api_key=config.openrouter_api_key
        )

    def generate(
        self,
        query: str,
        context: List[str],
        system_prompt: str = None
    ) -> str:
        """
        Generate answer based on query and context

        Args:
            query: User question
            context: List of relevant text chunks
            system_prompt: Optional custom system prompt

        Returns:
            Generated answer
        """
        if system_prompt is None:
            system_prompt = "You are a helpful assistant that answers questions based on provided context."

        # Build context string
        context_str = "\n\n".join(context)

        # Build user prompt
        user_prompt = f"""Answer the question based on the context below. If the context doesn't contain relevant information, say so.

Context:
{context_str}

Question: {query}

Answer:"""

        # Generate response
        response = self.client.chat.completions.create(
            model=self.config.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )

        return response.choices[0].message.content

    def generate_with_metadata(
        self,
        query: str,
        context: List[str],
        system_prompt: str = None
    ) -> dict:
        """
        Generate answer with additional metadata

        Args:
            query: User question
            context: List of relevant text chunks
            system_prompt: Optional custom system prompt

        Returns:
            Dictionary with answer and metadata
        """
        answer = self.generate(query, context, system_prompt)

        return {
            "answer": answer,
            "sources": context,
            "model": self.config.llm_model,
            "num_sources": len(context)
        }

    def generate_stream(
        self,
        query: str,
        context: List[str],
        system_prompt: str = None
    ) -> Iterator[str]:
        """
        Generate answer with streaming for real-time display

        Args:
            query: User question
            context: List of relevant text chunks
            system_prompt: Optional custom system prompt

        Yields:
            Text chunks as they are generated

        Example:
            for chunk in generator.generate_stream(query, context):
                print(chunk, end="", flush=True)
        """
        if system_prompt is None:
            system_prompt = "You are a helpful assistant that answers questions based on provided context."

        # Build context string
        context_str = "\n\n".join(context)

        # Build user prompt
        user_prompt = f"""Answer the question based on the context below. If the context doesn't contain relevant information, say so.

Context:
{context_str}

Question: {query}

Answer:"""

        # Generate streaming response
        stream = self.client.chat.completions.create(
            model=self.config.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            stream=True
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def generate_stream_with_metadata(
        self,
        query: str,
        context: List[str],
        system_prompt: str = None
    ) -> dict:
        """
        Generate streaming answer with metadata

        Args:
            query: User question
            context: List of relevant text chunks
            system_prompt: Optional custom system prompt

        Returns:
            Dictionary with stream iterator and metadata

        Example:
            result = generator.generate_stream_with_metadata(query, context)
            for chunk in result['stream']:
                print(chunk, end="", flush=True)
            print(f"\\nUsed {result['num_sources']} sources")
        """
        return {
            "stream": self.generate_stream(query, context, system_prompt),
            "sources": context,
            "model": self.config.llm_model,
            "num_sources": len(context)
        }
