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

    def _build_prompt(
        self,
        query: str,
        context: List[str],
        system_prompt: str = None
    ) -> tuple[str, str]:
        """
        Build system and user prompts

        Args:
            query: User question
            context: List of relevant text chunks
            system_prompt: Optional custom system prompt

        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        if system_prompt is None:
            system_prompt = "You are a helpful assistant that answers questions based on provided context."

        context_str = "\n\n".join(context)

        user_prompt = f"""Answer the question based on the context below. If the context doesn't contain relevant information, say so.

Context:
{context_str}

Question: {query}

Answer:"""

        return system_prompt, user_prompt

    def _build_messages(
        self,
        query: str,
        context: List[str],
        system_prompt: str = None
    ) -> List[dict]:
        """
        Build messages for API call

        Args:
            query: User question
            context: List of relevant text chunks
            system_prompt: Optional custom system prompt

        Returns:
            List of message dictionaries
        """
        system_prompt, user_prompt = self._build_prompt(query, context, system_prompt)
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

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
        messages = self._build_messages(query, context, system_prompt)

        response = self.client.chat.completions.create(
            model=self.config.llm_model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )

        return response.choices[0].message.content

    def _build_metadata(self, context: List[str]) -> dict:
        """
        Build metadata dictionary

        Args:
            context: List of relevant text chunks

        Returns:
            Dictionary with metadata
        """
        return {
            "sources": context,
            "model": self.config.llm_model,
            "num_sources": len(context)
        }

    def generate_with_metadata(
        self,
        query: str,
        context: List[str],
        system_prompt: str = None,
        stream: bool = False
    ) -> dict:
        """
        Generate answer with additional metadata

        Args:
            query: User question
            context: List of relevant text chunks
            system_prompt: Optional custom system prompt
            stream: If True, returns streaming response

        Returns:
            Dictionary with answer/stream and metadata

        Examples:
            # Non-streaming
            result = generator.generate_with_metadata(query, context)
            print(result['answer'])

            # Streaming
            result = generator.generate_with_metadata(query, context, stream=True)
            for chunk in result['stream']:
                print(chunk, end="", flush=True)
        """
        metadata = self._build_metadata(context)

        if stream:
            metadata["stream"] = self.generate_stream(query, context, system_prompt)
        else:
            metadata["answer"] = self.generate(query, context, system_prompt)

        return metadata

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
        messages = self._build_messages(query, context, system_prompt)

        stream = self.client.chat.completions.create(
            model=self.config.llm_model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            stream=True
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
