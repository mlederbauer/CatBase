from typing import Any

import tiktoken
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)
from llama_index.core import Document


class Chunker:
    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int,
        tokenizer: Any = None,
        token_splitter: Any = None,
    ) -> None:
        """Initialize the Chunker class to chunk content (full documents or text) into smaller chunks.

        Args:
            chunk_size: The size of each chunk.
            chunk_overlap: The overlap between consecutive chunks.
            tokenizer: The tokenizer to use for tokenization.
            token_splitter: The token splitter to use for splitting tokens.

        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = (
            tokenizer if tokenizer else tiktoken.get_encoding("cl100k_base")
        )
        self.token_splitter = (
            token_splitter
            if token_splitter
            else SentenceTransformersTokenTextSplitter()
        )

    def tiktoken_len(self, text_content: str) -> int:
        """Calculates the number of tokens in the given text content.

        Args:
            text_content: The text content to calculate the number of tokens for.

        Returns:
            The number of tokens in the text content.

        """
        tokens = self.tokenizer.encode(text_content, disallowed_special=())
        return len(tokens)

    def create_text_chunks(self, text_content: str) -> list[str]:
        """Create text chunks from the given text content.

        Args:
            text_content: The text content to create chunks from.

        Returns:
            A list of text chunks.

        """
        char_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " ", ""],
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=self.tiktoken_len,
        )
        character_split_texts = char_splitter.split_text(text_content)

        token_split_texts = []
        for text in character_split_texts:
            token_split_texts += self.token_splitter.split_text(text)

        return token_split_texts

    def chunk_documents(self, documents: list[Document]) -> list[Document]:
        """Chunks a list of Document objects and returns a new list of Document objects for each chunk.

        Args:
            documents: The list of Document objects to chunk.

        Returns:
            A new list of Document objects for each chunk.

        """
        chunked_docs = []
        for doc in documents:
            chunks = self.create_text_chunks(doc.text)  # type: ignore[attr-defined]
            base_entry_id = doc.metadata["entry_id"].split("_chunk_")[0]
            for chunk_index, chunk in enumerate(chunks):
                chunk_metadata = doc.metadata.copy()
                chunk_metadata[
                    "entry_id"
                ] = f"{base_entry_id}_chunk_{chunk_index}"
                chunk_to_add = Document(text=chunk, metadata=chunk_metadata)  # type: ignore[call-arg]
                chunked_docs.append(chunk_to_add)

        return chunked_docs
