"""Classes and functions for processing documents into a vector database."""

from typing import Any

import openai
from chromadb import Documents, EmbeddingFunction, Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter


class Chunker:
    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int,
        tokenizer: Any,
        token_splitter: Any,
    ) -> None:
        """Initialize the Chunker class to chunk content (full documents or text) into smaller chunks.

        Args:
            chunk_size (int): The size of each chunk.
            chunk_overlap (int): The overlap between chunks.
            tokenizer (Any): The tokenizer object.
            token_splitter (Any): The token splitter object.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = tokenizer
        self.token_splitter = token_splitter

    def tiktoken_len(self, text_content: str) -> int:
        """Calculates the number of tokens in the given text content.

        Args:
            text_content (str): The text content to be tokenized.

        Returns:
                int: The number of tokens in the text content.
        """
        tokens = self.tokenizer.encode(text_content, disallowed_special=())
        return len(tokens)

    def create_text_chunks(self, text_content: str) -> list[str]:
        """Create text chunks from the given text content.

        Args:
            text_content (str): The text content to be split into chunks.
            The separators used for splitting are: ["\n\n", "\n", " ", ""].

        Returns:
            List[str]: The list of text chunks.
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


# TODO potentially interesting, from advanced biomedical tutorial https://colab.research.google.com/drive/1CpsOiLiLYKeGrhmq579_FmtGsD5uZ3Qe#scrollTo=5MUDQzCUp8cf
# class Chunker:
#     def __init__(self, context_window=3000, max_windows=5):
#         self.context_window = context_window
#         self.max_windows = max_windows
#         self.window_overlap = 0.02

#     def __call__(self, paper):
#         snippet_idx = 0

#         while snippet_idx < self.max_windows and paper:
#             endpos = int(self.context_window * (1.0 + self.window_overlap))
#             snippet, paper = paper[:endpos], paper[endpos:]

#             next_newline_pos = snippet.rfind('\n')
#             if paper and next_newline_pos != -1 and next_newline_pos >= self.context_window // 2:
#                 paper = snippet[next_newline_pos+1:] + paper
#                 snippet = snippet[:next_newline_pos]


#             yield snippet_idx, snippet.strip()
#             snippet_idx += 1
class EmbeddingGenerator(EmbeddingFunction):
    def __init__(self, embedding_model):
        """Initializes an EmbeddingGenerator object.

        Args:
            embedding_model: The embedding model to be used for generating embeddings.
        """
        self.embedding_model = embedding_model

    def __call__(self, input: Documents) -> Embeddings:
        """Generates embeddings for the given input documents.

        Args:
            input: The input documents to generate embeddings for.

        Returns:
            The generated embeddings.
        """
        response = openai.embeddings.create(input=input, model=self.embedding_model)
        embeddings = response.data[0].embedding

        return embeddings
