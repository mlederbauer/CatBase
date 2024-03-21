import os

import chromadb
import openai
import tiktoken
from langchain.text_splitter import SentenceTransformersTokenTextSplitter

from cat_base.utils import Chunker
from cat_base.utils.embedding import get_embedding_function

openai.api_key = os.getenv("OPENAI_API_KEY")


def create_database(database_name, documents) -> chromadb.Collection:
    chunker = Chunker(
        chunk_size=3000,
        chunk_overlap=500,
        tokenizer=tiktoken.get_encoding("cl100k_base"),
        token_splitter=SentenceTransformersTokenTextSplitter(),
    )
    chunked_docs = chunker.chunk_documents(documents)

    print(f"Chunked them into {len(chunked_docs)} documents")

    chroma_client = chromadb.PersistentClient()
    embedding_function = get_embedding_function()

    # create a collection
    collection = chroma_client.get_or_create_collection(
        name=database_name,
        embedding_function=embedding_function,  # type: ignore[arg-type]
    )

    for chunked_doc in chunked_docs:
        collection.add(
            documents=[chunked_doc.text],
            metadatas=[chunked_doc.metadata],
            ids=[chunked_doc.metadata["entry_id"]],
        )

    return collection


def list_databases() -> None:
    """List all collections of persistent client."""
    chroma_client = chromadb.PersistentClient()
    collections = chroma_client.list_collections()

    for collection in collections:
        print(collection.name)
