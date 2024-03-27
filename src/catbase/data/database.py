import os

import chromadb
import openai
import pandas as pd
import tiktoken
from dotenv import load_dotenv
from langchain.text_splitter import SentenceTransformersTokenTextSplitter
from llama_index.core import Document

from catbase.analysis import plot_UMAP
from catbase.utils import Chunker, get_embedding_function

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma")


def create_database(
    database_name: str, documents: list[Document]
) -> chromadb.Collection:
    """Create a database collection and add documents to it.

    Args:
        database_name: The name of the database collection.
        documents: A list of documents to be added to the collection.

    Returns:
        The created database collection.

    """
    chunker = Chunker(
        chunk_size=3000,
        chunk_overlap=500,
        tokenizer=tiktoken.get_encoding("cl100k_base"),
        token_splitter=SentenceTransformersTokenTextSplitter(),
    )
    chunked_docs = chunker.chunk_documents(documents)

    print(f"Chunked them into {len(chunked_docs)} documents")

    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
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


def get_database(database_name: str) -> chromadb.Collection:
    """Get a database collection by name."""
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    return chroma_client.get_collection(name=database_name)


def list_databases() -> None:
    """List all collections of persistent client."""
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collections = chroma_client.list_collections()

    for collection in collections:
        print(collection.name)


def delete_database(database_name: str) -> None:
    """List all collections of persistent client."""
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    chroma_client.delete_collection(name=database_name)


def inspect_database(database_name: str, plot: bool = False) -> None:
    """Inspect a database collection."""
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = chroma_client.get_collection(name=database_name)

    print(f"Collection: {collection.name}")
    inspection_df = pd.DataFrame(collection.peek())
    print(inspection_df.head())

    # number of documents is the ids column but without _chunk_{i} where i is the chunk number
    # first, delete the trailing _chunk_{i} from the ids column
    num_documents = (
        inspection_df["ids"].apply(lambda x: x.split("_chunk")[0]).nunique()
    )
    print(f"Number of documents: {num_documents}")
    print(inspection_df["metadatas"][0]["Summary"])
    print(inspection_df["metadatas"][0].keys())

    if plot:
        embeddings = inspection_df["embeddings"]
        plot_UMAP(embeddings)

    # TODO add some nicer human-readable format in here
    # TODO some visualization of the embeddings?
    # TODO find nicer summaries of chunks
