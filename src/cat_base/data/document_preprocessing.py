import chromadb

from cat_base.data.database_creation import create_database_from_documents
from cat_base.data.parsing import get_arxiv_documents, parse_documents


def create_database(database_name: str, pdf_directory: str) -> None:
    """Create a new database."""
    documents = parse_documents(pdf_directory)

    create_database_from_documents(database_name, documents)


def create_database_from_arxiv(
    database_name, keyword_list, max_docs
) -> chromadb.Collection:
    """Create a new database from arXiv."""
    documents = get_arxiv_documents(keyword_list, max_docs)

    return create_database_from_documents(database_name, documents)


def list_databases() -> None:
    """List all collections of persistent client."""
    chroma_client = chromadb.PersistentClient()
    collections = chroma_client.list_collections()

    for collection in collections:
        print(collection.name)
