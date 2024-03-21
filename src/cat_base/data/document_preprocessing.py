import chromadb

from cat_base.data.database_creation import create_database_from_documents
from cat_base.data.parsing import parse_documents


def create_database(database_name: str, pdf_directory: str) -> None:
    """Create a new database."""
    documents = parse_documents(pdf_directory)

    create_database_from_documents(database_name, documents)


def list_databases() -> None:
    """List all collections of persistent client."""
    chroma_client = chromadb.PersistentClient()
    collections = chroma_client.list_collections()

    for collection in collections:
        print(collection.name)
