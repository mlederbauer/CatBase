from chromadb.api.types import Documents
from chromadb.utils import embedding_functions
from chromadb.utils.embedding_functions import EmbeddingFunction


def get_embedding_function() -> EmbeddingFunction[Documents] | None:
    return embedding_functions.DefaultEmbeddingFunction()
