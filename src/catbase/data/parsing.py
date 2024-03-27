import os
from collections import defaultdict

from langchain_community.document_loaders import ArxivLoader
from llama_index.core import Document, SimpleDirectoryReader


def parse_metadata(document: Document) -> dict[str, str]:
    """Parse metadata from a Document object.

    Args:
    - document (Document): The Document object to parse metadata from.

    Returns:
    - dict: A dictionary containing the parsed metadata.
    """
    document_text = document.text
    lines = document_text.split("\n")
    title = lines[0]  # Title is the first line
    authors_end_idx = 0
    authors_list = []
    for i, line in enumerate(lines[1:], start=1):  # Start from the second line
        if "*" in line:
            authors_end_idx = i
            authors_list.append(line.strip())
            break
        authors_list.append(line.strip())
    authors = ", ".join(authors_list)

    summary_lines = lines[
        authors_end_idx + 1 : authors_end_idx + 11
    ]  # Get next 10 lines
    summary = "\n".join(summary_lines)

    return {
        "Published": "2000-00-00",  # Placeholder FIXME
        "Title": title,
        "Authors": authors,
        "Summary": summary,
        "entry_id": document.id_,
    }


def combine_documents(documents: list[Document]) -> list[Document]:
    """Combine multiple Document objects into a single Document object.

    Args:
    - documents (List[Document]): The list of Document objects to combine.

    Returns:
    - List[Document]: A list of combined Document objects.
    """
    docs_by_file_name = defaultdict(list)
    for doc in documents:
        file_name = doc.metadata.get("file_name")
        docs_by_file_name[file_name].append(doc)

    combined_documents = []
    for file_name, docs in docs_by_file_name.items():
        combined_text = "\n\n".join([doc.text for doc in docs])
        combined_metadata = docs[0].metadata.copy()
        combined_doc = Document(text=combined_text, metadata=combined_metadata)  # type: ignore[call-arg]
        combined_documents.append(combined_doc)

    return combined_documents


def parse_documents(pdf_directory: str) -> list[Document]:
    """Parse documents from a directory containing PDF files.

    Args:
    - pdf_directory (str): The path to the directory containing PDF files.

    Returns:
    - List[Document]: A list of parsed Document objects.
    """
    input_files = [
        f"{pdf_directory}/{pdf}"
        for pdf in os.listdir(pdf_directory)
        if pdf.endswith(".pdf")
    ]

    documents = SimpleDirectoryReader(input_files=input_files).load_data()

    full_text_documents = combine_documents(documents)

    print(f"Loaded {len(documents)} pages from {len(input_files)} PDF files.")

    for doc in full_text_documents:
        metadata = parse_metadata(doc)
        doc.metadata = metadata

    return full_text_documents


def parse_arxiv(keywords: str, max_docs: int) -> list[Document]:
    """Parse documents from the Arxiv based on a list of keywords.

    Args:
    - keywords (str): A comma-separated string of keywords to search for.
    - max_docs (int): The maximum number of documents to load for each keyword.

    Returns:
    - List[Document]: A list of Document objects with filtered metadata.
    """
    all_documents = []

    for keyword in keywords.split(","):
        docs = ArxivLoader(
            query=keyword, load_max_docs=max_docs, load_all_available_meta=True
        ).load()
        print(
            f"Loaded {len(docs)} documents from arXiv with keyword '{keyword}'."
        )

        for doc in docs:
            metadata = {
                "Published": doc.metadata.get("Published", "N/A"),
                "Title": doc.metadata.get("Title", "No Title"),
                "Authors": doc.metadata.get("Authors", []),
                "Summary": doc.metadata.get("Summary", "No Summary"),
                "entry_id": doc.metadata.get("entry_id", "No Entry ID"),
            }
            doc.metadata = metadata
            all_documents.append(Document(text=doc.page_content, metadata=doc.metadata))  # type: ignore[call-arg]

    return all_documents
