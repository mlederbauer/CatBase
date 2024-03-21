import os
from collections import defaultdict

import openai
from langchain_community.document_loaders import ArxivLoader
from llama_index.core import Document, SimpleDirectoryReader

openai.api_key = os.getenv("OPENAI_API_KEY")
chroma_db_path = os.getenv("CHROMA_DB_PATH")


# Parse metadata from 'document_texts'
def parse_metadata(document: Document) -> dict:
    document_text = document.text
    lines = document_text.split("\n")
    title = lines[0]  # Title is the first line
    authors_end_idx = 0
    # Authors are from the 2nd line until an '*' and a new line is encountered
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
        "Published": "2000-00-00",  # Placeholder
        "Title": title,
        "Authors": authors,
        "Summary": summary,
        "entry_id": document.id_,  # Assuming you want to keep the original Document ID
    }


def combine_documents(documents: list[Document]) -> list[Document]:
    # Step 1: Group documents by 'file_name'
    docs_by_file_name = defaultdict(list)
    for doc in documents:
        file_name = doc.metadata.get("file_name")
        docs_by_file_name[file_name].append(doc)

    # Step 2: Combine text for documents with the same 'file_name'
    combined_documents = []
    for file_name, docs in docs_by_file_name.items():
        combined_text = "\n\n".join([doc.text for doc in docs])

        # Assuming the first document's metadata (except text-related) is representative
        # for the combined document.
        combined_metadata = docs[0].metadata.copy()

        # Create a new Document object for the combined text.
        combined_doc = Document(text=combined_text, metadata=combined_metadata)  # type: ignore[call-arg]

        combined_documents.append(combined_doc)

    return combined_documents


def parse_documents(pdf_directory: str) -> list:
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
    """Parses documents from the Arxiv based on a list of keywords.

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
            # Initialize a dictionary with default values to avoid KeyError
            metadata = {
                "Published": doc.metadata.get("Published", "N/A"),
                "Title": doc.metadata.get("Title", "No Title"),
                "Authors": doc.metadata.get("Authors", []),
                "Summary": doc.metadata.get("Summary", "No Summary"),
                "entry_id": doc.metadata.get("entry_id", "No Entry ID"),
            }

            # Update the document's metadata with the filtered metadata
            doc.metadata = metadata

            # Create a Document object and add it to the all_documents list
            all_documents.append(Document(text=doc.page_content, metadata=doc.metadata))  # type: ignore[call-arg]

    return all_documents
