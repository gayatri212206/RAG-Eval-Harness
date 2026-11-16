
from pathlib import Path
from pypdf import PdfReader


# Project document directory
DOCUMENT_DIR = Path(__file__).parent.parent / "data" / "documents"


def load_txt(file_path):
    """Read text from a TXT file."""
    return file_path.read_text(encoding="utf-8")


def load_pdf(file_path):
    """Extract text from a PDF file."""
    reader = PdfReader(str(file_path))

    pages = []

    for page in reader.pages:
        text = page.extract_text()

        if text:
            pages.append(text)

    return "\n".join(pages)


def load_documents():
    """Load all supported documents from the directory."""

    documents = []

    for file_path in DOCUMENT_DIR.rglob("*"):

        if not file_path.is_file():
            continue

        if file_path.suffix.lower() == ".txt":
            content = load_txt(file_path)

        elif file_path.suffix.lower() == ".pdf":
            content = load_pdf(file_path)

        else:
            continue

        if content.strip():
            documents.append({
                "content": content,
                "metadata": {
                    "source": file_path.name,
                    "file_type": file_path.suffix.lower()
                }
            })

    return documents


if __name__ == "__main__":

    docs = load_documents()

    print(f"\nTotal documents loaded: {len(docs)}")

    for doc in docs:
        print("\nSource:", doc["metadata"]["source"])
        print("Characters:", len(doc["content"]))
        print("Preview:", doc["content"][:200])
