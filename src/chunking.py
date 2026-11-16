
import re


def simple_chunking(text, chunk_size=500, overlap=100):
    """
    Split text into fixed-size chunks with overlap.
    """

    if not text.strip():
        return []

    if overlap >= chunk_size:
        raise ValueError("Overlap must be smaller than chunk size.")

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def semantic_chunking(text, max_chunk_size=700):
    """
    Split text using paragraph and sentence boundaries.
    """

    if not text.strip():
        return []

    paragraphs = re.split(r"\n\s*\n", text.strip())

    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:

        paragraph = paragraph.strip()

        if not paragraph:
            continue

        if len(paragraph) > max_chunk_size:
            sentences = re.split(r"(?<=[.!?])\s+", paragraph)

            for sentence in sentences:

                if not sentence.strip():
                    continue

                if len(current_chunk) + len(sentence) > max_chunk_size:
                    if current_chunk:
                        chunks.append(current_chunk.strip())

                    current_chunk = sentence

                else:
                    current_chunk += " " + sentence

            continue

        if len(current_chunk) + len(paragraph) > max_chunk_size:

            if current_chunk:
                chunks.append(current_chunk.strip())

            current_chunk = paragraph

        else:
            current_chunk += "\n\n" + paragraph

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


if __name__ == "__main__":

    sample_text = """
    Python is a high-level programming language.

    Python supports object-oriented programming.

    Python is widely used in artificial intelligence.
    """

    simple_chunks = simple_chunking(sample_text)

    semantic_chunks = semantic_chunking(sample_text)

    print("\nSIMPLE CHUNKING")
    print("Total chunks:", len(simple_chunks))

    for i, chunk in enumerate(simple_chunks, 1):
        print(f"\nChunk {i}:\n{chunk}")

    print("\nSEMANTIC CHUNKING")
    print("Total chunks:", len(semantic_chunks))

    for i, chunk in enumerate(semantic_chunks, 1):
        print(f"\nChunk {i}:\n{chunk}")
