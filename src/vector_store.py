
import chromadb

from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

from src.ingestion import load_documents
from src.chunking import simple_chunking


class VectorStore:

    def __init__(self, db_path="./chroma_db"):

        self.client = chromadb.PersistentClient(
            path=db_path
        )

        self.embedding_function = DefaultEmbeddingFunction()

        self.collection = self.client.get_or_create_collection(
            name="rag_documents",
            embedding_function=self.embedding_function
        )

        print("ChromaDB connected successfully!")

    def add_documents(self, documents):

        ids = []
        texts = []
        metadatas = []

        for doc in documents:

            source = doc["metadata"]["source"]

            chunks = simple_chunking(
                doc["content"],
                chunk_size=500,
                overlap=100
            )

            for index, chunk in enumerate(chunks):

                ids.append(f"{source}_{index}")

                texts.append(chunk)

                metadatas.append({
                    "source": source,
                    "chunk_index": index,
                    "file_type": doc["metadata"]["file_type"]
                })

        if not texts:
            print("No chunks available to store.")
            return

        self.collection.upsert(
            ids=ids,
            documents=texts,
            metadatas=metadatas
        )

        print(f"Successfully stored {len(texts)} chunks!")

    def search(self, query, n_results=3):

        total = self.collection.count()

        if total == 0:
            print("Vector database is empty.")
            return []

        results = self.collection.query(
            query_texts=[query],
            n_results=min(n_results, total)
        )

        return results

    def show_results(self, results):

        if not results or not results.get("documents"):
            print("No results found.")
            return

        for index, document in enumerate(results["documents"][0], 1):

            metadata = results["metadatas"][0][index - 1]
            distance = results["distances"][0][index - 1]

            print(f"\nResult {index}")
            print("Source:", metadata["source"])
            print("Chunk:", metadata["chunk_index"])
            print("Distance:", round(distance, 4))
            print("Content:", document[:300])


if __name__ == "__main__":

    documents = load_documents()

    store = VectorStore()

    store.add_documents(documents)

    query = "What is Python and where is it used?"

    print("\nSearch Query:", query)

    results = store.search(query)

    store.show_results(results)
