
import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction


class SimpleRetriever:
    def __init__(self, db_path="./chroma_db"):
        self.client = chromadb.PersistentClient(path=db_path)

        self.collection = self.client.get_or_create_collection(
            name="rag_documents",
            embedding_function=DefaultEmbeddingFunction()
        )

    def retrieve(self, query, top_k=3):
        total = self.collection.count()

        if total == 0:
            return []

        results = self.collection.query(
            query_texts=[query],
            n_results=min(top_k, total)
        )

        retrieved = []

        for i, document in enumerate(results["documents"][0]):
            retrieved.append({
                "content": document,
                "source": results["metadatas"][0][i]["source"],
                "distance": results["distances"][0][i]
            })

        return retrieved


if __name__ == "__main__":
    retriever = SimpleRetriever()

    query = "What is Python and where is it used?"
    results = retriever.retrieve(query)

    print("\nSIMPLE RETRIEVAL")
    print("Query:", query)

    for i, result in enumerate(results, 1):
        print(f"\nResult {i}")
        print("Source:", result["source"])
        print("Distance:", round(result["distance"], 4))
        print("Content:", result["content"][:250])
