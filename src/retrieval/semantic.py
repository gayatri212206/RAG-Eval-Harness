
import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

from src.ingestion import load_documents
from src.chunking import semantic_chunking


class SemanticRetriever:
    def __init__(self):
        documents = load_documents()

        self.client = chromadb.Client()

        self.collection = self.client.create_collection(
            name="semantic_eval_chunks",
            embedding_function=DefaultEmbeddingFunction()
        )

        chunks = []
        metadatas = []
        ids = []

        for doc in documents:
            semantic_chunks = semantic_chunking(
                doc["content"],
                max_chunk_size=700
            )

            for index, chunk in enumerate(semantic_chunks):
                chunks.append(chunk)
                metadatas.append({
                    "source": doc["metadata"]["source"],
                    "chunk_index": index
                })
                ids.append(f"semantic_{len(ids)}")

        if chunks:
            self.collection.add(
                ids=ids,
                documents=chunks,
                metadatas=metadatas
            )

        print("Semantic collection initialized!")

    def retrieve(self, query, top_k=3):
        total = self.collection.count()

        if total == 0:
            return []

        results = self.collection.query(
            query_texts=[query],
            n_results=min(top_k, total)
        )

        return [
            {
                "content": results["documents"][0][i],
                "source": results["metadatas"][0][i]["source"],
                "distance": results["distances"][0][i]
            }
            for i in range(len(results["documents"][0]))
        ]


if __name__ == "__main__":
    retriever = SemanticRetriever()

    query = "What is Python and where is it used?"

    for i, result in enumerate(retriever.retrieve(query), 1):
        print(f"\nResult {i}")
        print("Source:", result["source"])
        print("Distance:", round(result["distance"], 4))
        print("Content:", result["content"][:250])
