
from rank_bm25 import BM25Okapi
from src.ingestion import load_documents
from src.chunking import simple_chunking
from src.retrieval.simple import SimpleRetriever


class HybridRetriever:
    def __init__(self):
        self.vector_retriever = SimpleRetriever()

        documents = load_documents()
        self.chunks = []

        for doc in documents:
            chunks = simple_chunking(
                doc["content"],
                chunk_size=500,
                overlap=100
            )

            for chunk in chunks:
                self.chunks.append({
                    "content": chunk,
                    "source": doc["metadata"]["source"]
                })

        tokenized_chunks = [
            chunk["content"].lower().split()
            for chunk in self.chunks
        ]

        self.bm25 = BM25Okapi(tokenized_chunks)

    def retrieve(self, query, top_k=3):
        if not self.chunks:
            return []

        # Vector search
        vector_results = self.vector_retriever.retrieve(
            query, top_k=top_k
        )

        # BM25 keyword search
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)

        bm25_ranked = sorted(
            range(len(bm25_scores)),
            key=lambda i: bm25_scores[i],
            reverse=True
        )[:top_k]

        # Combine results using Reciprocal Rank Fusion
        combined_scores = {}

        for rank, result in enumerate(vector_results):
            key = result["content"]
            combined_scores[key] = combined_scores.get(key, 0) + (
                1 / (60 + rank + 1)
            )

        for rank, index in enumerate(bm25_ranked):
            key = self.chunks[index]["content"]
            combined_scores[key] = combined_scores.get(key, 0) + (
                1 / (60 + rank + 1)
            )

        ranked_chunks = sorted(
            combined_scores.items(),
            key=lambda item: item[1],
            reverse=True
        )[:top_k]

        return [
            {
                "content": content,
                "hybrid_score": round(score, 5)
            }
            for content, score in ranked_chunks
        ]


if __name__ == "__main__":
    retriever = HybridRetriever()

    query = "What is Python and where is it used?"
    results = retriever.retrieve(query)

    print("\nHYBRID SEARCH")
    print("Query:", query)

    for i, result in enumerate(results, 1):
        print(f"\nResult {i}")
        print("Hybrid Score:", result["hybrid_score"])
        print("Content:", result["content"][:250])
