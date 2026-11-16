
from rank_bm25 import BM25Okapi
from src.retrieval.simple import SimpleRetriever


class RerankingRetriever:
    def __init__(self):
        self.retriever = SimpleRetriever()

    def retrieve(self, query, top_k=3, candidate_k=10):

        # Step 1: Retrieve initial candidates using vector search
        candidates = self.retriever.retrieve(
            query,
            top_k=candidate_k
        )

        if not candidates:
            return []

        # Step 2: Prepare candidates for BM25 reranking
        tokenized_candidates = [
            item["content"].lower().split()
            for item in candidates
        ]

        bm25 = BM25Okapi(tokenized_candidates)

        # Step 3: Calculate keyword relevance scores
        tokenized_query = query.lower().split()
        scores = bm25.get_scores(tokenized_query)

        # Step 4: Sort candidates by BM25 score
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )

        # Step 5: Return reranked results
        reranked_results = []

        for index in ranked_indices[:top_k]:
            reranked_results.append({
                "content": candidates[index]["content"],
                "source": candidates[index]["source"],
                "original_distance": candidates[index]["distance"],
                "rerank_score": round(float(scores[index]), 4)
            })

        return reranked_results


if __name__ == "__main__":
    retriever = RerankingRetriever()

    query = "What is Python and where is it used?"
    results = retriever.retrieve(query)

    print("\nRERANKING RESULTS")
    print("Query:", query)

    for i, result in enumerate(results, 1):
        print(f"\nResult {i}")
        print("Source:", result["source"])
        print("Original Distance:", round(result["original_distance"], 4))
        print("Rerank Score:", result["rerank_score"])
        print("Content:", result["content"][:250])
