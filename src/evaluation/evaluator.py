
import csv
import json
import re
from pathlib import Path

from src.retrieval.simple import SimpleRetriever
from src.retrieval.semantic import SemanticRetriever
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranking import RerankingRetriever
from src.ingestion import load_documents
from src.vector_store import VectorStore


ROOT_DIR = Path(__file__).resolve().parents[2]
DATASET_PATH = ROOT_DIR / "data" / "evaluation_dataset.json"
RESULTS_DIR = ROOT_DIR / "results"

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were",
    "in", "on", "of", "to", "for", "and", "or",
    "does", "do", "what", "which", "who", "when",
    "where", "how", "why", "can", "used", "use",
    "it", "this", "that", "with", "by", "as"
}


def tokenize(text):
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {
        word for word in words
        if word not in STOPWORDS and len(word) > 1
    }


def calculate_coverage(ground_truth, retrieved_context):
    expected_words = tokenize(ground_truth)

    if not expected_words:
        return 0.0

    retrieved_words = tokenize(retrieved_context)

    matched_words = expected_words.intersection(retrieved_words)

    return len(matched_words) / len(expected_words)


def main():
    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        dataset = json.load(file)

    print("Loading retrieval methods...")
        # Initialize vector database before evaluation
    vector_store = VectorStore()

    if vector_store.collection.count() == 0:
        print("Vector database is empty. Loading documents...")
        documents = load_documents()
        vector_store.add_documents(documents)
        
    retrievers = {
        "Simple": SimpleRetriever(),
        "Semantic": SemanticRetriever(),
        "Hybrid": HybridRetriever(),
        "Reranking": RerankingRetriever()
    }

    all_results = []
    summary = {}

    print("\nStarting Evaluation...")
    print("Total questions:", len(dataset))
    print("Retrieval methods:", len(retrievers))
    print("Total tests:", len(dataset) * len(retrievers))

    for method_name, retriever in retrievers.items():

        print(f"\nEvaluating {method_name}...")

        method_scores = []

        for item in dataset:
            question = item["question"]
            ground_truth = item["ground_truth"]

            retrieved_results = retriever.retrieve(
                question,
                top_k=3
            )

            retrieved_context = "\n".join(
                result["content"]
                for result in retrieved_results
            )

            coverage = calculate_coverage(
                ground_truth,
                retrieved_context
            )

            score = round(coverage * 100, 2)
            method_scores.append(score)

            all_results.append({
                "question_id": item["id"],
                "question": question,
                "method": method_name,
                "coverage_percentage": score,
                "retrieved_chunks": len(retrieved_results),
                "ground_truth": ground_truth,
                "retrieved_context": retrieved_context
            })

        average_score = sum(method_scores) / len(method_scores)

        summary[method_name] = {
            "average_coverage": round(average_score, 2),
            "questions_evaluated": len(method_scores)
        }

        print(f"Average Coverage: {average_score:.2f}%")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save detailed JSON report
    with open(
        RESULTS_DIR / "evaluation_results.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(all_results, file, indent=4, ensure_ascii=False)

    # Save CSV report
    with open(
        RESULTS_DIR / "evaluation_results.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=all_results[0].keys()
        )
        writer.writeheader()
        writer.writerows(all_results)

    # Save summary report
    with open(
        RESULTS_DIR / "evaluation_summary.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=4)

    print("\n========== EVALUATION SUMMARY ==========")

    for method, result in summary.items():
        print(
            f"{method}: "
            f"{result['average_coverage']}% "
            f"({result['questions_evaluated']} questions)"
        )

    print("\nTotal evaluation tests:", len(all_results))
    print("Reports saved in:", RESULTS_DIR)


if __name__ == "__main__":
    main()
