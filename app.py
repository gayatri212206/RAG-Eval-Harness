
import json
from pathlib import Path


import streamlit as st

from src.ingestion import load_documents
from src.vector_store import VectorStore

from src.retrieval.simple import SimpleRetriever
from src.retrieval.semantic import SemanticRetriever
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranking import RerankingRetriever


# --------------------------------
# PAGE CONFIGURATION
# --------------------------------

st.set_page_config(
    page_title="RAG Evaluation Dashboard",
    page_icon="🔍",
    layout="wide"
)

ROOT_DIR = Path(__file__).resolve().parent


# --------------------------------
# LOAD EVALUATION RESULTS
# --------------------------------

@st.cache_data
def load_evaluation_results():
    path = ROOT_DIR / "results" / "evaluation_summary.json"

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


# --------------------------------
# INITIALIZE RETRIEVERS
# --------------------------------

@st.cache_resource
def load_retrievers():

    # Initialize vector database
    vector_store = VectorStore()

    if vector_store.collection.count() == 0:
        documents = load_documents()
        vector_store.add_documents(documents)

    return {
        "Simple": SimpleRetriever(),
        "Semantic": SemanticRetriever(),
        "Hybrid": HybridRetriever(),
        "Reranking": RerankingRetriever()
    }


# --------------------------------
# DASHBOARD HEADER
# --------------------------------

st.title("🔍 RAG Evaluation Dashboard")

st.markdown(
    "Compare four retrieval strategies "
    "using a custom 50-question evaluation dataset."
)

st.divider()


# --------------------------------
# EVALUATION METRICS
# --------------------------------

summary = load_evaluation_results()

method_names = [
    "Simple",
    "Semantic",
    "Hybrid",
    "Reranking"
]

scores = {
    name: summary[name]["average_coverage"]
    for name in method_names
}

average_score = sum(scores.values()) / len(scores)

col1, col2, col3 = st.columns(3)

col1.metric("Evaluation Questions", "50")
col2.metric("Retrieval Methods", "4")
col3.metric("Average Coverage", f"{average_score:.2f}%")


st.divider()


# --------------------------------
# PERFORMANCE COMPARISON
# --------------------------------

st.subheader("📊 Retrieval Performance Comparison")

st.bar_chart(
    scores,
    x_label="Retrieval Method",
    y_label="Coverage (%)"
)

best_method = max(scores, key=scores.get)

st.success(
    f"Best performing method: {best_method} "
    f"({scores[best_method]:.2f}% coverage)"
)


# --------------------------------
# INTERACTIVE RETRIEVAL TESTING
# --------------------------------

st.divider()

st.subheader("🧪 Test Your Own Query")

query = st.text_input(
    "Enter your question:",
    placeholder="Example: What is Python and where is it used?"
)

selected_method = st.selectbox(
    "Select Retrieval Method",
    method_names
)

top_k = st.slider(
    "Number of results",
    min_value=1,
    max_value=7,
    value=3
)

if st.button("Search Documents", type="primary"):

    if not query.strip():
        st.warning("Please enter a question.")

    else:
        with st.spinner("Initializing retrieval and searching..."):

            try:
                retrievers = load_retrievers()

                retriever = retrievers[selected_method]

                results = retriever.retrieve(
                    query,
                    top_k=top_k
                )

                st.subheader("Retrieved Results")

                if not results:
                    st.info("No matching results found.")

                else:
                    for index, result in enumerate(results, start=1):

                        with st.expander(
                            f"Result {index}",
                            expanded=True
                        ):

                            st.write(result["content"])

                            if "source" in result:
                                st.caption(
                                    f"Source: {result['source']}"
                                )

                            if "distance" in result:
                                st.caption(
                                    f"Vector Distance: "
                                    f"{result['distance']:.4f}"
                                )

                            if "hybrid_score" in result:
                                st.caption(
                                    f"Hybrid Score: "
                                    f"{result['hybrid_score']:.5f}"
                                )

                            if "rerank_score" in result:
                                st.caption(
                                    f"Rerank Score: "
                                    f"{result['rerank_score']:.4f}"
                                )

            except Exception as error:
                st.error(f"An error occurred: {error}")


# --------------------------------
# FOOTER
# --------------------------------

st.divider()

st.caption(
    "RAG Evaluation Harness | "
    "Python • ChromaDB • BM25 • Streamlit • GitHub Actions"
)
