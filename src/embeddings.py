
import numpy as np
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction


class EmbeddingEngine:

    def __init__(self):

        print("Loading ONNX embedding model...")

        self.model = DefaultEmbeddingFunction()

        print("Embedding model loaded successfully!")

    def generate_embeddings(self, texts):

        if not texts:
            return np.empty((0, 384), dtype=np.float32)

        embeddings = self.model(texts)

        return np.asarray(embeddings, dtype=np.float32)

    def generate_query_embedding(self, query):

        embedding = self.generate_embeddings([query])

        return embedding[0]


if __name__ == "__main__":

    engine = EmbeddingEngine()

    sample_texts = [
        "Python is a programming language.",
        "Artificial Intelligence enables machines to learn.",
        "Databases store and organize information."
    ]

    vectors = engine.generate_embeddings(sample_texts)

    print("\nEmbedding Results:")
    print("Number of vectors:", len(vectors))
    print("Vector dimensions:", vectors.shape)

    print("\nFirst vector preview:")
    print(vectors[0][:10])
