from typing import List

import numpy as np
from langchain_core.documents import Document
from sklearn.metrics.pairwise import cosine_similarity

from util import EmbeddingManager, preprocess


def search(query, top_n=None):
    # Get the singleton instance and load embeddings
    embedding_manager = EmbeddingManager.get_instance()
    embedding_manager.load_embeddings()

    vectorizer = embedding_manager.get_vectorizer()
    tfidf_matrix = embedding_manager.get_tfidf_matrix()
    metadata = embedding_manager.get_metadata()

    preprocessed_query = preprocess(query)

    # Convert the query to a TF-IDF vector
    query_vector = vectorizer.transform([preprocessed_query])

    similarity_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()

    if top_n is None:
        # Take all relevant results
        top_indices = np.argsort(similarity_scores)[::-1]
    else:
        # Take only the top n relevant results
        top_indices = np.argsort(similarity_scores)[-top_n:][::-1]

    return [
        {
            "id": metadata[i]["id"],
            "similarity_score": similarity_scores[i],
            "text": metadata[i].get("raw_text", "")  # Ensure text is included in metadata
        }
        for i in top_indices
    ]


def search_documents(query, top_n=None) -> List[Document]:
    """Method that returns results in suitable format for use in LangChain retrievers"""
    # Get the singleton instance and load embeddings
    embedding_manager = EmbeddingManager.get_instance()
    embedding_manager.load_embeddings()

    vectorizer = embedding_manager.get_vectorizer()
    tfidf_matrix = embedding_manager.get_tfidf_matrix()
    metadata = embedding_manager.get_metadata()

    preprocessed_query = preprocess(query)

    # Convert the query to a TF-IDF vector
    query_vector = vectorizer.transform([preprocessed_query])

    similarity_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()

    if top_n is None:
        # Take all relevant results
        top_indices = np.argsort(similarity_scores)[::-1]
    else:
        # Take only the top n relevant results
        top_indices = np.argsort(similarity_scores)[-top_n:][::-1]

    return [
        Document(
            page_content=metadata[i].get("raw_text", ""),
            metadata={"id": metadata[i]["id"], "type": metadata[i]["type"], "similarity_score": float(similarity_scores[i])},
        ) for i in top_indices
    ]
