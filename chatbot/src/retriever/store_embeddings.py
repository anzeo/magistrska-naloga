import yaml
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import AI_ACT_YAML_PATH
from src.retriever.util import EmbeddingManager, preprocess


def process_section(data, key):
    """Processes either 'cleni' or 'tocke' and returns preprocessed text + metadata."""
    metadata, preprocessed_texts = [], []
    for d in data[key]:
        if key == "cleni":
            text = (
                    d['poglavje']['naslov'] + "\n" +
                    (d['oddelek']['naslov'] + "\n" if d['oddelek'] else '') +
                    d['naslov'] + "\n" +
                    d['vsebina']
            )
        else:
            text = (d['vsebina'])

        preprocessed_text = preprocess(text)

        preprocessed_texts.append(preprocessed_text)
        metadata.append({"id": d['id_elementa'], "type": key, "raw_text": text})
    return {"metadata": metadata, "preprocessed_embeddings": preprocessed_texts}


def prepare_data():
    """Reads YAML file, preprocesses text, and stores TF-IDF embeddings."""
    with open(AI_ACT_YAML_PATH, "r") as file:
        data = yaml.safe_load(file)

    print("Storing embeddings...")

    datasets = {key: process_section(data, key) for key in ["cleni", "tocke"]}

    all_texts, all_metadata = [], []
    for key, values in datasets.items():
        all_texts += values["preprocessed_embeddings"]
        all_metadata += values["metadata"]

    # Train a joint model for both 'cleni' and 'tocke'
    vectorizer = TfidfVectorizer(norm="l2")
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    EmbeddingManager().get_instance().save_data(vectorizer, tfidf_matrix, all_metadata)

    print("Embeddings saved!")

if __name__ == "__main__":
    prepare_data()
