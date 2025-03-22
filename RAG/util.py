import json
import os
import joblib
import numpy as np
import yaml
import classla
from nltk.corpus import stopwords
from scipy.sparse import save_npz, load_npz
from sklearn.feature_extraction.text import TfidfVectorizer

# Download Slovene model if not available
classla.download('sl')
nlp = classla.Pipeline('sl', processors='tokenize,pos,lemma')
stop_words = set(stopwords.words('slovene'))

# File paths for storing embeddings
BASE_PATH = "tfidf_embeddings"
EMBEDDINGS_FILE = f"{BASE_PATH}/embeddings.npz"
VECTORIZER_FILE = f"{BASE_PATH}/vectorizer.pkl"
METADATA_FILE = f"{BASE_PATH}/metadata.json"


class EmbeddingManager:
    """Manages storage and retrieval of TF-IDF embeddings."""

    _instance = None

    @staticmethod
    def get_instance():
        """Returns the singleton instance."""
        if EmbeddingManager._instance is None:
            EmbeddingManager._instance = EmbeddingManager()
        return EmbeddingManager._instance

    def __init__(self):
        self._vectorizer = None
        self._tfidf_matrix = None
        self._metadata = None

    def get_vectorizer(self):
        """Loads vectorizer from disk or cache."""
        if self._vectorizer is None:
            self._vectorizer = joblib.load(VECTORIZER_FILE)
        return self._vectorizer

    def get_tfidf_matrix(self):
        """Loads TF-IDF matrix from disk or cache."""
        if self._tfidf_matrix is None:
            self._tfidf_matrix = load_npz(EMBEDDINGS_FILE)
        return self._tfidf_matrix

    def get_metadata(self):
        """Loads metadata from disk or cache."""
        if self._metadata is None:
            with open(METADATA_FILE, "r") as f:
                self._metadata = np.array(json.load(f))
        return self._metadata

    def load_embeddings(self):
        """Ensures embeddings are stored and loads them."""
        if not os.path.exists(EMBEDDINGS_FILE):
            print(f"Missing embeddings. Generating now...")
            prepare_data()
        self.get_vectorizer()
        self.get_tfidf_matrix()
        self.get_metadata()

    @staticmethod
    def save_data(vectorizer, tfidf_matrix, metadata):
        """Saves vectorizer, TF-IDF matrix, and metadata."""
        os.makedirs(os.path.dirname(EMBEDDINGS_FILE), exist_ok=True)

        save_npz(EMBEDDINGS_FILE, tfidf_matrix)
        joblib.dump(vectorizer, VECTORIZER_FILE)

        with open(METADATA_FILE, "w") as f:
            json.dump(metadata, f)


def preprocess(text):
    """Preprocesses text using lemmatization and stopword removal."""
    doc = nlp(text)
    return ' '.join(
        token.words[0].lemma.lower()
        for sentence in doc.sentences
        for token in sentence.tokens
        if (token.words[0].lemma.isalpha() or token.words[0].lemma.isdigit())
        and token.words[0].lemma.lower() not in stop_words
    )


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
    with open("ai_act.yaml", "r") as file:
        data = yaml.safe_load(file)

    datasets = {key: process_section(data, key) for key in ["cleni", "tocke"]}

    all_texts, all_metadata = [], []
    for key, values in datasets.items():
        all_texts += values["preprocessed_embeddings"]
        all_metadata += values["metadata"]

    # Train a joint model for both 'cleni' and 'tocke'
    vectorizer = TfidfVectorizer(norm="l2")
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    EmbeddingManager().get_instance().save_data(vectorizer, tfidf_matrix, all_metadata)
