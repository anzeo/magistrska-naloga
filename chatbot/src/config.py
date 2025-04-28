from pathlib import Path

# Root directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Paths
DB_DIR = PROJECT_ROOT / "db"
DB_PATH = DB_DIR / "chatbot.sqlite"

DATA_DIR = PROJECT_ROOT / "data"
AI_ACT_YAML_PATH = DATA_DIR / "ai_act.yaml"

TFIDF_EMBEDDINGS_DIR = PROJECT_ROOT / "src" / "retriever" / "tfidf_embeddings"
EMBEDDINGS_PATH = TFIDF_EMBEDDINGS_DIR / "embeddings.npz"
METADATA_PATH = TFIDF_EMBEDDINGS_DIR / "metadata.json"
VECTORIZER_PATH = TFIDF_EMBEDDINGS_DIR / "vectorizer.pkl"
