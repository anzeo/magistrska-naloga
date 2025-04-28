# Makefile
serve-terminal:
	python -m src.run_chatbot

store:
	python -m src.retriever.store_embeddings

serve-api:
	fastapi dev src/api/controller.py