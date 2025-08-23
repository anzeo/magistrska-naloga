# Use official Python 3.11.9 slim image
FROM python:3.11.9-slim

# Set working directory inside container
WORKDIR /app

# Install Python dependencies, Download NLTK stopwords and Download Classla Slovenian models
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt \
 && python -m nltk.downloader stopwords \
 && python -c "import classla; classla.download('sl')"

# Copy project files into container
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Run FastAPI with Uvicorn (production mode)
CMD ["uvicorn", "src.api.controller:app", "--host", "0.0.0.0", "--port", "8000"]
