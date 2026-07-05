FROM python:3.11-slim

WORKDIR /app

COPY . /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    zstd \
    tesseract-ocr \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download model
RUN ollama serve & \
    OLLAMA_PID=$! && \
    sleep 10 && \
    ollama pull llama3.2 && \
    kill $OLLAMA_PID

EXPOSE 8501

CMD sh -c "ollama serve & sleep 5 && streamlit run chatapp.py --server.port=${PORT:-8501} --server.address=0.0.0.0"