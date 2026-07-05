FROM python:3.11-slim

WORKDIR /app

# Copy application files
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

# Start Ollama temporarily, download the model, then stop the server
RUN ollama serve & \
    OLLAMA_PID=$! && \
    sleep 10 && \
    ollama pull llama3.2 && \
    kill $OLLAMA_PID

# Streamlit port
EXPOSE 8501

# Start Ollama and Streamlit
CMD sh -c "ollama serve & sleep 5 && streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0"