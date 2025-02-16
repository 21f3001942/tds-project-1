FROM python:3.9-slim
WORKDIR /

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    nodejs \
    npm \
    tesseract-ocr \
    libpq-dev \
    curl\
    gcc \
    && rm -rf /var/lib/apt/lists/*


# Install Python dependencies directly
RUN pip install --no-cache-dir \
    requests\
    fastapi==0.68.1 \
    uvicorn==0.15.0 \
    python-multipart==0.0.5 \
    aiofiles==0.7.0 \
    aiohttp==3.8.1 \
    beautifulsoup4==4.9.3 \
    duckdb==0.3.1 \
    GitPython==3.1.24 \
    mysql-connector-python==8.0.26 \
    numpy==1.21.2 \
    openai==1.3.0 \
    pandas==1.3.3 \
    Pillow==8.3.2 \
    psycopg2-binary==2.9.1 \
    python-dateutil==2.8.2 \
    pytesseract==0.3.8 \
    scikit-learn==0.24.2 \
    SpeechRecognition==3.8.1 \
    markdown==3.3.4 \
    httpx==0.24.1

# Copy application code
COPY . .

# Create data directory
RUN mkdir -p /data

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "main.py"]