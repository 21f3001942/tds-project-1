FROM python:3.12-slim
RUN pip install uv
RUN pip install fastapi
RUN pip install requests
RUN apt-get update && apt-get install -y git
RUN pip install pandas
RUN pip install openai
RUN pip install GitPython
RUN pip install mysql.connector
RUN apt-get update && apt-get install -y libpq-dev gcc
RUN pip install psycopg2
RUN pip install datetime
RUN pip install python-dateutil
RUN pip install Pillow
RUN pip install scikit-learn
RUN pip install aiohttp
RUN pip install aiofiles
RUN pip install duckdb
RUN pip install beautifulsoup4
RUN pip install SpeechRecognition
RUN pip install markdown
RUN pip install GitPython
RUN pip install pytesseract
RUN pip install uvicorn

# Install Node.js and npm (use your preferred version of Node.js)
RUN apt-get update && \
    apt-get install -y curl && \
    curl -sL https://deb.nodesource.com/setup_16.x | bash - && \
    apt-get install -y nodejs

# Optional: Verify Node.js and npm installation
RUN node -v && npm -v

# Set the working directory to /app
WORKDIR /app

# Copy your Python application files
COPY . .
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
