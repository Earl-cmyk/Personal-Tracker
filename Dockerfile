FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Copy application files
COPY . .

# Create necessary directories
RUN mkdir -p /app/data /app/static/icons

# Initialize database
RUN python -c "from app import init_db; init_db()"

# Expose port
EXPOSE 5000

# Start services
CMD sh -c "ollama serve & sleep 5 && python app.py"