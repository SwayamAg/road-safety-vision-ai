# ==============================================================================
# Road Safety Vision-AI: Production Multi-Stage Dockerfile
# Base: Debian Python 3.10-slim (Lightweight, Cloud-Optimized)
# ==============================================================================
FROM python:3.10-slim

# Prevent Python from writing .pyc files and buffer stdout/stderr for real-time logs
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    MODEL_DIR="models"

# Install system dependencies for headless OpenCV, video decoding (FFmpeg), and OpenMP
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code, configuration files, and pre-trained models
COPY . .

# Ensure standard runtime directories exist
RUN mkdir -p DATA/IP DATA/OP models

# Expose default HTTP port
EXPOSE 8000

# Healthcheck configuration
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:' + str(os.environ.get('PORT', 8000)) + '/health')" || exit 1

# Production Uvicorn startup
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
