FROM python:3.11-slim

# Avoid interactive prompts and python buffering
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080 \
    HOST=0.0.0.0

WORKDIR /app

# Install system dependencies required by OpenCV and MediaPipe Tasks
RUN apt-get update && apt-get install -y --no-install-recommends \
    libegl1 \
    libgles2 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY backend/requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy backend application and models
COPY backend/ backend/
COPY models/ models/

# Expose standard container port
EXPOSE 8080

# Production start command
CMD ["sh", "-c", "exec uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
