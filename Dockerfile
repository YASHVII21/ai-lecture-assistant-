# ─────────────────────────────────────────────
# AI Lecture Assistant — Hugging Face Spaces
# ─────────────────────────────────────────────

# Use slim Python 3.10 base
FROM python:3.10-slim

# System dependencies: ffmpeg for audio processing, git for whisper
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    build-essential \
    libsndfile1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy and install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the Whisper "small" model so first request isn't slow
# (downloads ~500MB into ~/.cache/whisper)
RUN python -c "import whisper; whisper.load_model('small'); print('Whisper model ready.')"

# Copy the rest of the app
COPY . .

# Create required runtime directories
RUN mkdir -p uploads data

# HF Spaces requires port 7860
ENV PORT=7860
ENV HOST=0.0.0.0
ENV FLASK_DEBUG=false

# Expose the port
EXPOSE 7860

# Launch the app
CMD ["python", "main.py"]
