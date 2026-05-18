# Deployment Guide

## Local Development

### Prerequisites
- Python 3.9 or later
- ~2 GB RAM minimum (for Whisper "small" model)
- FFmpeg (auto-installed via `imageio-ffmpeg`)
- Google Gemini API key

### Steps

```bash
# Clone and setup
git clone https://github.com/yourusername/ai-lecture-assistant.git
cd ai-lecture-assistant

# Create virtual environment
python -m venv venv
# Windows: venv\Scripts\activate
# Linux/macOS: source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env with your API key

# Initialize data directory
mkdir -p data uploads
echo '{"students":[{"id":"s1","name":"Student","username":"student","password":"student123"}],"teachers":[{"id":"t1","name":"Teacher","username":"teacher","password":"teacher123"}]}' > data/users.json

# Run
python main.py
```

Visit **http://localhost:5000**

---

## Production Deployment (Render)

### Why Render?
- Free tier available
- Easy Python deployment
- Supports WebSocket

### Steps

1. **Push to GitHub** (ensure no secrets in code)

2. **Create a Render account** at [render.com](https://render.com)

3. **New Web Service** → Connect your GitHub repo

4. **Configuration:**
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `python main.py`
   - **Environment:** Python 3

5. **Environment Variables:**
   Set these in Render dashboard:
   ```
   GEMINI_API_KEY=your_key_here
   SECRET_KEY=a-strong-random-string
   HOST=0.0.0.0
   PORT=10000
   FLASK_DEBUG=false
   WHISPER_MODEL=tiny
   ```

> **⚠️ Important:** Free tier has limited RAM (~512MB). Use `WHISPER_MODEL=tiny` for free tier. For `small` model, you need at least 2GB RAM (paid tier).

---

## Production Deployment (Railway)

1. **Install Railway CLI:** `npm install -g @railway/cli`
2. **Login:** `railway login`
3. **Initialize:** `railway init`
4. **Deploy:** `railway up`
5. **Set env vars:** `railway variables set GEMINI_API_KEY=your_key`

---

## Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create data directories
RUN mkdir -p data uploads

EXPOSE 5000

CMD ["python", "main.py"]
```

```bash
# Build
docker build -t lecture-assistant .

# Run
docker run -p 5000:5000 \
  -e GEMINI_API_KEY=your_key \
  -e HOST=0.0.0.0 \
  -e FLASK_DEBUG=false \
  lecture-assistant
```

---

## Important Notes

### Memory Requirements
| Whisper Model | Minimum RAM |
|--------------|-------------|
| `tiny` | ~1 GB |
| `base` | ~1 GB |
| `small` | ~2 GB |
| `medium` | ~5 GB |
| `large` | ~10 GB |

### Rate Limits
- Google Gemini free tier: ~15 requests/minute
- The app handles rate limits with retry logic and fallback quiz generation

### File Storage
- Uploaded files are automatically deleted after processing
- Live session data is stored in `data/sessions.json`
- For production, consider using a database (PostgreSQL, MongoDB) for data persistence

### HTTPS
- For production, always use HTTPS
- Render and Railway provide free SSL certificates
- If self-hosting, use nginx as a reverse proxy with Let's Encrypt
