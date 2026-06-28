import threading

import os
import re
import json
import uuid
import time
import random
import base64
import subprocess
import numpy as np
from collections import Counter
from datetime import datetime

from faster_whisper import WhisperModel
import librosa
import yt_dlp
import imageio_ffmpeg
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.utils import secure_filename
from flask_socketio import SocketIO, emit, join_room, leave_room
from google import genai

# load .env if available (optional)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# make sure ffmpeg is on PATH
ffmpeg_path = os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe())
os.environ['PATH'] += os.pathsep + ffmpeg_path

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', 'YOUR GEMINI API KEY')
GEMINI_MODEL   = os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')
client = genai.Client(api_key=GEMINI_API_KEY)

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'lect-asst-dev-key-2026')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

# Fix for HF Spaces / reverse proxy — lets Flask see the real HTTPS scheme
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

# Session cookies that work on HF Spaces (HTTPS + SameSite)
app.config['SESSION_COOKIE_SAMESITE'] = 'None'
app.config['SESSION_COOKIE_SECURE'] = True
app.config['SESSION_COOKIE_HTTPONLY'] = True

socketio = SocketIO(app, async_mode='threading', max_http_buffer_size=10 * 1024 * 1024,
                    cors_allowed_origins='*')

os.makedirs('uploads', exist_ok=True)
os.makedirs('data', exist_ok=True)

# Seed default users if file is missing (e.g. first run on cloud deployment)
_USERS_PATH = os.path.join('data', 'users.json')
if not os.path.exists(_USERS_PATH):
    _default_users = {
        "students": [
            {"id": "s1", "name": "Yash Verma",    "username": "student", "password": "student123"},
            {"id": "s2", "name": "Priya Sharma",  "username": "priya",   "password": "priya123"},
            {"id": "s3", "name": "Rahul Patel",   "username": "rahul",   "password": "rahul123"}
        ],
        "teachers": [
            {"id": "t1", "name": "Dr. Kumar", "username": "teacher", "password": "teacher123"}
        ]
    }
    with open(_USERS_PATH, 'w', encoding='utf-8') as _f:
        json.dump(_default_users, _f, indent=2)
    print("Seeded default users.json")

# active live sessions keyed by 4-digit code
live_sessions = {}

USERS_FILE   = os.path.join('data', 'users.json')
RESULTS_FILE = os.path.join('data', 'results.json')
SESSIONS_FILE = os.path.join('data', 'sessions.json')

def load_json(path, default=None):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default if default is not None else []

def save_json(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# Windows console sometimes can't handle emoji - just replace them
import sys
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ('utf-8', 'utf8'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

def safe_print(*args, **kwargs):
    try:
        print(*args, **kwargs)
    except (UnicodeEncodeError, UnicodeDecodeError):
        safe_args = [str(a).encode('ascii', 'replace').decode('ascii') for a in args]
        print(*safe_args, **kwargs)

WHISPER_MODEL_SIZE = os.getenv('WHISPER_MODEL', 'small')
print(f'Loading Whisper ({WHISPER_MODEL_SIZE})...')
whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device='cpu', compute_type='int8')
print('Ready.')

# common filler words to ignore in analysis
STOPWORDS = set("""
a an the is are was were be been being have has had do does did will would shall
should may might can could and but or nor for yet so at by from in into of on to
up with as it its he she they them their his her we our you your this that these
those what which who whom how when where why all each every some any no not very
much more most other another such than too also just about above after again
against between both during few here there then once only own same still through
under until while i me my myself let lets say said says well get got now even
know see look make like go going one two three four five six seven eight nine ten
dont really right thing things way back well thats called also used using uses
use see need first second third next last basically actually probably something
someone want wants give gives given take takes come comes yes okay sure today
tomorrow already always never nothing everything everyone anyone something
talk talks talking point points new old different same another try tries tell
tells told ask asks asked put puts become becomes becomes become start starts
people person guys lot lots kind kinds part parts means mean meant example
""".split())

NOISE_WORDS = set("""
song songs likes dislikes listen listening heard hear choice choices
guess guessing correct incorrect wrong answer answers question questions
paul john mary alice bob charlie teacher student class classroom
axis basis simple simplicity fast faster slow slower light heavy
called calls watch watching play playing showed show shows shown
gender male female voice voices name names type types group groups
""".split())


# transcription - beam search gives noticeably better results than greedy
def transcribe_audio(filepath):
    raw_segments, _ = whisper_model.transcribe(
        filepath,
        language='en',
        beam_size=5,
        temperature=0.0,
        condition_on_previous_text=True,
        no_speech_threshold=0.5,
        compression_ratio_threshold=2.4,
        log_prob_threshold=-1.0,
        vad_filter=True,
    )
    segments = []
    full_text_parts = []
    for seg in raw_segments:
        text = seg.text.strip()
        if not text or len(text) < 3:
            continue
        if re.match(r'^[.!?,;:\s]+$', text):
            continue
        segments.append({
            'text': text,
            'start': seg.start,
            'end': seg.end,
        })
        full_text_parts.append(text)
    return ' '.join(full_text_parts), segments


# ── Tone Analysis ──
def analyze_audio_tone(filepath, segments):
    try:
        y, sr = librosa.load(filepath, sr=16000)
        duration = len(y) / sr
        frame_length = int(0.05 * sr)
        hop_length = int(0.025 * sr)
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        rms_times = librosa.frames_to_time(range(len(rms)), sr=sr, hop_length=hop_length)

        mean_energy = float(np.mean(rms))
        std_energy = float(np.std(rms))

        emphasis_scores = []
        for seg in segments:
            start_t, end_t = seg["start"], seg["end"]
            mask = (rms_times >= start_t) & (rms_times <= end_t)
            seg_rms = rms[mask]
            if len(seg_rms) == 0:
                emphasis_scores.append(0)
                continue
            seg_mean = float(np.mean(seg_rms))
            word_count = len(seg["text"].split())
            seg_duration = max(end_t - start_t, 0.1)
            speaking_rate = word_count / seg_duration
            energy_score = max(0, (seg_mean - mean_energy) / max(std_energy, 0.001))
            rate_score = max(0, 1.5 - (speaking_rate / 3.0))
            combined = (energy_score * 0.6 + rate_score * 0.4) * 50
            emphasis_scores.append(round(min(max(combined, 0), 100), 1))

        score_threshold = np.percentile(emphasis_scores, 70) if emphasis_scores else 50

        emphasized_segments = []
        for i, seg in enumerate(segments):
            text = seg["text"].strip()
            if len(text.split()) < 6:
                continue
            if text.lower().strip('.!?') in ['correct', 'yes', 'okay', 'right', 'no', 'you']:
                continue
            if emphasis_scores[i] >= score_threshold:
                emphasized_segments.append({
                    "text": text,
                    "start": round(seg["start"], 1),
                    "end": round(seg["end"], 1),
                    "emphasis_score": emphasis_scores[i]
                })

        n_points = max(5, min(30, len(rms) // 10))
        chunk_size = max(1, len(rms) // n_points)
        max_rms = float(np.max(rms)) if len(rms) > 0 else 1.0

        energy_profile = []
        for i in range(0, len(rms), chunk_size):
            chunk = rms[i:i + chunk_size]
            t = float(rms_times[min(i, len(rms_times) - 1)])
            val = float(np.mean(chunk))
            energy_profile.append({
                "time": round(t, 1),
                "energy": round((val / max_rms) * 100, 2) if max_rms > 0 else 0
            })

        return {
            "emphasized_segments": sorted(emphasized_segments, key=lambda x: x["emphasis_score"], reverse=True)[:6],
            "energy_profile": energy_profile[:n_points],
            "avg_energy": round((mean_energy / max_rms) * 100, 2) if max_rms > 0 else 50,
            "duration": round(duration, 1)
        }
    except Exception as e:
        print(f"Tone analysis error: {e}")
        return {"emphasized_segments": [], "energy_profile": [], "avg_energy": 0, "duration": 0}


# ── NLP Concepts ──
def _detect_proper_nouns(transcript):
    proper = set()
    for match in re.finditer(r'[a-z]\s+([A-Z][a-z]{2,})', transcript):
        word = match.group(1).lower()
        if word not in STOPWORDS and len(word) >= 3:
            proper.add(word)
    return proper


def extract_key_concepts(transcript, tone_data=None):
    proper_nouns = _detect_proper_nouns(transcript)
    words = re.findall(r'[a-zA-Z]{3,}', transcript.lower())
    filtered = [w for w in words if w not in STOPWORDS and len(w) >= 4]
    total = max(len(filtered), 1)
    freq = Counter(filtered)

    keywords = []
    for word, count in freq.most_common(40):
        if count < 2 or word in proper_nouns or word in NOISE_WORDS:
            continue
        keywords.append({"term": word, "count": count, "score": round(count / total * 100, 2)})

    sentences = re.split(r'[.!?]', transcript)
    bigram_freq = Counter()
    for sent in sentences:
        ws = re.findall(r'[a-zA-Z]{3,}', sent.lower())
        clean = [w for w in ws if w not in STOPWORDS and w not in NOISE_WORDS and w not in proper_nouns and len(w) >= 4]
        for i in range(len(clean) - 1):
            bigram_freq[f"{clean[i]} {clean[i+1]}"] += 1

    concepts = [{"term": b, "count": c, "score": round(c / total * 100, 2)}
                for b, c in bigram_freq.most_common(15) if c >= 1]

    definitions = []
    for pat in [r"([^.]*(?:is called|is a|is an|refers to|means|defined as|known as)[^.]*\.)",
                r"([^.]*(?:is when|is where|is how)[^.]*\.)"]:
        for m in re.finditer(pat, transcript, re.IGNORECASE):
            d = m.group(1).strip()
            if 20 < len(d) < 250:
                definitions.append(d)

    emphasized_words = set()
    if tone_data and tone_data.get("emphasized_segments"):
        for seg in tone_data["emphasized_segments"]:
            for w in re.findall(r'[a-zA-Z]{4,}', seg["text"].lower()):
                if w not in STOPWORDS and w not in NOISE_WORDS and w not in proper_nouns:
                    emphasized_words.add(w)

    for kw in keywords:
        kw["emphasized"] = kw["term"] in emphasized_words

    return {"keywords": keywords[:12], "concepts": concepts[:8], "definitions": definitions[:8]}


# ── Gemini: Notes ──
def generate_notes(transcript):
    prompt = f"""You are an expert academic note-taker. Create clear, organized study notes.

Rules:
- Extract ALL key concepts, definitions, examples
- Group by topic with clear headings
- Use bullet points, concise but comprehensive
- Include formulas, names, specific details
- Make it useful for exam prep

Transcript:
{transcript}

Study Notes:"""
    response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    return response.text


# ── Gemini: Quiz ──
def generate_questions(transcript):
    prompt = f"""You are an expert educator. Generate exactly 5 multiple-choice quiz questions.

RULES:
- Test UNDERSTANDING, not just recall
- Cover DIFFERENT topics
- 4 options each, only ONE correct
- Wrong options must be plausible
- NO "All of the above"

Transcript:
{transcript}

Respond ONLY with valid JSON array (no markdown, no backticks):
[
  {{
    "question": "Your question?",
    "options": ["A", "B", "C", "D"],
    "answer": "The correct option text",
    "topic": "Brief topic label (2-3 words)"
  }}
]"""
    response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    raw = response.text.strip()

    # Strip markdown code fences in all forms: ```json, ```JSON, ``` etc.
    if '```' in raw:
        lines = raw.split('\n')
        lines = [l for l in lines if not l.strip().startswith('```')]
        raw = '\n'.join(lines).strip()

    # If Gemini added text before/after the JSON array, extract the array
    start = raw.find('[')
    end = raw.rfind(']')
    if start != -1 and end != -1 and end > start:
        raw = raw[start:end+1]

    try:
        questions = json.loads(raw)
        valid = []
        for q in questions:
            if (isinstance(q, dict) and 'question' in q and 'options' in q
                    and 'answer' in q and len(q['options']) == 4):
                # Accept question even if answer not in options (fix it)
                if q['answer'] not in q['options']:
                    # Try case-insensitive match
                    for opt in q['options']:
                        if opt.strip().lower() == q['answer'].strip().lower():
                            q['answer'] = opt
                            break
                    else:
                        q['answer'] = q['options'][0]  # fallback to first option
                if 'topic' not in q:
                    q['topic'] = 'General'
                valid.append(q)
        return valid[:5]
    except json.JSONDecodeError as e:
        print(f"  Quiz JSON parse error: {e} | raw[:200]: {raw[:200]}", flush=True)
        return []


# ══════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════

# ── Auth ──
@app.route('/')
def index():
    if 'user' in session:
        if session['role'] == 'teacher':
            return redirect(url_for('teacher_dashboard'))
        return redirect(url_for('student_dashboard'))
    return render_template('login.html')


@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username', '').strip()
    password = data.get('password', '').strip()
    role = data.get('role', 'student')

    users = load_json(USERS_FILE, {"students": [], "teachers": []})
    pool = users.get('students' if role == 'student' else 'teachers', [])

    for user in pool:
        if user['username'] == username and user['password'] == password:
            session['user'] = user['id']
            session['name'] = user['name']
            session['role'] = role
            return jsonify({"ok": True, "redirect": "/teacher" if role == "teacher" else "/student"})

    return jsonify({"ok": False, "error": "Invalid credentials"}), 401


@app.route('/logout')
def logout():
    session.clear()
    return redirect('/')


# ── Student ──
@app.route('/student')
def student_dashboard():
    if 'user' not in session or session.get('role') != 'student':
        return redirect('/')
    return render_template('student.html', name=session['name'])


# ── Teacher ──
@app.route('/teacher')
def teacher_dashboard():
    if 'user' not in session or session.get('role') != 'teacher':
        return redirect('/')
    return render_template('teacher.html', name=session['name'])


# ── Shared processing pipeline ──
def process_audio(filepath):
    """Run the full pipeline on an audio file. Returns dict with results."""
    print("  [1/5] Transcribing...", flush=True)
    transcript, segments = transcribe_audio(filepath)

    print("  [2/5] Analyzing tone...", flush=True)
    tone_data = analyze_audio_tone(filepath, segments)

    print("  [3/5] Extracting concepts...", flush=True)
    concepts = extract_key_concepts(transcript, tone_data)

    print("  [4/5] Generating notes...", flush=True)
    notes = None
    for attempt in range(3):
        try:
            notes = generate_notes(transcript)
            break
        except Exception as e:
            print(f"  Notes attempt {attempt+1} failed: {e}", flush=True)
            wait = [10, 30, 60][attempt] if attempt < 3 else 60
            print(f"  Waiting {wait}s before retry...", flush=True)
            time.sleep(wait)
    if notes is None:
        notes = "AI notes unavailable — Gemini API rate limit reached. Wait 1 minute and try again."

    time.sleep(2)  # Brief pause between API calls to avoid rate limits

    print("  [5/5] Generating quiz...", flush=True)
    questions = []
    for attempt in range(3):
        try:
            questions = generate_questions(transcript)
            if questions:
                break
        except Exception as e:
            print(f"  Quiz attempt {attempt+1} failed: {e}", flush=True)
            wait = [10, 30, 60][attempt] if attempt < 3 else 60
            print(f"  Waiting {wait}s before retry...", flush=True)
            time.sleep(wait)

    # Fallback to local quiz generation if Gemini rate limit is exhausted
    if not questions:
        print("  Using fallback quiz (API rate-limited)...", flush=True)
        questions = _generate_fallback_quiz(transcript, concepts)

    print("  Done!", flush=True)

    return {
        "transcript": transcript,
        "notes": notes,
        "questions": questions,
        "concepts": concepts,
        "tone": tone_data
    }


# ── Upload file ──
@app.route('/upload', methods=['POST'])
def upload():
    if 'user' not in session:
        return jsonify({"error": "Not logged in"}), 401

    if 'audio' not in request.files:
        return jsonify({"error": "No audio file"}), 400
    file = request.files['audio']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        result = process_audio(filepath)
        return jsonify(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


# ── Upload from YouTube / video URL ──
@app.route('/upload-url', methods=['POST'])
def upload_url():
    if 'user' not in session:
        return jsonify({"error": "Not logged in"}), 401

    data = request.get_json()
    url = (data.get('url') or '').strip()
    if not url:
        return jsonify({"error": "No URL provided"}), 400

    # Basic URL validation
    if not re.match(r'https?://', url):
        return jsonify({"error": "Invalid URL"}), 400

    # Download audio with yt-dlp
    out_id = uuid.uuid4().hex[:8]
    out_path = os.path.join(app.config['UPLOAD_FOLDER'], f'yt_{out_id}.%(ext)s')
    final_path = None

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': out_path,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '128',
        }],
        'noplaylist': True,
        'max_filesize': 100 * 1024 * 1024,  # 100MB limit
        'socket_timeout': 30,
        'quiet': True,
        'no_warnings': True,
    }

    try:
        print(f"  [0/5] Downloading audio from URL...", flush=True)
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            # Find the downloaded file
            final_path = os.path.join(
                app.config['UPLOAD_FOLDER'],
                f'yt_{out_id}.mp3'
            )
            # Sometimes yt-dlp uses slightly different naming
            if not os.path.exists(final_path):
                # Try to find any file with the out_id
                for f in os.listdir(app.config['UPLOAD_FOLDER']):
                    if f.startswith(f'yt_{out_id}'):
                        final_path = os.path.join(app.config['UPLOAD_FOLDER'], f)
                        break

            if not os.path.exists(final_path):
                return jsonify({"error": "Failed to download audio"}), 500

            title = info.get('title', 'Unknown')
            duration = info.get('duration', 0)
            # Strip emojis/non-ASCII from title for safe printing on Windows
            safe_title = title.encode('ascii', 'replace').decode('ascii')
            safe_print(f"  Downloaded: {safe_title} ({duration}s)", flush=True)

        result = process_audio(final_path)
        result['source_title'] = title
        return jsonify(result)

    except yt_dlp.utils.DownloadError as e:
        error_msg = str(e)
        if 'Private video' in error_msg:
            return jsonify({"error": "This video is private"}), 400
        elif 'Video unavailable' in error_msg:
            return jsonify({"error": "Video unavailable or removed"}), 400
        elif 'age' in error_msg.lower():
            return jsonify({"error": "Age-restricted video — cannot download"}), 400
        return jsonify({"error": f"Download failed: {error_msg[:120]}"}), 500
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        if final_path and os.path.exists(final_path):
            os.remove(final_path)



# ── Submit Quiz Result ──
@app.route('/api/submit-quiz', methods=['POST'])
def submit_quiz():
    if 'user' not in session:
        return jsonify({"error": "Not logged in"}), 401

    data = request.get_json()
    results = load_json(RESULTS_FILE, [])

    entry = {
        "student_id": session['user'],
        "student_name": session['name'],
        "score": data.get('score', 0),
        "total": data.get('total', 5),
        "answers": data.get('answers', []),
        "topics": data.get('topics', []),
        "timestamp": datetime.now().isoformat()
    }
    results.append(entry)
    save_json(RESULTS_FILE, results)

    return jsonify({"ok": True})


# ── Teacher Analytics ──
@app.route('/api/analytics')
def analytics():
    if 'user' not in session or session.get('role') != 'teacher':
        return jsonify({"error": "Unauthorized"}), 403

    results = load_json(RESULTS_FILE, [])
    users = load_json(USERS_FILE, {"students": [], "teachers": []})

    # Per-student stats
    student_stats = {}
    topic_scores = {}

    for r in results:
        sid = r['student_id']
        if sid not in student_stats:
            student_stats[sid] = {
                "name": r['student_name'],
                "attempts": 0,
                "total_score": 0,
                "total_questions": 0,
                "topics_correct": {},
                "topics_total": {},
                "last_attempt": r['timestamp']
            }

        s = student_stats[sid]
        s['attempts'] += 1
        s['total_score'] += r['score']
        s['total_questions'] += r['total']
        s['last_attempt'] = max(s['last_attempt'], r['timestamp'])

        # Per-topic tracking
        for ans in r.get('answers', []):
            topic = ans.get('topic', 'General')
            correct = ans.get('correct', False)

            s['topics_total'][topic] = s['topics_total'].get(topic, 0) + 1
            if correct:
                s['topics_correct'][topic] = s['topics_correct'].get(topic, 0) + 1

            topic_scores.setdefault(topic, {"correct": 0, "total": 0})
            topic_scores[topic]["total"] += 1
            if correct:
                topic_scores[topic]["correct"] += 1

    # Build response
    students = []
    for sid, s in student_stats.items():
        avg = round((s['total_score'] / max(s['total_questions'], 1)) * 100)
        strong = []
        weak = []
        for topic, total in s['topics_total'].items():
            correct = s['topics_correct'].get(topic, 0)
            pct = round((correct / total) * 100)
            if pct >= 60:
                strong.append(topic)
            else:
                weak.append(topic)

        students.append({
            "id": sid, "name": s['name'], "attempts": s['attempts'],
            "avg_score": avg, "strong_topics": strong, "weak_topics": weak,
            "last_attempt": s['last_attempt'][:10]
        })

    # Class-wide topic performance
    topics = []
    for topic, data in sorted(topic_scores.items()):
        topics.append({
            "topic": topic,
            "correct": data["correct"],
            "total": data["total"],
            "percentage": round((data["correct"] / max(data["total"], 1)) * 100)
        })

    return jsonify({
        "students": students,
        "topics": topics,
        "total_attempts": len(results),
        "total_students": len(student_stats)
    })


# ══════════════════════════════════════════════
# SESSION HISTORY
# ══════════════════════════════════════════════

@app.route('/api/sessions')
def get_sessions():
    if 'user' not in session:
        return jsonify({"error": "Not logged in"}), 401
    sessions = load_json(SESSIONS_FILE, [])
    user_name = session.get('name', '')
    role = session.get('role', 'student')

    filtered = []
    for s in reversed(sessions):  # newest first
        entry = {
            "code": s.get("code"),
            "teacher_name": s.get("teacher_name"),
            "created_at": s.get("created_at"),
            "ended_at": s.get("ended_at"),
            "duration": s.get("duration", 0),
            "student_count": s.get("student_count", 0),
            "chunk_count": s.get("chunk_count", 0),
            "has_notes": bool(s.get("notes")),
            "has_quiz": len(s.get("quiz_results", [])) > 0,
            "quiz_avg": 0
        }
        quiz_results = s.get("quiz_results", [])
        if quiz_results:
            entry["quiz_avg"] = round(sum(r["score"] for r in quiz_results) / len(quiz_results), 1)

        if role == 'teacher' and s.get("teacher_id") == session.get('user'):
            entry["quiz_results"] = quiz_results
            entry["students"] = s.get("students", [])
            filtered.append(entry)
        elif role == 'student' and user_name in s.get("students", []):
            # Find this student's quiz result
            for qr in quiz_results:
                if qr.get("name") == user_name:
                    entry["my_score"] = qr["score"]
                    entry["my_total"] = qr["total"]
                    break
            filtered.append(entry)

    return jsonify({"sessions": filtered[:20]})  # max 20


@app.route('/api/sessions/<code>/notes')
def get_session_notes(code):
    if 'user' not in session:
        return jsonify({"error": "Not logged in"}), 401
    sessions = load_json(SESSIONS_FILE, [])
    for s in sessions:
        if s.get("code") == code:
            return jsonify({
                "notes": s.get("notes", ""),
                "transcript": s.get("transcript", ""),
                "concepts": s.get("concepts", {})
            })
    return jsonify({"error": "Session not found"}), 404


# ══════════════════════════════════════════════
# LIVE LECTURE — Routes & SocketIO Events
# ══════════════════════════════════════════════

@app.route('/live')
def live_page():
    if 'user' not in session:
        return redirect('/')
    return render_template('live.html', name=session['name'], role=session['role'])


@app.route('/api/live/create', methods=['POST'])
def live_create():
    if 'user' not in session or session.get('role') != 'teacher':
        return jsonify({"error": "Unauthorized"}), 403
    # Generate unique 4-digit code
    for _ in range(100):
        code = str(random.randint(1000, 9999))
        if code not in live_sessions:
            break
    live_sessions[code] = {
        "teacher_id": session['user'],
        "teacher_name": session['name'],
        "teacher_sid": None,
        "transcript": "",
        "segments": [],
        "concepts": {"keywords": [], "concepts": [], "definitions": []},
        "notes": "",
        "tone": {"emphasized_segments": [], "energy_profile": [], "avg_energy": 0, "duration": 0},
        "students": {},
        "chunk_count": 0,
        "last_notes_chunk": 0,
        "created_at": datetime.now().isoformat(),
        "quiz_active": False,
        "quiz_questions": [],
        "quiz_responses": {}
    }
    return jsonify({"ok": True, "code": code})


@app.route('/api/live/join', methods=['POST'])
def live_join():
    if 'user' not in session:
        return jsonify({"error": "Not logged in"}), 401
    data = request.get_json()
    code = str(data.get('code', '')).strip()
    if code not in live_sessions:
        return jsonify({"error": "Session not found"}), 404
    return jsonify({"ok": True, "code": code, "teacher_name": live_sessions[code]["teacher_name"]})


def _convert_webm_to_wav(webm_path, wav_path):
    """Convert webm/opus audio to wav using ffmpeg."""
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_exe, '-y', '-i', webm_path,
        '-ar', '16000', '-ac', '1', '-f', 'wav', wav_path
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
        if result.returncode != 0:
            stderr_text = result.stderr.decode('utf-8', errors='replace')[-300:]
            print(f"  [FFMPEG] Conversion failed (code {result.returncode}): {stderr_text}", flush=True)
    except Exception as e:
        print(f"  [FFMPEG] Exception: {e}", flush=True)


def _process_chunk_in_background(code, chunk_path, sid):
    """Process a single audio chunk: transcribe, extract concepts, detect emphasis, optionally generate notes."""
    print(f"  [CHUNK] Starting processing for session {code}...", flush=True)
    sess = live_sessions.get(code)
    if not sess:
        print(f"  [CHUNK] Session {code} not found, aborting", flush=True)
        return

    wav_path = chunk_path.replace('.webm', '.wav')
    try:
        # 1. Convert webm → wav
        print(f"  [CHUNK] Converting webm -> wav...", flush=True)
        _convert_webm_to_wav(chunk_path, wav_path)

        if not os.path.exists(wav_path):
            print(f"  [CHUNK] WAV conversion failed - file not created", flush=True)
            return

        wav_size = os.path.getsize(wav_path)
        print(f"  [CHUNK] WAV created: {wav_size} bytes", flush=True)

        # 2. Whisper transcribe the chunk
        print(f"  [CHUNK] Running Whisper transcription...", flush=True)
        raw_segs, _ = whisper_model.transcribe(wav_path, language='en', condition_on_previous_text=True, vad_filter=True)
        chunk_text = ' '.join([s.text.strip() for s in raw_segs]).strip()

        if not chunk_text:
            print(f"  [CHUNK] Whisper returned empty text, skipping", flush=True)
            return

        safe_print(f"  [CHUNK] Transcribed: '{chunk_text[:80]}...'", flush=True)

        # Accumulate transcript
        offset = sess["tone"]["duration"]
        new_segments = []
        for seg in result.get("segments", []):
            new_segments.append({
                "text": seg["text"].strip(),
                "start": round(seg["start"] + offset, 1),
                "end": round(seg["end"] + offset, 1),
            })

        sess["transcript"] += (" " if sess["transcript"] else "") + chunk_text
        sess["segments"].extend(new_segments)
        sess["chunk_count"] += 1

        # 3. Emit transcript update
        socketio.emit('transcript_update', {
            "chunk_text": chunk_text,
            "full_transcript": sess["transcript"],
            "chunk_num": sess["chunk_count"]
        }, room=code)

        # 4. NLP concept extraction (local, fast — runs every chunk)
        concepts = extract_key_concepts(sess["transcript"])
        sess["concepts"] = concepts
        socketio.emit('concepts_update', concepts, room=code)

        # 5. Librosa tone analysis on this chunk
        try:
            y, sr = librosa.load(wav_path, sr=16000)
            chunk_duration = len(y) / sr
            sess["tone"]["duration"] = round(offset + chunk_duration, 1)

            frame_length = int(0.05 * sr)
            hop_length = int(0.025 * sr)
            rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

            if len(rms) > 0:
                mean_energy = float(np.mean(rms))
                max_rms = float(np.max(rms))
                avg_pct = round((mean_energy / max(max_rms, 0.001)) * 100, 2)

                # Add energy samples to profile
                rms_times = librosa.frames_to_time(range(len(rms)), sr=sr, hop_length=hop_length)
                n_new = max(2, min(8, len(rms) // 10))
                chunk_size = max(1, len(rms) // n_new)
                for i in range(0, len(rms), chunk_size):
                    chunk_rms = rms[i:i + chunk_size]
                    t = float(rms_times[min(i, len(rms_times) - 1)]) + offset
                    val = float(np.mean(chunk_rms))
                    sess["tone"]["energy_profile"].append({
                        "time": round(t, 1),
                        "energy": round((val / max(max_rms, 0.001)) * 100, 2)
                    })
                sess["tone"]["avg_energy"] = avg_pct

                # Check for emphasis in new segments
                std_energy = float(np.std(rms))
                for seg in new_segments:
                    if len(seg["text"].split()) < 6:
                        continue
                    seg_start = seg["start"] - offset
                    seg_end = seg["end"] - offset
                    mask = (rms_times >= seg_start) & (rms_times <= seg_end)
                    seg_rms = rms[mask]
                    if len(seg_rms) == 0:
                        continue
                    seg_mean = float(np.mean(seg_rms))
                    word_count = len(seg["text"].split())
                    seg_duration = max(seg_end - seg_start, 0.1)
                    speaking_rate = word_count / seg_duration
                    energy_score = max(0, (seg_mean - mean_energy) / max(std_energy, 0.001))
                    rate_score = max(0, 1.5 - (speaking_rate / 3.0))
                    combined = (energy_score * 0.6 + rate_score * 0.4) * 50
                    if combined >= 20:
                        sess["tone"]["emphasized_segments"].append({
                            "text": seg["text"],
                            "start": seg["start"],
                            "end": seg["end"],
                            "emphasis_score": round(min(max(combined, 0), 100), 1)
                        })

                socketio.emit('emphasis_update', {
                    "emphasized_segments": sorted(
                        sess["tone"]["emphasized_segments"],
                        key=lambda x: x["emphasis_score"], reverse=True
                    )[:8],
                    "energy_profile": sess["tone"]["energy_profile"][-50:],
                    "avg_energy": sess["tone"]["avg_energy"],
                    "duration": sess["tone"]["duration"]
                }, room=code)
        except Exception as e:
            safe_print(f"  Live tone analysis error: {e}", flush=True)

        # 6. Gemini notes -- every 6 chunks (~60s) if we have enough content
        chunks_since_notes = sess["chunk_count"] - sess["last_notes_chunk"]
        word_count = len(sess["transcript"].split())
        if chunks_since_notes >= 6 and word_count >= 50:
            try:
                notes = generate_notes(sess["transcript"])
                sess["notes"] = notes
                sess["last_notes_chunk"] = sess["chunk_count"]
                socketio.emit('notes_update', {"notes": notes}, room=code)
            except Exception as e:
                safe_print(f"  Live notes generation error: {e}", flush=True)

        print(f"  [CHUNK] Processing complete for chunk #{sess['chunk_count']}", flush=True)

    except Exception as e:
        safe_print(f"  Chunk processing error: {e}", flush=True)
    finally:
        # Cleanup temp files
        for p in [chunk_path, wav_path]:
            if os.path.exists(p):
                try:
                    os.remove(p)
                except:
                    pass


# ── SocketIO Events ──

@socketio.on('connect')
def on_connect():
    print(f"  Client connected: {request.sid}", flush=True)


@socketio.on('disconnect')
def on_disconnect():
    sid = request.sid
    # Remove student from any session they were in
    for code, sess in list(live_sessions.items()):
        if sid in sess["students"]:
            name = sess["students"].pop(sid)
            socketio.emit('student_left', {
                "name": name, "count": len(sess["students"])
            }, room=code)
    print(f"  Client disconnected: {sid}", flush=True)


@socketio.on('teacher_join')
def on_teacher_join(data):
    code = data.get('code', '')
    if code in live_sessions:
        live_sessions[code]["teacher_sid"] = request.sid
        join_room(code)
        emit('joined', {"role": "teacher", "code": code})


@socketio.on('student_join')
def on_student_join(data):
    code = data.get('code', '')
    name = data.get('name', 'Student')
    if code not in live_sessions:
        emit('join_error', {"error": "Session not found"})
        return
    join_room(code)
    live_sessions[code]["students"][request.sid] = name
    # Send current state to newly joined student
    sess = live_sessions[code]
    emit('session_state', {
        "transcript": sess["transcript"],
        "concepts": sess["concepts"],
        "notes": sess["notes"],
        "tone": {
            "emphasized_segments": sorted(
                sess["tone"]["emphasized_segments"],
                key=lambda x: x["emphasis_score"], reverse=True
            )[:8],
            "energy_profile": sess["tone"]["energy_profile"][-50:],
            "avg_energy": sess["tone"]["avg_energy"],
            "duration": sess["tone"]["duration"]
        }
    })
    # Notify everyone
    socketio.emit('student_joined', {
        "name": name, "count": len(sess["students"])
    }, room=code)


@socketio.on('audio_chunk')
def on_audio_chunk(data):
    code = data.get('code', '')
    audio_raw = data.get('audio', '')
    sess = live_sessions.get(code)
    if not sess or request.sid != sess.get('teacher_sid'):
        return

    # Decode base64 audio from client
    try:
        if isinstance(audio_raw, str):
            audio_blob = base64.b64decode(audio_raw)
        elif isinstance(audio_raw, (bytes, bytearray)):
            audio_blob = bytes(audio_raw)
        else:
            print(f"  Unexpected audio type: {type(audio_raw)}", flush=True)
            return
    except Exception as e:
        print(f"  Audio decode error: {e}", flush=True)
        return

    print(f"  Received audio chunk: {len(audio_blob)} bytes for session {code}", flush=True)

    if len(audio_blob) < 100:
        print(f"  Chunk too small ({len(audio_blob)} bytes), skipping", flush=True)
        return

    # Save the webm chunk to a temp file
    chunk_id = uuid.uuid4().hex[:8]
    chunk_path = os.path.join(app.config['UPLOAD_FOLDER'], f'live_{chunk_id}.webm')
    with open(chunk_path, 'wb') as f:
        f.write(audio_blob)

    print(f"  Saved chunk, spawning processing...", flush=True)

    # Process in background thread
    t = threading.Thread(target=_process_chunk_in_background, args=(code, chunk_path, request.sid), daemon=True)
    t.start()


@socketio.on('end_session')
def on_end_session(data):
    code = data.get('code', '')
    sess = live_sessions.get(code)
    if not sess or request.sid != sess.get('teacher_sid'):
        return
    # Save session history before cleaning up
    try:
        sessions = load_json(SESSIONS_FILE, [])
        student_names = list(sess['students'].values())
        quiz_results = []
        for sid, resp in sess.get('quiz_responses', {}).items():
            quiz_results.append({
                "name": resp.get("name", "Unknown"),
                "score": resp.get("score", 0),
                "total": resp.get("total", 0)
            })
        session_record = {
            "code": code,
            "teacher_id": sess["teacher_id"],
            "teacher_name": sess["teacher_name"],
            "created_at": sess["created_at"],
            "ended_at": datetime.now().isoformat(),
            "duration": sess["tone"]["duration"],
            "student_count": len(student_names),
            "students": student_names,
            "transcript": sess["transcript"],
            "notes": sess["notes"],
            "concepts": sess["concepts"],
            "chunk_count": sess["chunk_count"],
            "quiz_results": quiz_results
        }
        sessions.append(session_record)
        save_json(SESSIONS_FILE, sessions)
        print(f"  Session {code} saved to history", flush=True)
    except Exception as e:
        print(f"  Error saving session: {e}", flush=True)

    socketio.emit('session_ended', {"message": "The lecture has ended."}, room=code)
    live_sessions.pop(code, None)


@socketio.on('request_notes')
def on_request_notes(data):
    """Teacher can manually request notes regeneration."""
    code = data.get('code', '')
    sess = live_sessions.get(code)
    if not sess or request.sid != sess.get('teacher_sid'):
        return
    if len(sess["transcript"].split()) < 30:
        emit('notes_update', {"notes": "_Not enough content yet to generate notes. Keep lecturing!_"})
        return

    def gen_notes():
        try:
            notes = generate_notes(sess["transcript"])
            sess["notes"] = notes
            sess["last_notes_chunk"] = sess["chunk_count"]
            socketio.emit('notes_update', {"notes": notes}, room=code)
        except Exception as e:
            print(f"  Manual notes error: {e}", flush=True)
            socketio.emit('notes_update', {"notes": "_Notes generation failed (API limit). Try again in a minute._"}, room=code)

    threading.Thread(target=gen_notes, daemon=True).start()


@socketio.on('send_quiz')
def on_send_quiz(data):
    """Teacher triggers a live quiz from the current transcript."""
    code = data.get('code', '')
    sess = live_sessions.get(code)
    if not sess or request.sid != sess.get('teacher_sid'):
        return
    if len(sess['transcript'].split()) < 30:
        emit('quiz_status', {"error": "Not enough content yet for a quiz. Keep lecturing!"})
        return
    if sess.get('quiz_active'):
        emit('quiz_status', {"error": "A quiz is already active."})
        return

    def gen_quiz():
        try:
            print(f"  Generating live quiz for session {code}...", flush=True)
            questions = generate_questions(sess['transcript'])
            if not questions:
                socketio.emit('quiz_status', {"error": "Quiz generation failed. Try again."}, room=sess.get('teacher_sid'))
                return
            sess['quiz_questions'] = questions
            sess['quiz_active'] = True
            sess['quiz_responses'] = {}
            # Send questions WITHOUT answers to students
            safe_questions = []
            for q in questions:
                safe_questions.append({
                    "question": q["question"],
                    "options": q["options"],
                    "topic": q.get("topic", "General")
                })
            socketio.emit('live_quiz', {"questions": safe_questions}, room=code)
            print(f"  Quiz sent: {len(questions)} questions", flush=True)
        except Exception as e:
            print(f"  Quiz generation error: {e}", flush=True)
            socketio.emit('quiz_status', {"error": "Quiz generation failed."}, room=sess.get('teacher_sid'))

    threading.Thread(target=gen_quiz, daemon=True).start()


@socketio.on('quiz_answer')
def on_quiz_answer(data):
    """Student submits quiz answers."""
    code = data.get('code', '')
    answers = data.get('answers', [])
    sess = live_sessions.get(code)
    if not sess or not sess.get('quiz_active'):
        return

    sid = request.sid
    student_name = sess['students'].get(sid, 'Unknown')
    questions = sess['quiz_questions']

    # Score the answers
    score = 0
    total = len(questions)
    results = []
    for i, q in enumerate(questions):
        picked = answers[i] if i < len(answers) else ""
        correct = picked == q['answer']
        if correct:
            score += 1
        results.append({
            "correct": correct,
            "picked": picked,
            "answer": q['answer'],
            "topic": q.get('topic', 'General')
        })

    # Store response
    sess['quiz_responses'][sid] = {
        "name": student_name,
        "score": score,
        "total": total,
        "answers": results,
        "submitted_at": datetime.now().isoformat()
    }

    # Send result back to student
    emit('quiz_result', {
        "score": score,
        "total": total,
        "results": results
    })

    # Send updated scores to teacher
    score_list = []
    for s, resp in sess['quiz_responses'].items():
        score_list.append({
            "name": resp['name'],
            "score": resp['score'],
            "total": resp['total']
        })
    socketio.emit('quiz_scores', {"responses": score_list}, room=sess.get('teacher_sid'))

    # Save to results.json for dashboard analytics
    try:
        # Find student_id from name
        users = load_json(USERS_FILE, {"students": [], "teachers": []})
        student_id = sid
        for u in users.get('students', []):
            if u['name'] == student_name:
                student_id = u['id']
                break
        all_results = load_json(RESULTS_FILE, [])
        all_results.append({
            "student_id": student_id,
            "student_name": student_name,
            "score": score,
            "total": total,
            "answers": results,
            "topics": [r['topic'] for r in results],
            "source": "live",
            "session_code": code,
            "timestamp": datetime.now().isoformat()
        })
        save_json(RESULTS_FILE, all_results)
    except Exception as e:
        print(f"  Error saving quiz result: {e}", flush=True)


def _generate_fallback_quiz(transcript, concepts):
    """Generate a varied local quiz when the Gemini API is unavailable."""
    questions = []
    keywords = [k['term'] for k in concepts.get('keywords', [])]
    definitions = concepts.get('definitions', [])
    bigrams = [c['term'] for c in concepts.get('concepts', [])]

    # Extract meaningful sentences for fill-in-the-blank style
    sentences = [s.strip() for s in re.split(r'[.!?]', transcript) if len(s.strip().split()) >= 8]

    # Strategy 1: Definition-based questions
    for defn in definitions[:2]:
        # Find the key term being defined
        for pat in [r'(\w+(?:\s+\w+)?)\s+(?:is called|is a|is an|refers to|means|defined as|known as)',
                    r'(?:is called|is a|is an|refers to|means|defined as|known as)\s+(\w+(?:\s+\w+)?)']:
            m = re.search(pat, defn, re.IGNORECASE)
            if m:
                term = m.group(1).strip()
                distractors = [k.title() for k in keywords if k.lower() != term.lower()][:3]
                while len(distractors) < 3:
                    distractors.append(f"Unrelated concept {len(distractors)+1}")
                options = [term.title()] + distractors[:3]
                random.shuffle(options)
                questions.append({
                    "question": f"According to the lecture, which term is described as follows: '{defn[:120]}...'?",
                    "options": options,
                    "answer": term.title(),
                    "topic": term.title()
                })
                break

    # Strategy 2: Keyword frequency-based questions
    if len(keywords) >= 4:
        top_term = keywords[0]
        other_terms = keywords[1:4]
        options = [top_term.title()] + [t.title() for t in other_terms]
        random.shuffle(options)
        questions.append({
            "question": "Which of the following concepts was discussed MOST frequently in the lecture?",
            "options": options,
            "answer": top_term.title(),
            "topic": "Key Concepts"
        })

    # Strategy 3: True/False style (as MCQ)
    if len(keywords) >= 2:
        real_topic = keywords[0].title()
        fake_topics = ["Quantum Computing", "Marine Biology", "Renaissance Art", "Stock Trading"]
        fake = [f for f in fake_topics if f.lower() != real_topic.lower()][:3]
        options = [real_topic] + fake[:3]
        random.shuffle(options)
        questions.append({
            "question": "Which of the following topics was actually covered in this lecture?",
            "options": options,
            "answer": real_topic,
            "topic": "Lecture Content"
        })

    # Strategy 4: Sentence-completion style
    for sent in sentences[:3]:
        words = sent.split()
        key_words = [w for w in words if w.lower() not in STOPWORDS and len(w) >= 4]
        if key_words:
            target = random.choice(key_words)
            blanked = sent.replace(target, '______', 1)
            distractors = [k.title() for k in keywords if k.lower() != target.lower()][:3]
            while len(distractors) < 3:
                distractors.append(f"concept_{len(distractors)+1}")
            options = [target] + distractors[:3]
            random.shuffle(options)
            questions.append({
                "question": f"Fill in the blank: '{blanked[:150]}'",
                "options": options,
                "answer": target,
                "topic": "Comprehension"
            })
            if len(questions) >= 5:
                break

    # Ensure we always have at least 1 question
    if not questions:
        questions = [{
            "question": "What was the primary subject discussed in this lecture?",
            "options": [
                keywords[0].title() if keywords else "The main topic",
                "An unrelated subject",
                "No specific topic was covered",
                "The lecture was about history"
            ],
            "answer": keywords[0].title() if keywords else "The main topic",
            "topic": "General"
        }]

    return questions[:5]


if __name__ == '__main__':
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "true").lower() == "true"
    socketio.run(app, debug=debug, host=host, port=port, use_reloader=False)
