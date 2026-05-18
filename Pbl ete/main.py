import whisper
import os
import json
import re
import time
import uuid
import numpy as np
from collections import Counter
from datetime import datetime
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from werkzeug.utils import secure_filename
from google import genai
import librosa
import yt_dlp

# FFmpeg
import imageio_ffmpeg
ffmpeg_path = os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe())
os.environ["PATH"] += os.pathsep + ffmpeg_path

# ── Gemini ──
GEMINI_API_KEY = "AIzaSyCNRWmPjW_OiIPym0l5U14JcvcRZxGraCg"
GEMINI_MODEL = "gemini-2.5-flash"
client = genai.Client(api_key=GEMINI_API_KEY)

# ── Flask ──
app = Flask(__name__)
app.secret_key = "lecture-assistant-secret-key-2026"
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024
os.makedirs('uploads', exist_ok=True)
os.makedirs('data', exist_ok=True)

# ── Data paths ──
USERS_FILE = os.path.join('data', 'users.json')
RESULTS_FILE = os.path.join('data', 'results.json')

def load_json(path, default=None):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default if default is not None else []

def save_json(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# ── Whisper ──
print("Loading Whisper model...")
whisper_model = whisper.load_model("tiny")
print("Ready.")

# ── Stopwords ──
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


# ── Transcribe ──
def transcribe_audio(filepath):
    result = whisper_model.transcribe(filepath)
    segments = []
    for seg in result.get("segments", []):
        segments.append({
            "text": seg["text"].strip(),
            "start": seg["start"],
            "end": seg["end"],
        })
    return result["text"], segments


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
        if count < 3 or word in proper_nouns or word in NOISE_WORDS:
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
                for b, c in bigram_freq.most_common(15) if c >= 2]

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
    if raw.startswith("```"):
        lines = raw.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        raw = "\n".join(lines)
    try:
        questions = json.loads(raw)
        valid = []
        for q in questions:
            if (isinstance(q, dict) and "question" in q and "options" in q
                    and "answer" in q and len(q["options"]) == 4 and q["answer"] in q["options"]):
                if "topic" not in q:
                    q["topic"] = "General"
                valid.append(q)
        return valid[:5]
    except json.JSONDecodeError:
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

    time.sleep(10)

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
        fallback_topics = [k['term'] for k in concepts.get('keywords', [])][:5]
        for i, term in enumerate(fallback_topics):
            questions.append({
                "question": f"Which of the following best describes the significance of '{term.title()}' in the context of this lecture?",
                "options": [
                    f"It was a primary focus of the discussion.",
                    f"It was mentioned briefly but not explained.",
                    f"It is unrelated to the main topic.",
                    f"It is a completely incorrect concept."
                ],
                "answer": f"It was a primary focus of the discussion.",
                "topic": term.title()
            })
        if not questions:
             # Just in case concept extraction also missed
             questions = [{
                "question": "What was the main topic discussed?",
                "options": ["Option A", "Option B", "Option C", "Option D"],
                "answer": "Option A", "topic": "General"
             }]

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
            print(f"  Downloaded: {title} ({duration}s)", flush=True)

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


if __name__ == '__main__':
    app.run(debug=True, port=5000, use_reloader=False)