# API Reference

## Authentication

### POST `/login`
Login with credentials.

**Request Body:**
```json
{
  "username": "student",
  "password": "student123",
  "role": "student"
}
```

**Response (200):**
```json
{
  "ok": true,
  "redirect": "/student"
}
```

**Response (401):**
```json
{
  "ok": false,
  "error": "Invalid credentials"
}
```

### GET `/logout`
Clear session and redirect to login page.

---

## Student Endpoints

### POST `/upload`
Upload an audio file for processing.

**Content-Type:** `multipart/form-data`  
**Auth Required:** Yes

**Form Data:**
- `audio` — Audio file (MP3, WAV, M4A, max 100MB)

**Response (200):**
```json
{
  "transcript": "Full transcription text...",
  "notes": "AI-generated study notes in markdown...",
  "questions": [
    {
      "question": "What is machine learning?",
      "options": ["Option A", "Option B", "Option C", "Option D"],
      "answer": "Option A",
      "topic": "ML Basics"
    }
  ],
  "concepts": {
    "keywords": [{"term": "machine", "count": 15, "score": 2.4, "emphasized": true}],
    "concepts": [{"term": "machine learning", "count": 8, "score": 1.3}],
    "definitions": ["Machine learning is a subset of artificial intelligence..."]
  },
  "tone": {
    "emphasized_segments": [
      {"text": "This is very important...", "start": 45.2, "end": 52.1, "emphasis_score": 78.5}
    ],
    "energy_profile": [{"time": 0.0, "energy": 45.2}],
    "avg_energy": 52.3,
    "duration": 300.5
  }
}
```

### POST `/upload-url`
Process audio from a YouTube or video URL.

**Request Body:**
```json
{
  "url": "https://www.youtube.com/watch?v=..."
}
```

**Response:** Same as `/upload`, plus `source_title` field.

### POST `/api/submit-quiz`
Submit quiz answers.

**Request Body:**
```json
{
  "score": 4,
  "total": 5,
  "answers": [
    {"topic": "ML Basics", "correct": true, "picked": "Option A", "answer": "Option A"}
  ],
  "topics": ["ML Basics", "Neural Networks"]
}
```

---

## Teacher Endpoints

### GET `/api/analytics`
Get class performance analytics.

**Auth Required:** Teacher only

**Response (200):**
```json
{
  "students": [
    {
      "id": "s1",
      "name": "Yash Verma",
      "attempts": 3,
      "avg_score": 80,
      "strong_topics": ["ML Basics"],
      "weak_topics": ["Neural Networks"],
      "last_attempt": "2026-05-18"
    }
  ],
  "topics": [
    {"topic": "ML Basics", "correct": 8, "total": 10, "percentage": 80}
  ],
  "total_attempts": 5,
  "total_students": 3
}
```

---

## Session History

### GET `/api/sessions`
Get session history for the current user.

**Response:**
```json
{
  "sessions": [
    {
      "code": "1234",
      "teacher_name": "Dr. Kumar",
      "created_at": "2026-05-18T10:00:00",
      "ended_at": "2026-05-18T11:00:00",
      "duration": 3600,
      "student_count": 25,
      "chunk_count": 360,
      "has_notes": true,
      "has_quiz": true,
      "quiz_avg": 72.5
    }
  ]
}
```

### GET `/api/sessions/<code>/notes`
Get notes and transcript for a specific session.

---

## Live Lecture

### POST `/api/live/create`
Create a new live session (teacher only).

**Response:**
```json
{
  "ok": true,
  "code": "1234"
}
```

### POST `/api/live/join`
Join an existing live session.

**Request Body:**
```json
{
  "code": "1234"
}
```

---

## WebSocket Events

### Client → Server

| Event | Data | Description |
|-------|------|-------------|
| `teacher_join` | `{code}` | Teacher joins the session room |
| `student_join` | `{code, name}` | Student joins the session room |
| `audio_chunk` | `{code, audio}` | Base64-encoded audio chunk (10s) |
| `end_session` | `{code}` | Teacher ends the lecture |
| `request_notes` | `{code}` | Teacher requests notes regeneration |
| `send_quiz` | `{code}` | Teacher triggers quiz generation |
| `quiz_answer` | `{code, answers}` | Student submits quiz answers |

### Server → Client

| Event | Data | Description |
|-------|------|-------------|
| `transcript_update` | `{chunk_text, full_transcript, chunk_num}` | New transcription chunk |
| `concepts_update` | `{keywords, concepts, definitions}` | Updated key concepts |
| `emphasis_update` | `{emphasized_segments, energy_profile, avg_energy, duration}` | Updated emphasis data |
| `notes_update` | `{notes}` | Updated AI notes |
| `student_joined` | `{name, count}` | Student joined notification |
| `student_left` | `{name, count}` | Student left notification |
| `live_quiz` | `{questions}` | Quiz questions (no answers) |
| `quiz_result` | `{score, total, results}` | Individual quiz result |
| `quiz_scores` | `{responses}` | All quiz scores (teacher) |
| `session_ended` | `{message}` | Session ended notification |
| `session_state` | `{transcript, concepts, notes, tone}` | Full state for late-joining students |
