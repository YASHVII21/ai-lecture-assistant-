# System Architecture

## Overview

The AI Lecture Assistant is a full-stack web application built on Flask with real-time capabilities via Socket.IO. It processes lecture audio through a multi-stage AI pipeline combining local ML models (Whisper, Librosa) with cloud AI services (Google Gemini).

## High-Level Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        CLIENT (Browser)                       │
│                                                               │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────┐ │
│  │  Login Page  │  │  Student UI  │  │   Teacher UI        │ │
│  │  (login.html)│  │(student.html)│  │  (teacher.html)     │ │
│  └─────────────┘  └──────────────┘  └─────────────────────┘ │
│                    ┌──────────────┐                           │
│                    │  Live Mode   │ ← WebSocket (Socket.IO)  │
│                    │ (live.html)  │                           │
│                    └──────────────┘                           │
└──────────────────────────┬───────────────────────────────────┘
                           │ HTTP + WebSocket
                           ▼
┌──────────────────────────────────────────────────────────────┐
│                     SERVER (Flask + Eventlet)                  │
│                                                               │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                    Route Handlers                        │ │
│  │  /login  /upload  /upload-url  /api/*  /live            │ │
│  └──────────────────────┬──────────────────────────────────┘ │
│                         │                                     │
│  ┌──────────────────────▼──────────────────────────────────┐ │
│  │              AI Processing Pipeline                      │ │
│  │                                                          │ │
│  │  ┌──────────┐  ┌───────────┐  ┌──────────────────────┐ │ │
│  │  │ Whisper  │  │  Librosa  │  │  NLP Engine          │ │ │
│  │  │ (STT)   │  │  (Tone)   │  │  (Concepts/Bigrams)  │ │ │
│  │  └──────────┘  └───────────┘  └──────────────────────┘ │ │
│  │                                                          │ │
│  │  ┌──────────────────────────────────────────────────┐   │ │
│  │  │           Google Gemini API (Cloud)                │   │ │
│  │  │    Notes Generation  ←→  Quiz Generation          │   │ │
│  │  └──────────────────────────────────────────────────┘   │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                               │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                Socket.IO Event Handlers                  │ │
│  │  audio_chunk → teacher_join → student_join → send_quiz  │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                               │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                Data Layer (JSON Files)                    │ │
│  │  users.json    results.json    sessions.json             │ │
│  └─────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

## Processing Pipeline

### File Upload Pipeline
```
Audio File → [Whisper Transcribe] → [Librosa Tone Analysis] → [NLP Concept Extraction]
                                                                       │
                                                                       ▼
                                                              [Gemini Notes] → [Gemini Quiz]
                                                                       │
                                                                       ▼
                                                              JSON Response to Client
```

### Live Lecture Pipeline
```
Microphone → [10s chunks] → WebSocket → [WebM→WAV Convert] → [Whisper Transcribe]
                                                                       │
                                                              ┌────────┼────────┐
                                                              ▼        ▼        ▼
                                                          Transcript Concepts  Emphasis
                                                         (every chunk)       (every chunk)
                                                              │
                                                              ▼
                                                         [Gemini Notes]
                                                        (every 6 chunks / ~60s)
```

## Module Breakdown

| Module | Responsibility |
|--------|---------------|
| `transcribe_audio()` | Whisper STT with beam search + hallucination filtering |
| `analyze_audio_tone()` | Librosa RMS energy + emphasis scoring |
| `extract_key_concepts()` | Custom NLP: keyword freq, bigrams, definitions |
| `generate_notes()` | Gemini API → structured study notes |
| `generate_questions()` | Gemini API → 5 MCQ quiz questions |
| `_generate_fallback_quiz()` | Local fallback quiz (definitions, frequency, fill-blank) |
| `process_audio()` | Orchestrates the full pipeline |
| `_process_chunk_in_background()` | Live chunk processing in eventlet green thread |

## Data Storage

The application uses JSON file-based storage (no database required):

- **`users.json`** — User credentials (students + teachers)
- **`results.json`** — All quiz attempt results with per-question answers
- **`sessions.json`** — Live session history with transcripts, notes, and quiz results
- **`live_sessions`** — In-memory dict for active live sessions (not persisted during session)

## Security Model

- Session-based authentication via Flask sessions
- Role-based access control (student vs teacher)
- API keys stored in environment variables
- File uploads sandboxed to `uploads/` directory and auto-cleaned
- Input sanitization for quiz options (DOM-based rendering, no innerHTML for user data)
