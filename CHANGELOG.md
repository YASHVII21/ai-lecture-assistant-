# Changelog

All notable changes to this project will be documented in this file.

## [2.0.0] — 2026-05-18

### Added
- Live Lecture Mode with real-time transcription via WebSocket
- Live quiz system (teacher sends, students answer in real-time)
- Session history for both teachers and students
- PDF export for AI-generated notes
- YouTube/video URL processing via yt-dlp
- Voice energy profiling and teacher emphasis detection
- Class analytics dashboard for teachers
- Environment variable configuration (.env support)
- Comprehensive GitHub documentation

### Changed
- Improved transcription accuracy (beam search, temperature fallback, hallucination filtering)
- Improved fallback quiz generation with varied question strategies
- Reduced processing delay from 10s to 2s between API calls
- Moved API keys to environment variables for security
- Pinned all dependency versions for reproducibility

### Fixed
- Teacher dashboard student table not showing data (empty template literals)
- Corrupted UTF-8 characters in teacher dashboard
- Topic tags showing empty text in performance table
- XSS vulnerability in live quiz option rendering
- Secret key now configurable via environment

## [1.0.0] — 2026-04-01

### Added
- Initial release
- Audio file upload and transcription
- AI notes generation via Gemini
- Quiz generation with MCQ
- Student and teacher roles
- Key concept extraction
- Basic analytics
