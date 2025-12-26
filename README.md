# 🤖 Ask-A — Immersive Voice AI Assistant

Ask-A is an AI-powered conversational assistant that enables users to interact using both text and voice.  
It features an immersive animated UI, intelligent AI responses, and follows an industry-standard architecture.

---

## 🌟 Key Features

- 🎙️ **Voice-based user instructions** (Speech-to-Text)
- 🔊 **AI voice responses** (Text-to-Speech)
- 🧠 **Context-aware** conversational memory
- 🌌 **Particle network** animated background
- 🧠 **“Thinking…” wave animation** during AI processing
- 🎧 **Audio-reactive** AI visuals
- 📱 **Mobile-friendly** smooth scrolling & animations
- 🔁 **Cloud AI** with local fallback support

---

## 🧠 System Architecture



1. **Browser**: Captures audio via Microphone.
2. **Speech-to-Text**: Web Speech API converts audio to text.
3. **Flask Backend**: Sends text query to `/chat` endpoint.
4. **AI Engine**: Google Gemini processes query (Ollama acts as fallback).
5. **Response**: Text is sent back to the browser.
6. **Visuals**: Browser triggers Text-to-Speech and audio-reactive animations.

> **Note:** Microphone access is handled on the client side due to browser security rules. This is the standard approach for modern AI web applications.

---

## 🛠️ Tech Stack

### Frontend
- **HTML5 & CSS3**: Glassmorphism, VFX, and Canvas Animations.
- **JavaScript**: Web Speech API & Canvas API.

### Backend
- **Python / Flask**: API routing and logic.
- **Google Gemini API**: Primary LLM.
- **Ollama**: Local LLM fallback.

---

## 📂 Project Structure

```text
Ask-A/
├── app.py              # Flask application
├── backend.py          # AI logic (Gemini + fallback)
├── memory.py           # Chat memory handling
├── index.html          # Complete immersive UI
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation