# 🎭 Voice Agent with Personas

An interactive **AI Voice Agent** powered by:
- 🎙️ **AssemblyAI** → speech-to-text (live transcription)  
- 🤖 **Google Gemini** → persona-driven responses  
- 🗣️ **Murf.ai** → text-to-speech (optional)  
- 📰 **NewsAPI.org** → fetch latest news  

It supports multiple **Doraemon-themed personas** (Doraemon, Nobita, Shizuka, Gian, Suneo) with greetings, jokes, and real-time conversations.

---

## 🚀 Features
- 🎭 Persona switching (Doraemon, Nobita, Shizuka, Gian, Suneo)  
- ⚡ Persona actions: Greeting, Stand-up comedy, News updates  
- 🎙️ Real-time voice input (via WebSocket streaming)  
- 🗣️ AI voice output (Murf.ai TTS)  
- 🖥️ Clean HTML/JS UI with chat + audio playback  

---

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/voice-agent-personas.git
   cd voice-agent-personas
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Add environment variables**  
   Create a `.env` file in the project root:
   ```env
   ASSEMBLY_API_KEY=your_assemblyai_api_key
   GEMINI_API_KEY=your_gemini_api_key
   MURF_API_KEY=your_murf_api_key
   NEWS_API_KEY=your_newsapi_api_key
   ```

---

## ▶️ Running the App

1. **Start the FastAPI server**
   ```bash
   uvicorn assembly2:app --reload
   ```

2. **Open the UI**  
   Navigate to: [http://localhost:8000](http://localhost:8000)

---

## 🧩 Project Structure
```
.
├── index.html         # Frontend UI
├── assembly2.py       # FastAPI backend (personas, websocket, actions)
├── requirements.txt   # Python dependencies
├── README.md          # Documentation
└── .env               # API keys (ignored by git)
```

---

## 🛠️ Tech Stack
- **Frontend:** HTML, CSS, Vanilla JS  
- **Backend:** FastAPI (Python)  
- **APIs:** AssemblyAI, Google Gemini, Murf.ai, NewsAPI  

---

## ⚠️ Notes
- Without `murf` installed, audio replies won’t work (but text chat will still function).  
- API keys are required for all features to work properly.  

---

## 📜 License
MIT License © 2025
