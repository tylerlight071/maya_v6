# MAYA

A local, voice-driven desktop AI companion. Wake-word activation, natural conversation, on-screen widgets, and a tool-calling LLM that can control music, search the web, read your screen, manage your calendar and email, generate images, and more — all running on your own machine.

> **Privacy model:** Everything runs locally. The LLM (Ollama), speech-to-text (faster-whisper), text-to-speech (F5-TTS), and face recognition all execute on-device. All backend services bind to `127.0.0.1` only. Personal data (memory, voice reference, credentials, biometric images) is git-ignored and never leaves the machine.

---

## Architecture

```
┌──────────────┐     spawns      ┌──────────────────────────┐
│  Electron    │ ──────────────► │  Python backend          │
│  (main.js)   │                 │  (server.py + modules)   │
│              │  ◄── WebSocket ─┤  Ollama · Whisper · F5-TTS│
│  React UI    │   (127.0.0.1)   │  InsightFace · tools     │
└──────────────┘                 └──────────────────────────┘
```

- **`Backend/`** — Python. `server.py` runs the audio loop, LLM streaming, and tool dispatch. Each widget has a companion module + its own localhost WebSocket microserver.
- **`Electron/`** — desktop shell. Spawns the backend, hosts the renderer, manages screen capture, browser/video webviews, and graceful shutdown.
- **`UI/`** — React + TypeScript + Vite renderer. The orb canvas, chat panel, and widgets.
- **`TTS/`** — voice reference for F5-TTS voice cloning.

### Port map

| Service | Port | Service | Port |
|---|---|---|---|
| Chat (main) | 8765 | ComfyUI bridge | 8770 |
| Vision / face | 8766 | System monitor | 8771 |
| Spotify | 8767 | Timer | 8772 |
| Weather | 8768 | Sticky notes | 8773 |
| Images | 8769 | Calendar | 8774 |
| | | News | 8775 |
| | | Alarm | 8776 |

---

## Prerequisites

- **Windows** (the backend uses PowerShell, `os.startfile`, and `pycaw` for volume control)
- **Python 3.11**
- **Node.js 18+**
- **An NVIDIA GPU with CUDA** — required for F5-TTS and recommended for Whisper/vision
- **[Ollama](https://ollama.com/)** running locally, with the models below pulled

### Pull the Ollama models

```bash
ollama pull huihui_ai/dolphin3-abliterated:8b-llama3.1-q4_K_M   # main chat LLM
ollama pull huihui_ai/gemma3-abliterated:4b                     # vision (multimodal)
ollama pull qwen2.5vl:7b                                        # screen analysis
```

(Model IDs are configured at the top of `Backend/server.py`.)

---

## Setup

### 1. Backend

```bash
cd Backend
python -m venv .venv
.venv\Scripts\activate

# Install PyTorch matching your CUDA version FIRST:
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt
```

### 2. UI / Electron

```bash
# from the repo root
npm run install-deps        # installs UI dependencies
cd Electron && npm install  # installs Electron
```

### 3. Configuration files (not committed — create these yourself)

These live in `Backend/models/` and are git-ignored:

| File | Purpose | How to get it |
|---|---|---|
| `config.json` | App settings: ComfyUI dir, file-search roots, app allowlist, ntfy topic, GPU assignment | See the template below |
| `google_credentials.json` | Gmail + Calendar OAuth client | [Google Cloud Console](https://console.cloud.google.com/) → OAuth desktop client. Tokens are created on first `authenticate calendar` / `connect Gmail`. |
| `spotify_config.json` | `{ "client_id": "...", "client_secret": "..." }` | [Spotify Developer Dashboard](https://developer.spotify.com/dashboard) |

Voice cloning reference goes in `TTS/models/voice_ref.wav` (a short clean recording of the target voice) with the matching transcript in `voice_ref.txt`.

<details>
<summary><code>config.json</code> template</summary>

```json
{
  "ntfy_topic": "Maya",
  "comfyui_dir": "C:/path/to/ComfyUI",
  "stt_gpu": 1,
  "tts_gpu": 0,
  "file_search_roots": [
    "C:/Users/you/Documents",
    "C:/Users/you/Desktop",
    "C:/Users/you/Downloads"
  ],
  "app_allowlist": {
    "notepad": "C:/Windows/notepad.exe",
    "calculator": "C:/Windows/System32/calc.exe",
    "explorer": "C:/Windows/explorer.exe"
  }
}
```

`file_search_roots` and `app_allowlist` are deliberate safety boundaries — the assistant can only search/open files under those roots and only launch apps on that allowlist.
</details>

---

## Running

```bash
# from the repo root — starts Vite, waits for it, then launches Electron + backend
cd UI && npm run dev
```

Or use `start.bat` from the repo root (Windows).

### Using it

1. Say **"Hey Maya"** (or "Hi/OK Maya", or just "Maya") to wake it.
2. Speak naturally — Maya transcribes, thinks, and replies in voice, opening relevant widgets as it goes.
3. End with a closing phrase ("thanks Maya", "that's all", "go to sleep") to return to idle.
4. You can also type in the chat panel, and attach images for the vision model.

**Shortcuts:** `Esc` toggles fullscreen. `Ctrl+Shift+I` opens DevTools (development builds only).

---

## Security notes

- All WebSocket servers bind to `127.0.0.1` and the Electron shell runs with `contextIsolation` on / `nodeIntegration` off, web security enabled, a minimal IPC bridge, and webview/navigation hardening.
- Camera/mic/screen permissions are granted only to the MAYA renderer origin, never to pages opened in the in-app browser.
- **Known limitation:** the localhost WebSocket servers do not yet enforce the generated session token (`models/.session_token`). Until they do, treat any process on your machine as able to reach them. See `MAYA_Code_Audit.md` for the full list and the recommended handshake fix.
- Never commit `Backend/models/` credential/cookie files or `Backend/memory/` — the `.gitignore` covers them, but double-check before pushing.

---

## Project layout

```
Maya/
├── Backend/            # Python backend
│   ├── server.py       # main loop: audio, LLM, tool dispatch
│   ├── memory.py       # persistent memory / episodes / projects
│   ├── vision.py       # face recognition + presence
│   ├── comfyui.py      # image generation bridge
│   ├── spotify.py · gmail.py · calendar_service.py · weather.py · news.py · ...
│   ├── models/         # configs, credentials, tokens  (git-ignored)
│   ├── memory/         # conversation memory            (git-ignored)
│   └── requirements.txt
├── Electron/           # main.js, preload.js
├── UI/                 # React + TS + Vite renderer
│   └── src/            # App, canvas, components, hooks
├── TTS/                # voice reference                (git-ignored)
└── start.bat
```
