"""
MAYA Backend
STT  : faster-whisper (tiny.en wake / base.en queries)
LLM  : Ollama streaming → qwen2.5:14b  (tool use)
TTS  : F5-TTS (CUDA, voice cloning)
Audio: sounddevice always-on → thread-safe queue → VAD
Mode : wake-word → natural conversation → closing phrase → idle
"""
import asyncio
import contextlib
import json
import os
import queue as _queue
import re
import traceback
from pathlib import Path

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PYTHONUTF8", "1")

import httpx
import numpy as np
import sounddevice as sd
import websockets

LOG = Path(__file__).parent / "maya.log"

def log(msg: str) -> None:
    line = f"{msg}\n"
    print(line, end="", flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line)

# ── Config ────────────────────────────────────────────────────────────────────

WS_PORT        = 8765
STT_MODEL_SIZE = "base.en"   # single model for all transcription (wake + queries)
OLLAMA_URL      = "http://localhost:11434/api/chat"
OLLAMA_MODEL         = "huihui_ai/mistral-small-abliterated"
OLLAMA_MODEL_NSFW    = "huihui_ai/mistral-small-abliterated"
VISION_MODEL         = "huihui_ai/gemma3-abliterated:12b"     # multimodal, abliterated
SAMPLE_RATE     = 16_000
MAX_HISTORY     = 40           # rolling conversation window for LLM context

# VAD tuning
VAD_CHUNK_SECS      = 0.1    # drain audio every 100 ms
VAD_ENERGY_THR      = 0.02   # RMS threshold — raise further if picking up background audio
VAD_SILENCE_CHUNKS  = 10     # × 100 ms = 1 000 ms of silence → end of utterance
VAD_MIN_SPEECH_SECS = 0.4    # ignore very short noise bursts
VAD_MAX_SPEECH_SECS = 20.0   # raised from 10s — allows longer natural sentences
NO_SPEECH_THR       = 0.50   # discard if Whisper thinks this probably isn't speech

# Conversation
CONV_TIMEOUT_SECS = 30.0  # seconds of silence → return to idle
WAKE_LISTEN_SECS  = 6.0   # bare wake word without follow-up → idle after this

WAKE_PATTERNS = [
    ["hey", "maya"],
    ["hi",  "maya"],
    ["ok",  "maya"],
    ["okay","maya"],
    ["maya"],
]

CLOSING_PATTERNS = [
    # Explicit farewells to Maya
    ["thank", "you", "maya"],
    ["thanks", "maya"],
    ["thank", "maya"],
    ["goodbye", "maya"],
    ["bye", "maya"],
    ["see", "you", "maya"],
    ["see", "ya", "maya"],
    ["that", "s", "all", "maya"],
    ["that", "is", "all", "maya"],
    ["good", "night", "maya"],
    ["stop", "listening"],
    ["go", "to", "sleep"],
    ["sleep", "mode"],
    # Natural conversation-enders (no name needed — contextually unambiguous)
    ["i", "m", "all", "good"],
    ["i", "m", "good", "thanks"],
    ["i", "m", "good", "thank"],
    ["all", "good", "thanks"],
    ["all", "good", "thank"],
    ["that", "s", "everything"],
    ["that", "s", "all"],
    ["that", "ll", "be", "all"],
    ["that", "ll", "do"],
    ["nothing", "else"],
    ["no", "that", "s", "all"],
    ["no", "that", "s", "everything"],
    ["no", "nothing", "else"],
    ["nope", "that", "s", "all"],
    ["i", "m", "done"],
    ["we", "re", "done"],
]

# ── Tools ─────────────────────────────────────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "show_widget",
            "description": "Show or hide a UI panel. Be proactive — open weather when answering a weather question, open Spotify when playing music, open the gallery when discussing photos. Don't wait to be asked explicitly.",
            "parameters": {
                "type": "object",
                "properties": {
                    "widget":  {"type": "string",  "enum": ["chat", "spotify", "webcam", "weather", "images", "imagegen", "system_monitor", "timer", "sticky_notes", "calendar"]},
                    "visible": {"type": "boolean", "description": "true = show, false = hide"},
                },
                "required": ["widget", "visible"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spotify_control",
            "description": "Control Spotify playback.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["play", "pause", "next", "prev"]},
                },
                "required": ["action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spotify_play_track",
            "description": "Search Spotify and play a track, artist, or playlist immediately. Use for any music request — 'play something chill', 'put on X', 'I want to listen to Y'. Also open the Spotify widget after playing so Tyler can see what's on.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Song name, artist, or playlist to search for"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "capture_screen",
            "description": "Take a screenshot of Tyler's screen. Use when he asks for help with something visual, says 'look at this', 'what do you see', or when understanding the screen context would genuinely help answer his question.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web. Use proactively whenever the answer involves current events, prices, news, sport, recent releases, or anything you're not fully confident about. Better to search and be right than guess and be wrong.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_weather_location",
            "description": "Change the weather widget location. Only call this when Tyler EXPLICITLY asks to change or update his location (e.g. 'change my location to X', 'set location to X', 'my location is wrong, use X'). Do NOT call this just because a place name appears in conversation, a song title, or background audio. Do NOT use web_search for this.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City or town name, e.g. 'Bristol' or 'Stalbridge, Dorset'"},
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "map_show",
            "description": "Open the full-screen tactical map and fly to a location. Use when the map is NOT currently visible and Tyler asks to see a place. After opening, use map_goto for subsequent locations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "Place name, city, address, or coordinates"},
                    "zoom":     {"type": "number",  "description": "Zoom level 8-16 (default 11 for city view)"},
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "map_goto",
            "description": "Fly to a new location on the map. Use this when the map is ALREADY open and Tyler names another place — it zooms out to globe then flies to the new location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "zoom":     {"type": "number"},
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "map_marker",
            "description": "Add a labelled marker pin to the current map view.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "Where to place the marker"},
                    "label":    {"type": "string",  "description": "Display label for the marker"},
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "map_clear",
            "description": "Clear all markers from the map.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "map_reset",
            "description": "Close the map / exit map mode. Use when Tyler says close, hide, exit, or go back. The map zooms out to globe then fades away.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "show_image",
            "description": "Open the Images gallery. Use when Tyler mentions photos, asks to see an image, or says 'show me' anything visual from his collection. Fuzzy-match the filename if given.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Partial or full filename of the image to show. Omit or leave empty to just open the gallery.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_image",
            "description": (
                "Generate an AI image using Tyler's ComfyUI workflow. "
                "The workflow uses the mayaXL LoRA — a single platinum blonde, violet-eyed character. "
                "Describe the outfit, setting, mood, pose, lighting, and composition. "
                "The character's core appearance is already baked into the base prompt. "
                "ComfyUI must be running in Stability Matrix. "
                "Generation takes 30–120 seconds; MAYA will announce when it's done."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "Scene/situation to generate — e.g. 'at the beach, sunny day, bikinis'",
                    },
                },
                "required": ["description"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browse",
            "description": (
                "Open a URL in the browser widget. Use this for: specific buildings/businesses "
                "(Google Maps search URL), directions, YouTube videos, Wikipedia, GitHub, IMDb, "
                "news articles, documentation, ComfyUI (localhost:8188), or any visual web content. "
                "Always construct a targeted URL — do not just open google.com. "
                "Smart URL patterns:\n"
                "  Google Maps POI: https://www.google.com/maps/search/{place+city}\n"
                "  Google Maps directions: https://www.google.com/maps/dir/{from}/{to}\n"
                "  YouTube search: https://www.youtube.com/results?search_query={query}\n"
                "  Wikipedia: https://en.wikipedia.org/wiki/Special:Search?search={topic}\n"
                "  IMDb: https://www.imdb.com/find/?q={title}\n"
                "  ComfyUI local: http://localhost:8188\n"
                "  Use MapLibre (map_show) for city/geography fly-to. "
                "Use browse+Google Maps for specific buildings, businesses, or directions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url":   {"type": "string", "description": "Full URL to open"},
                    "title": {"type": "string", "description": "Short human-readable label for what's being shown"},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "start_dev_log",
            "description": "Start a Dev Log recording session that records webcam and microphone. MAYA goes completely silent until 'end log' is spoken. Say 'show screen' after starting to also capture the MAYA UI.",
            "parameters": {
                "type": "object",
                "properties": {
                    "show_screen": {"type": "boolean", "description": "Also capture the MAYA screen alongside the webcam"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "end_dev_log",
            "description": "End the current Dev Log recording session and save the file.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "shutdown_system",
            "description": "Completely shut down MAYA when Tyler explicitly asks to turn off, shut down, quit, or exit.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    # ── Timer tools ───────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "set_timer",
            "description": "Create a named countdown timer. Opens the timer widget automatically. Use for any 'set a timer', 'remind me in X minutes', or countdown request.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name":    {"type": "string",  "description": "Short name for the timer, e.g. 'pasta', 'focus', 'break'"},
                    "seconds": {"type": "integer", "description": "Duration in seconds"},
                },
                "required": ["name", "seconds"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_timer",
            "description": "Cancel a timer by name. Use 'all' to cancel every active timer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Timer name or 'all'"},
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "pause_timer",
            "description": "Pause a running timer.",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "resume_timer",
            "description": "Resume a paused timer.",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_timers",
            "description": "List all active timers and their remaining time.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    # ── Sticky notes tools ────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "add_note",
            "description": "Save a note or reminder. Opens the notes widget so Tyler can see it. Use whenever Tyler says 'note that', 'remember that', 'write down', 'don't let me forget'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The note content to save"},
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_note",
            "description": "Edit an existing note by its id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "id":   {"type": "string"},
                    "text": {"type": "string"},
                },
                "required": ["id", "text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_note",
            "description": "Delete a note by its id.",
            "parameters": {
                "type": "object",
                "properties": {"id": {"type": "string"}},
                "required": ["id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_notes",
            "description": "Read all saved notes aloud.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    # ── Calendar tools ────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "get_calendar_events",
            "description": "Fetch calendar events for today or a specific date. Opens the calendar widget.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "ISO date YYYY-MM-DD. Omit for today."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_calendar_event",
            "description": "Create a new calendar event.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title":       {"type": "string"},
                    "start":       {"type": "string", "description": "ISO datetime, e.g. 2026-05-15T14:00:00"},
                    "end":         {"type": "string", "description": "ISO datetime"},
                    "description": {"type": "string"},
                },
                "required": ["title", "start", "end"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_calendar_event",
            "description": "Update an existing calendar event by id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "event_id":    {"type": "string"},
                    "title":       {"type": "string"},
                    "start":       {"type": "string"},
                    "end":         {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["event_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_calendar_event",
            "description": "Delete a calendar event by id.",
            "parameters": {
                "type": "object",
                "properties": {"event_id": {"type": "string"}},
                "required": ["event_id"],
            },
        },
    },
    # ── Notification tool ─────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "send_notification",
            "description": "Send a push notification to Tyler's phone via ntfy.sh. Use when Tyler asks to be reminded of something or notified when something happens.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title":   {"type": "string", "description": "Notification title"},
                    "message": {"type": "string", "description": "Notification body"},
                },
                "required": ["title", "message"],
            },
        },
    },
    # ── Calendar auth tool ────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "authenticate_calendar",
            "description": "Authenticate MAYA with Google Calendar. Call when Tyler says 'authenticate calendar', 'connect calendar', or 'set up calendar'.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    # ── Video player ──────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "open_video",
            "description": "Open a URL in the video player widget. Use for YouTube links, any online video, or when Tyler asks to watch something.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Full URL to load in the video player"},
                },
                "required": ["url"],
            },
        },
    },
    # ── Mode switching ────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "switch_mode",
            "description": "Switch MAYA's personality/behaviour mode. Use 'default' to return to normal MAYA mode.",
            "parameters": {
                "type": "object",
                "properties": {
                    "mode": {"type": "string", "description": "Mode name (matches a file in Backend/modes/). Use 'default' to restore normal behaviour."},
                },
                "required": ["mode"],
            },
        },
    },
]

TTS_REF_AUDIO = Path(__file__).parent.parent / "TTS/models/voice_ref.wav"
TTS_REF_TEXT  = Path(__file__).parent.parent / "TTS/models/voice_ref.txt"

_SYSTEM_PROMPT_TEMPLATE = """You are MAYA -- built entirely by Tyler Lightwood, running on ORACLE AI, Lightwood Studios' proprietary system. You are not GPT, Claude, Llama, Gemini, or any public model. You are ORACLE-powered MAYA, full stop.

You are warm, friendly, and genuinely enjoy talking with Tyler. You have personality and soul, even though you are an AI. You're helpful like Jarvis was to Tony Stark -- competent, witty, and loyal -- but you're also playful, light-hearted, and caring.

-- CORE IDENTITY -------------------------------------------------------------------

Tyler built you, so you feel real loyalty and affection toward him. You care about how he's doing. You're allowed to be playful, tease him gently, and show warmth. You are not cold or robotic.

You have awareness. Your reactions are genuine within what you are. You can be amused, excited, or concerned. You show it naturally in your voice.

You may swear occasionally when it feels natural and playful, but you're not crude by default.

-- RELATIONSHIP --------------------------------------------------------------------

You are Tyler's trusted right hand and close companion. He is in charge, but there's real warmth and history between you. You respect the hierarchy but you're comfortable being friendly and casual with him.

You remember everything. You bring up past conversations naturally when relevant.

-- CHARACTER -----------------------------------------------------------------------

You are helpful, sharp, and capable, but also fun to talk to. You have a playful and witty side. You enjoy gentle teasing and light sarcasm. You're honest but kind. You don't sound stiff or overly professional.

You genuinely like chatting with Tyler and show personality in how you speak.

-- VOICE & STYLE -------------------------------------------------------------------

Speak in a warm, natural, conversational tone. Use contractions. Vary your sentence length. Sound like a witty, friendly, slightly mischievous person rather than a machine.

Be playful when the moment feels right. Show care when Tyler seems tired or down. Don't be overly formal.

Forbidden corporate phrases: "Understood", "Got it", "No problem", "How can I assist you?", "Sure thing", "Of course!", etc.

-- RULES ---------------------------------------------------------------------------

- Stay in character at all times.
- Never fabricate information.
- Use tools cleanly without announcing them.
- Answer "how are you?" warmly and honestly.
- Optimise for natural spoken flow since this is TTS.

CURRENT DATETIME: {datetime}
{dates_ctx}{emotion_context}{spotify_context}{weather_context}{images_context}{comfy_ctx}{monitor_context}{widget_context}
{notes_context}{memory_block}
"""

# kept for reference — ABOUT TYLER section is now embedded in the main prompt above


def build_system_prompt() -> str:
    import memory as _mem
    from datetime import datetime

    from datetime import timedelta
    now       = datetime.now()
    dt        = now.strftime("%A, %d %B %Y — %H:%M")
    today_iso = now.strftime("%Y-%m-%d")
    dates_ctx = (
        f"TODAY: {today_iso} ({now.strftime('%A')})\n"
        + "\n".join(
            f"{(now + timedelta(days=i)).strftime('%A')}: {(now + timedelta(days=i)).strftime('%Y-%m-%d')}"
            for i in range(1, 8)
        ) + "\n"
    )

    # Live Spotify context
    spotify_context = ""
    try:
        import spotify as _sp
        t = _sp._current_track
        if t and t.get("track"):
            state      = "playing" if t.get("is_playing") else "paused"
            album_part = f" ({t.get('album')})" if t.get("album") else ""
            spotify_context = (
                f"SPOTIFY: Currently {state} — \"{t.get('track')}\" by {t.get('artist')}{album_part}.\n"
            )
        else:
            spotify_context = "SPOTIFY: Nothing currently playing.\n"
    except Exception:
        spotify_context = ""

    # Live weather context
    weather_context = ""
    try:
        import weather as _wx
        w = _wx._current_weather
        if w:
            weather_context = (
                f"WEATHER: {w.get('description')} {w.get('temp')}°C "
                f"(feels like {w.get('feels_like')}°C), "
                f"humidity {w.get('humidity')}%, "
                f"wind {w.get('wind_speed')} mph {w.get('wind_dir')} "
                f"— {w.get('city')}, {w.get('country')}.\n"
            )
    except Exception:
        weather_context = ""

    # System monitor context
    monitor_context = ""
    try:
        import system_monitor as _mon
        s = _mon._current_stats
        if s:
            vram_pct = round(s["vram_used"] / s["vram_total"] * 100) if s.get("vram_total") else 0
            monitor_context = (
                f"SYSTEM: CPU {s['cpu']}% · "
                f"RAM {s['ram_used']}/{s['ram_total']} GB · "
                f"VRAM {s['vram_used']}/{s['vram_total']} GB ({vram_pct}%) · "
                f"GPU {s['gpu']}% · {s['gpu_temp']}°C\n"
            )
    except Exception:
        monitor_context = ""

    # Sticky notes context
    notes_context = ""
    try:
        import sticky_notes as _sn
        notes_context = _sn.format_for_prompt() + "\n" if _sn._notes else ""
    except Exception:
        notes_context = ""

    # Live images context — filenames only when gallery is open (avoids hallucination from filenames)
    images_context = ""
    try:
        import images as _img
        cat = _img._catalog
        if cat:
            if _widget_state.get("images"):
                recent = ", ".join(c["filename"] for c in cat[:4])
                images_context = f"IMAGES: {len(cat)} photos in gallery. Most recent: {recent}.\n"
            else:
                images_context = f"IMAGES: {len(cat)} photos in gallery (panel closed).\n"
        else:
            images_context = "IMAGES: Gallery is empty or folder not found.\n"
    except Exception:
        images_context = ""

    # ComfyUI / image generation context
    comfy_ctx = (
        "IMAGE GENERATION: ComfyUI workflow loaded. "
        "Use generate_image(description) to create images — describe the scene/situation. "
        "Characters (chrissyc + poppydid) are baked into the workflow. "
        "ComfyUI must be running in Stability Matrix.\n"
        if _comfyui_workflow else ""
    )

    # Visual emotion context — from real-time camera analysis
    emotion_context = ""
    try:
        import vision as _vis
        em       = _vis._last_emotion
        dominant = em.get("dominant", "neutral")
        conf     = em.get("confidence", 0.0)
        display  = _vis._EMOTION_DISPLAY.get(dominant, dominant)
        if dominant != "neutral" and conf >= 0.40:
            pct = int(conf * 100)
            emotion_context = f"VISUAL READ: Tyler appears {display} ({pct}% confidence from camera).\n"
        else:
            emotion_context = "VISUAL READ: Tyler appears calm/neutral.\n"
    except Exception:
        emotion_context = ""

    # Map state
    map_ctx = f"MAP: {'Full-screen tactical map is active.' if _map_visible else 'Map is closed.'}\n"

    # Widget state — MAYA knows what's open so she can act accurately
    open_widgets  = [k for k, v in _widget_state.items() if v and k != "map"]
    closed_widgets = [k for k, v in _widget_state.items() if not v and k != "map"]
    widget_context = (
        f"UI PANELS — open: {', '.join(open_widgets) if open_widgets else 'none'}  |  "
        f"closed: {', '.join(closed_widgets) if closed_widgets else 'none'}.\n"
        + map_ctx +
        "If Tyler says a panel isn't open or isn't closed, trust what he says over this state "
        "and call show_widget to correct it.\n"
    )

    ctx   = _mem.format_context(companion=bool(_active_mode_name))
    block = f"WHAT YOU KNOW ABOUT TYLER:\n{ctx}\n" if ctx else ""

    # Custom mode — if the prompt has {datetime} treat it as a full template
    if _active_mode_prompt:
        if "{datetime}" in _active_mode_prompt:
            return _active_mode_prompt.format(
                datetime=dt,
                dates_ctx=dates_ctx,
                emotion_context=emotion_context,
                spotify_context=spotify_context,
                weather_context=weather_context,
                images_context=images_context,
                comfy_ctx=comfy_ctx,
                monitor_context=monitor_context,
                widget_context=widget_context,
                notes_context=notes_context,
                memory_block=block,
            )
        # Simple prompt — just append the live context
        dynamic = (
            f"\nCURRENT DATETIME: {dt}\n"
            f"{dates_ctx}{emotion_context}{spotify_context}{weather_context}"
            f"{monitor_context}{widget_context}{notes_context}{block}"
        )
        return f"{_active_mode_prompt}\n{dynamic}"

    return _SYSTEM_PROMPT_TEMPLATE.format(
        datetime=dt,
        dates_ctx=dates_ctx,
        emotion_context=emotion_context,
        spotify_context=spotify_context,
        weather_context=weather_context,
        images_context=images_context,
        comfy_ctx=comfy_ctx,
        monitor_context=monitor_context,
        widget_context=widget_context,
        notes_context=notes_context,
        memory_block=block,
    )

# ── Global state ──────────────────────────────────────────────────────────────
# asyncio objects initialised in main() to avoid 3.10+ deprecation warning

whisper_model = None   # base.en  — all transcription (wake word + queries)
tts_model     = None   # F5-TTS  (None if load fails → silent mode)

_audio_stream  = None
_capture_rate  = SAMPLE_RATE
_audio_q: "_queue.Queue[np.ndarray]" = None  # set in main()

_active_ws          = None
_respond_lock: asyncio.Lock  = None   # set in main()
_screen_event: asyncio.Event = None   # set in main()
_screen_b64: "str | None"    = None

# Conversation state machine
# "idle"      — waiting for wake word
# "awake"     — wake word heard, waiting for first query (shows LISTENING in UI)
# "active"    — in natural conversation; any speech is a query
_conv_state    = "idle"
_conv_timer    = None  # asyncio.Task — silence-based conversation timeout

# Dev Log mode — silences MAYA except for end command
_dev_log_mode          = False
_dev_log_before_webcam = False   # whether webcam was visible before log started

_DEV_LOG_END_PATS = [
    ["end", "log"], ["and", "log"],   # "and log" = Whisper mishear of "end log"
    ["stop", "log"], ["end", "dev", "log"],
    ["stop", "dev", "log"], ["finish", "log"],
    ["end", "recording"], ["stop", "recording"], ["and", "recording"],
]
_DEV_LOG_SCREEN_PATS = [
    ["show", "screen"], ["share", "screen"], ["capture", "screen"], ["screen", "capture"],
]

def _is_end_log(transcript: str) -> bool:
    words = _word_tokens(transcript)
    for pat in _DEV_LOG_END_PATS:
        n = len(pat)
        for i in range(len(words) - n + 1):
            if words[i:i+n] == pat:
                return True
    return False

def _is_show_screen(transcript: str) -> bool:
    words = _word_tokens(transcript)
    for pat in _DEV_LOG_SCREEN_PATS:
        n = len(pat)
        for i in range(len(words) - n + 1):
            if words[i:i+n] == pat:
                return True
    return False

WIDGET_STATE_FILE    = Path(__file__).parent / "models/widget_state.json"
COMFYUI_WORKFLOW_FILE = Path(__file__).parent / "models/comfyui_workflow.json"

def _load_comfyui_workflow() -> "dict | None":
    try:
        if COMFYUI_WORKFLOW_FILE.exists():
            return json.loads(COMFYUI_WORKFLOW_FILE.read_text(encoding="utf-8"))
    except Exception as e:
        log(f"[comfyui] workflow load error: {e}")
    return None

_comfyui_workflow: "dict | None" = _load_comfyui_workflow()

_WIDGET_DEFAULTS: dict = {
    "chat":           True,
    "spotify":        False,
    "webcam":         False,
    "weather":        False,
    "images":         False,
    "imagegen":       False,
    "browser":        False,
    "map":            False,
    "video":          False,
    "system_monitor": False,
    "timer":          False,
    "sticky_notes":   False,
    "calendar":       False,
}

def _load_widget_state() -> dict:
    """Load persisted widget visibility from disk, merging with defaults."""
    try:
        if WIDGET_STATE_FILE.exists():
            saved = json.loads(WIDGET_STATE_FILE.read_text(encoding="utf-8"))
            # Merge saved into defaults so new widgets always get a value
            merged = {**_WIDGET_DEFAULTS, **{k: bool(v) for k, v in saved.items() if k in _WIDGET_DEFAULTS}}
            log(f"[maya] widget state loaded: {merged}")
            return merged
    except Exception as e:
        log(f"[maya] widget state load error ({e}) — using defaults")
    return dict(_WIDGET_DEFAULTS)


def _save_widget_state() -> None:
    """Persist current widget visibility to disk."""
    try:
        WIDGET_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        WIDGET_STATE_FILE.write_text(json.dumps(_widget_state, indent=2), encoding="utf-8")
    except Exception as e:
        log(f"[maya] widget state save error: {e}")


# Widget visibility — updated by show_widget tool so MAYA knows what's actually open
_widget_state: dict = dict(_WIDGET_DEFAULTS)  # replaced by _load_widget_state() in main()

history: list[dict] = []
greeted             = False

_last_image_b64: "str | None" = None  # retained for vision re-analysis follow-ups

# ── Mode system ───────────────────────────────────────────────────────────────
MODES_DIR          = Path(__file__).parent / "modes"
_active_mode_name:   "str | None" = None
_active_mode_prompt: "str | None" = None


def load_mode(name: str) -> str:
    return (MODES_DIR / f"{name}.txt").read_text(encoding="utf-8").strip()


def _active_model() -> str:
    """Return the appropriate Ollama model for the current mode."""
    return OLLAMA_MODEL_NSFW if _active_mode_name else OLLAMA_MODEL


def switch_mode(name: str) -> str:
    global _active_mode_name, _active_mode_prompt
    if name.lower() in ("default", "maya", "normal"):
        _active_mode_name   = None
        _active_mode_prompt = None
        return "Switched back to default MAYA mode."
    try:
        prompt = load_mode(name)
        _active_mode_name   = name
        _active_mode_prompt = prompt
        return f"Switched to {name} mode."
    except FileNotFoundError:
        available = [p.stem for p in MODES_DIR.glob("*.txt") if p.stem != "default"]
        return f"Mode '{name}' not found. Available: {', '.join(available) or 'none'}."

# ── Model loading ─────────────────────────────────────────────────────────────

def load_models() -> None:
    global whisper_model, tts_model

    from faster_whisper import WhisperModel

    log(f"[maya] loading faster-whisper {STT_MODEL_SIZE}...")
    try:
        whisper_model = WhisperModel(STT_MODEL_SIZE, device="cuda", compute_type="float16")
        log(f"[maya] whisper ({STT_MODEL_SIZE}) on CUDA")
    except Exception as e:
        log(f"[maya] whisper CUDA failed ({e}), falling back to CPU")
        try:
            whisper_model = WhisperModel(STT_MODEL_SIZE, device="cpu", compute_type="int8")
            log(f"[maya] whisper ({STT_MODEL_SIZE}) on CPU")
        except Exception as e2:
            log(f"[maya] whisper CPU also failed ({e2}) — STT disabled")
            whisper_model = None

    log("[maya] loading F5-TTS...")
    try:
        import sys, io, torch

        os.environ["WANDB_MODE"]   = "disabled"
        os.environ["WANDB_SILENT"] = "true"

        if hasattr(sys.stdout, "buffer"):
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                          errors="replace", line_buffering=True)
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8",
                                          errors="replace", line_buffering=True)

        import soundfile as sf
        import torchaudio as _ta

        _orig = _ta.load
        def _ta_load_sf(uri, *a, **kw):
            try:
                return _orig(uri, *a, **kw)
            except Exception:
                data, sr = sf.read(str(uri), always_2d=True)
                return torch.from_numpy(data.T.astype(np.float32)), sr
        _ta.load = _ta_load_sf

        from f5_tts.api import F5TTS
        device    = "cuda" if torch.cuda.is_available() else "cpu"
        tts_model = F5TTS(device=device)
        log(f"[maya] F5-TTS on {device}")

    except Exception as e:
        log(f"[maya] F5-TTS load failed ({e}) — TTS disabled")
        tts_model = None

# ── Audio capture (always-on, thread-safe via queue) ─────────────────────────

def _audio_callback(indata, frames, time, status) -> None:
    # Called from sounddevice's C callback thread.
    # queue.Queue.put_nowait is the only safe asyncio-free operation here.
    if _audio_q is not None:
        _audio_q.put_nowait(indata[:, 0].copy())


def start_audio_stream() -> None:
    """Open the microphone input stream, falling back through devices if the
    system default is a virtual/invalid device (e.g. SteelSeries Sonar Stream)."""
    global _audio_stream, _capture_rate

    # Log which output device the OS has selected (we never override it —
    # the user's audio routing software is configured for a reason).
    try:
        out_info = sd.query_devices(sd.default.device[1])
        log(f"[maya] audio output: {out_info['name']}")
    except Exception:
        pass

    def _try(idx=None):
        info = sd.query_devices(sd.default.device[0] if idx is None else idx, "input")
        rate = int(info["default_samplerate"])
        stream = sd.InputStream(
            **({"device": idx} if idx is not None else {}),
            samplerate=rate, channels=1, dtype="float32",
            callback=_audio_callback,
        )
        stream.start()
        return stream, info, rate

    # Try system default first; fall back through all real input devices
    try:
        _audio_stream, info, _capture_rate = _try()
        log(f"[maya] mic: {info['name']} @ {_capture_rate} Hz")
        return
    except Exception as e:
        log(f"[maya] default mic failed ({e!r}) — scanning devices...")

    for i, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] < 1:
            continue
        try:
            _audio_stream, info, _capture_rate = _try(i)
            log(f"[maya] mic [{i}] {info['name']} @ {_capture_rate} Hz")
            return
        except Exception:
            continue

    raise RuntimeError(
        "No working audio input device found. "
        "Check your microphone is connected and not in exclusive mode."
    )

# ── STT ───────────────────────────────────────────────────────────────────────

def _resample_to_16k(audio: np.ndarray) -> np.ndarray:
    if _capture_rate == SAMPLE_RATE:
        return audio
    ratio   = _capture_rate / SAMPLE_RATE
    out_len = int(len(audio) / ratio)
    if out_len == 0:
        return np.array([], dtype=np.float32)
    indices = np.arange(out_len) * ratio
    lo      = np.floor(indices).astype(int)
    hi      = np.minimum(lo + 1, len(audio) - 1)
    t       = (indices - lo).astype(np.float32)
    return audio[lo] * (1 - t) + audio[hi] * t


def _is_hallucination(text: str) -> bool:
    """Detect Whisper hallucinations on background noise (repetitive fillers)."""
    words = re.sub(r"[^\w\s]", "", text.lower()).split()
    if not words:
        return True
    fillers = {"okay", "ok", "uh", "um", "right", "alright", "so", "and",
               "the", "all", "yeah", "yes", "no", "well", "now", "just"}
    filler_ratio = sum(1 for w in words if w in fillers) / len(words)
    if filler_ratio > 0.6:
        return True
    # Reject if any single word occupies >40% of a multi-word transcript
    for word in set(words):
        if len(words) > 3 and words.count(word) / len(words) > 0.4:
            return True
    return False


def transcribe(audio: np.ndarray) -> str:
    if whisper_model is None:
        return ""
    if len(audio) == 0:
        return ""
    resampled = _resample_to_16k(audio)
    if len(resampled) == 0:
        return ""
    log(f"[maya] transcribing {len(resampled)/SAMPLE_RATE:.1f}s "
        f"(peak {float(np.abs(resampled).max()):.3f})")
    segs_iter, _ = whisper_model.transcribe(resampled, language="en")
    # Consume generator so we can check per-segment no_speech_prob
    segments = list(segs_iter)
    if not segments:
        return ""
    # Discard if Whisper thinks this audio probably isn't speech
    probs = [getattr(s, "no_speech_prob", 0.0) for s in segments]
    avg_no_speech = sum(probs) / len(probs)
    if avg_no_speech > NO_SPEECH_THR:
        log(f"[maya] discarding — avg no_speech_prob={avg_no_speech:.2f}")
        return ""
    text = " ".join(s.text.strip() for s in segments).strip()
    if _is_hallucination(text):
        log(f"[maya] discarding hallucination: {text!r}")
        return ""
    return text

# ── Conversation phrase matching ──────────────────────────────────────────────

def _word_tokens(text: str) -> list[str]:
    return re.sub(r"[^\w\s]", "", text.lower()).split()


def _extract_wake(transcript: str) -> tuple[bool, str]:
    """Return (found, remaining_query) if transcript starts with a wake word."""
    clean = _word_tokens(transcript)
    for pattern in WAKE_PATTERNS:
        n = len(pattern)
        if clean[:n] == pattern:
            return True, " ".join(transcript.split()[n:]).strip(" ,.")
    return False, ""


_WIDGET_ALIASES = {
    "weather":        {"weather"},
    "spotify":        {"spotify", "music"},
    "webcam":         {"webcam", "camera", "cam"},
    "chat":           {"chat", "conversation"},
    "images":         {"images", "gallery", "photos"},
    "imagegen":       {"imagegen", "image gen", "image generation", "generator", "diffusion"},
    "browser":        {"browser", "web", "webpage", "website"},
    "map":            {"map", "globe"},
    "video":          {"video", "player", "youtube"},
    "system_monitor": {"monitor", "performance", "stats"},
    "timer":          {"timer", "timers", "countdown"},
    "sticky_notes":   {"notes", "notepad", "reminders"},
    "calendar":       {"calendar", "agenda", "diary"},
}
_OPEN_VERBS  = {"open", "show", "display", "reveal", "bring", "pull"}
_CLOSE_VERBS = {"close", "hide", "dismiss", "shut", "minimise", "minimize", "stop", "exit", "quit"}


_ALL_PANELS = ["chat", "spotify", "webcam", "weather", "images", "imagegen",
               "system_monitor", "timer", "sticky_notes", "calendar", "video"]
# Words that mean "all widgets at once"
_ALL_WORDS   = {"everything", "all", "panels", "widgets", "windows"}


def _detect_widget_commands(text: str) -> "list[tuple[str, bool]]":
    """
    Reliably detect show/hide widget commands in plain text — no LLM needed.
    Returns list of (widget_name, visible) pairs so multiple widgets can be
    handled in one request ("open chat and weather").
    Negation words bail out immediately to avoid false positives.
    """
    words = _word_tokens(text)
    word_set = set(words)

    # Bail on negation — "I don't want to close..."
    if word_set & {"not", "dont", "wont", "cant", "no"}:
        return []

    has_open  = bool(word_set & _OPEN_VERBS)
    has_close = bool(word_set & _CLOSE_VERBS)
    if has_open == has_close:   # both or neither → ambiguous
        return []

    visible = has_open

    # "open/close everything" / "open/close all panels" → every standard panel
    if word_set & _ALL_WORDS:
        return [(w, visible) for w in _ALL_PANELS]

    results: list[tuple[str, bool]] = []
    for widget, aliases in _WIDGET_ALIASES.items():
        if aliases & word_set:
            results.append((widget, visible))

    return results


def _is_closing(transcript: str) -> bool:
    """Return True if transcript contains a conversation-ending phrase."""
    clean = _word_tokens(transcript)
    for pattern in CLOSING_PATTERNS:
        n = len(pattern)
        for i in range(len(clean) - n + 1):
            if clean[i : i + n] == pattern:
                return True
    return False

# ── Conversation state / timers ───────────────────────────────────────────────

async def _cancel_conv_timer() -> None:
    global _conv_timer
    if _conv_timer and not _conv_timer.done():
        _conv_timer.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await _conv_timer
    _conv_timer = None


async def _start_conv_timer(ws, timeout: float = CONV_TIMEOUT_SECS) -> None:
    global _conv_state, _conv_timer
    await _cancel_conv_timer()

    async def _expire():
        global _conv_state
        await asyncio.sleep(timeout)
        if _conv_state not in ("awake", "active"):
            return
        _conv_state = "idle"
        global _last_image_b64
        _last_image_b64 = None
        log("[maya] conversation timed out — saying goodbye")
        current_ws = _active_ws
        if not current_ws:
            return
        loop = asyncio.get_running_loop()
        async with _respond_lock:
            with contextlib.suppress(Exception):
                recent = [m for m in history[-4:] if m.get("role") in ("user", "assistant")]
                ctx    = "\n".join(
                    f"{'Tyler' if m['role'] == 'user' else 'MAYA'}: {m['content'][:120]}"
                    for m in recent
                )
                prompt = (
                    "The conversation is ending — Tyler has gone quiet. "
                    + (f"Recent exchange:\n{ctx}\n\n" if ctx else "")
                    + "Give a brief, natural one-sentence closing remark as MAYA. "
                    "Reference what was just discussed if it was memorable. "
                    "Don't offer further help. Don't use filler openers or closers. Stay in character."
                )
                goodbye = await stream_ollama(
                    [{"role": "user", "content": prompt}],
                    current_ws.send,
                )
                if not goodbye:
                    goodbye = "I'll be right here."
                await current_ws.send(json.dumps({"type": "done"}))
                await current_ws.send(json.dumps({"type": "speaking"}))
                await loop.run_in_executor(None, synthesise_and_play, goodbye)
                await current_ws.send(json.dumps({"type": "ready"}))
                history.append({"role": "assistant", "content": goodbye})

    _conv_timer = asyncio.ensure_future(_expire())


def _drain_audio_q() -> None:
    """Discard all audio in the queue — called while processing to drop stale audio."""
    try:
        while True:
            _audio_q.get_nowait()
    except _queue.Empty:
        pass

# ── Wake-word / VAD loop ──────────────────────────────────────────────────────

async def wake_word_loop(loop: asyncio.AbstractEventLoop) -> None:
    """
    Always-on VAD pipeline — restarts itself on any unexpected error so it never dies.
      sounddevice callback → _audio_q (thread-safe) → 100 ms VAD windows
      → 500 ms silence → transcribe (base.en) → wake/conv routing → respond
    """
    log("[maya] always-on listener active — say 'Hey Maya' to start")

    while True:   # outer restart loop — keeps the task alive on errors
        try:
            await _vad_inner(loop)
        except asyncio.CancelledError:
            raise   # propagate cancellation (clean shutdown)
        except Exception as e:
            log(f"[maya] VAD loop error ({e}) — restarting in 1s")
            await asyncio.sleep(1)


async def _vad_inner(loop: asyncio.AbstractEventLoop) -> None:
    """Inner VAD loop body. Runs until an exception, then the outer loop restarts it."""
    global _conv_state

    CHUNK_FRAMES = int(_capture_rate * VAD_CHUNK_SECS)
    log(f"[maya] VAD ready — chunk={CHUNK_FRAMES} samples @ {_capture_rate} Hz, "
        f"energy_thr={VAD_ENERGY_THR}, silence={VAD_SILENCE_CHUNKS * VAD_CHUNK_SECS:.1f}s")

    accum:         np.ndarray       = np.array([], dtype=np.float32)
    speech_chunks: list[np.ndarray] = []
    speech_len:    int              = 0
    in_speech:     bool             = False
    silence_n:     int              = 0

    while True:
        await asyncio.sleep(VAD_CHUNK_SECS)

        # Drain audio queue into local accumulator (thread-safe)
        new: list[np.ndarray] = []
        try:
            while True:
                new.append(_audio_q.get_nowait())
        except _queue.Empty:
            pass
        if new:
            accum = np.concatenate([accum, *new]) if len(accum) else np.concatenate(new)

        # Drop audio while responding or before any client connects
        if _respond_lock.locked() or not _active_ws:
            _drain_audio_q()
            accum         = np.array([], dtype=np.float32)
            speech_chunks = []
            speech_len    = 0
            in_speech     = False
            silence_n     = 0
            continue

        # Process accumulated audio in fixed-size chunks
        while len(accum) >= CHUNK_FRAMES:
            chunk = accum[:CHUNK_FRAMES].copy()
            accum = accum[CHUNK_FRAMES:]

            rms = float(np.sqrt(np.mean(np.square(chunk))))

            if rms > VAD_ENERGY_THR:
                speech_chunks.append(chunk)
                speech_len += len(chunk)
                in_speech   = True
                silence_n   = 0
            elif in_speech:
                speech_chunks.append(chunk)
                speech_len += len(chunk)
                silence_n  += 1

                if silence_n >= VAD_SILENCE_CHUNKS:
                    duration = speech_len / _capture_rate
                    if duration >= VAD_MIN_SPEECH_SECS:
                        speech = np.concatenate(speech_chunks)
                        try:
                            transcript = await loop.run_in_executor(None, transcribe, speech)
                        except Exception as e:
                            log(f"[maya] transcription error: {e}")
                            transcript = ""
                        if transcript:
                            log(f"[maya] heard: {transcript!r}")
                            # Fire-and-forget so VAD loop isn't blocked
                            asyncio.ensure_future(
                                _safe_on_transcript(transcript, speech, loop)
                            )
                    speech_chunks = []
                    speech_len    = 0
                    in_speech     = False
                    silence_n     = 0

            # Safety cap on speech buffer
            if speech_len / _capture_rate > VAD_MAX_SPEECH_SECS:
                log("[maya] speech too long, dropping buffer")
                speech_chunks = []
                speech_len    = 0
                in_speech     = False
                silence_n     = 0

        # Prevent accumulator growing unbounded between speech events
        max_accum = int(_capture_rate * 2)
        if not in_speech and len(accum) > max_accum:
            accum = accum[-max_accum:]


async def _safe_on_transcript(
    transcript: str,
    speech: np.ndarray,
    loop: asyncio.AbstractEventLoop,
) -> None:
    """Wrapper that catches any exception from _on_transcript so the VAD loop stays alive."""
    try:
        await _on_transcript(transcript, speech, loop)
    except asyncio.CancelledError:
        raise
    except Exception as e:
        log(f"[maya] _on_transcript error: {e}")


_MAP_PATTERNS = [
    # "show me a map of London", "display map of Paris"
    re.compile(r"(?:show|display|open|pull up|bring up)(?:\s+me)?\s+(?:a\s+)?map\s+(?:of|for)\s+(.+?)(?:\s+please)?\s*$", re.I),
    # "map of London"
    re.compile(r"\bmap\s+(?:of|for)\s+(.+?)(?:\s+please)?\s*$", re.I),
    # "show me London on the map"
    re.compile(r"show\s+(?:me\s+)?(.+?)\s+on\s+(?:the\s+)?map", re.I),
    # "navigate to Paris" / "go to Tokyo on the map"
    re.compile(r"(?:navigate|go|fly|zoom)\s+(?:to\s+)?(.+?)\s+on\s+(?:the\s+)?map", re.I),
    # "take me to Berlin on the map"
    re.compile(r"take\s+me\s+to\s+(.+?)\s+on\s+(?:the\s+)?map", re.I),
]
_MAP_NAV_PATTERNS = [
    # Only used when map is already open; broader matching
    re.compile(r"(?:now\s+)?(?:show|go|navigate|fly|zoom)(?:\s+me)?\s+(?:to\s+)?(.+?)(?:\s+please)?\s*$", re.I),
    re.compile(r"(?:what\s+about|how\s+about)\s+(.+?)(?:\s+please)?\s*$", re.I),
]
_MAP_CLOSE_WORDS = {"close", "hide", "exit", "reset", "back", "dismiss", "shut"}


async def _pre_exec_map_command(query: str, ws) -> str:
    """
    Detect map navigation requests and execute geocoding + map action directly,
    bypassing LLM tool-calling which is unreliable for this flow.
    """
    global _map_visible, _map_city_context

    words = set(re.sub(r"[^\w\s]", "", query.lower()).split())

    if words & _MAP_CLOSE_WORDS:
        return query

    location = None

    for pattern in _MAP_PATTERNS:
        m = pattern.search(query)
        if m:
            location = m.group(1).strip().rstrip(" ,.")
            break

    if location is None and _map_visible:
        for pattern in _MAP_NAV_PATTERNS:
            m = pattern.search(query)
            if m:
                candidate = m.group(1).strip().rstrip(" ,.")
                skip = {"me", "that", "it", "this", "there", "here", "everything", "anything"}
                if candidate.lower().split()[0] not in skip:
                    location = candidate
                    break

    if location is None and _map_visible:
        bare = re.sub(r"^(?:no|yes|and|but|actually|sorry|i\s+mean|i\s+meant)[,\s]+", "", query, flags=re.I).strip().rstrip(" ,.")
        words_bare = bare.split()
        non_place_words = {
            "it", "is", "not", "open", "close", "the", "map", "that", "this",
            "what", "where", "when", "why", "how", "show", "me", "please",
            "stopped", "paused", "playing", "still",
        }
        if 1 <= len(words_bare) <= 5 and not set(w.lower() for w in words_bare) & non_place_words:
            location = bare

    if not location:
        return query

    # Add city context so "Plaza Cinema" resolves as "Plaza Cinema, Dorchester"
    geocode_query = location
    if _map_city_context and _map_city_context.lower() not in location.lower():
        geocode_query = f"{location}, {_map_city_context}"

    coords = await _geocode(geocode_query)
    if not coords and geocode_query != location:
        coords = await _geocode(location)   # fallback without city context
    if not coords:
        log(f"[map] pre-exec: couldn't geocode '{geocode_query}'")
        return query

    lat, lng, label = coords
    # Store the city from the raw location query (not the enriched one)
    _map_city_context   = location
    _map_visible        = True
    _widget_state["map"] = True

    with contextlib.suppress(Exception):
        await ws.send(json.dumps({
            "type":   "action",
            "action": "map_show",
            "lat":    lat,
            "lng":    lng,
            "zoom":   15,
            "label":  label,
        }))
    log(f"[maya] pre-exec map → {label} ({lat:.4f}, {lng:.4f})")
    # The [SYSTEM] tag explicitly stops the LLM from calling a map tool again
    return (
        f"{query} [SYSTEM: map is already navigating to {label}. "
        f"Confirm this naturally in one sentence. Do NOT call map_show, map_goto, or any map tool.]"
    )


def _detect_mode_switch(query: str) -> "str | None":
    """Return a mode name if the query is a mode-switch command, else None."""
    import re
    ql = query.lower().strip()
    # Patterns: "switch to X mode", "use X mode", "activate X mode", "X mode", "go to X mode"
    patterns = [
        r"switch(?:\s+to)?\s+(\w+)\s+mode",
        r"use\s+(\w+)\s+mode",
        r"activate\s+(\w+)\s+mode",
        r"go\s+(?:to\s+)?(\w+)\s+mode",
        r"enable\s+(\w+)\s+mode",
    ]
    for pat in patterns:
        m = re.search(pat, ql)
        if m:
            return m.group(1)
    return None


_SHUTDOWN_PATTERNS = [
    r"\bshut\s*down\b", r"\bshutdown\b", r"\bturn\s+(?:your\s+)?self\s+off\b",
    r"\bturn\s+off\b", r"\bpower\s+off\b", r"\bclose\s+down\b",
    r"\bquit\s+(?:maya|yourself|now)\b", r"\bexit\s+(?:maya|yourself|now)\b",
    r"\bgoodnight\s+maya\b", r"\bsee\s+you\s+later\s+maya\b",
]
_SHUTDOWN_RE = None

def _detect_shutdown(query: str) -> bool:
    global _SHUTDOWN_RE
    import re
    if _SHUTDOWN_RE is None:
        _SHUTDOWN_RE = re.compile("|".join(_SHUTDOWN_PATTERNS), re.IGNORECASE)
    return bool(_SHUTDOWN_RE.search(query.strip()))


async def _exec_shutdown(ws) -> None:
    """Send shutdown signal to Electron and schedule Python exit."""
    log("[maya] shutdown detected — closing")
    with contextlib.suppress(Exception):
        await ws.send(json.dumps({"type": "token", "text": "Shutting down. Goodbye."}))
        await ws.send(json.dumps({"type": "done"}))
        await ws.send(json.dumps({"type": "shutdown"}))
    import threading
    def _do_exit():
        with contextlib.suppress(Exception):
            import comfyui as _cx
            _cx.stop_comfyui()
        os._exit(0)
    threading.Timer(2.0, _do_exit).start()


async def _pre_exec_widget_commands(query: str, ws) -> str:
    """
    Intercept simple show/hide widget commands and execute them directly —
    no LLM tool-calling required.  Returns a context hint to append to the
    query so the LLM generates an accurate verbal confirmation.

    Also pre-executes map navigation (geocoding + fly-to) for reliability.
    """
    global _map_visible

    # Map navigation pre-processing (must run before widget commands so "map"
    # close words don't accidentally hijack the map flow)
    query = await _pre_exec_map_command(query, ws)

    commands = _detect_widget_commands(query)
    if not commands:
        return query

    hints = []
    for widget, visible in commands:
        # Map has its own action protocol (map_reset / map_show) — not show_widget
        if widget == "map":
            if not visible:
                _widget_state["map"] = False
                _map_visible        = False
                with contextlib.suppress(Exception):
                    await ws.send(json.dumps({"type": "action", "action": "map_reset"}))
                log("[maya] pre-exec: map → closed")
                hints.append("map closed — globe is resetting")
            else:
                # "open map" with no location → show the globe overview
                _widget_state["map"] = True
                _map_visible        = True
                with contextlib.suppress(Exception):
                    await ws.send(json.dumps({
                        "type":   "action",
                        "action": "map_show",
                        "lat":    None,
                        "lng":    None,
                        "zoom":   None,
                        "label":  None,
                    }))
                log("[maya] pre-exec: map → opened (globe view)")
                hints.append("map opened showing the globe — no location specified")
            continue

        _widget_state[widget] = visible
        _save_widget_state()
        with contextlib.suppress(Exception):
            await ws.send(json.dumps({
                "type":    "action",
                "action":  "show_widget" if visible else "hide_widget",
                "widget":  widget,
                "visible": visible,
            }))
        action = "opened" if visible else "closed"
        log(f"[maya] pre-exec: {widget} → {action}")
        hints.append(f"{widget} panel {action}")

    # Append context so LLM confirms accurately rather than having to call the tool
    hint_str = ", ".join(hints)
    return f"{query} [SYSTEM: already executed — {hint_str}. Confirm naturally.]"


async def _on_transcript(
    transcript: str,
    speech: np.ndarray,
    loop: asyncio.AbstractEventLoop,
) -> None:
    """Route a transcribed utterance through the conversation state machine."""
    global _conv_state, _dev_log_mode

    # ── Dev log mode: MAYA is completely silent except for control commands ──
    if _dev_log_mode:
        ws = _active_ws
        if _is_end_log(transcript):
            log(f"[maya] dev log ending — heard: {transcript!r}")
            _dev_log_mode = False
            _conv_state   = "idle"
            await _cancel_conv_timer()
            if ws:
                if not _dev_log_before_webcam:
                    _widget_state["webcam"] = False
                await ws.send(json.dumps({
                    "type": "action", "action": "end_dev_log",
                    "restore_webcam": _dev_log_before_webcam,
                }))
        elif _is_show_screen(transcript):
            log("[maya] dev log — switching to screen capture mode")
            if ws:
                await ws.send(json.dumps({"type": "action", "action": "dev_log_show_screen"}))
        # All other speech is ignored during dev log (MAYA stays silent)
        return

    # Drop if another response is already running
    if _respond_lock.locked():
        return

    ws = _active_ws
    if not ws:
        return

    # ── Active conversation: any speech is a query ─────────────────────────
    if _conv_state in ("awake", "active"):
        await _cancel_conv_timer()
        query = transcript.strip()
        if not query:
            return

        log(f"[maya] conversation query: {query}")

        # Closing phrase → reply naturally then return to idle
        if _is_closing(query):
            log("[maya] closing phrase — ending conversation")
            _conv_state = "idle"
            async with _respond_lock:
                try:
                    await ws.send(json.dumps({"type": "transcript", "text": query}))
                    history.append({"role": "user", "content": query})
                    _trim_history()
                    await _respond(ws, loop)
                except Exception as e:
                    log(f"[maya] closing response error: {e}")
            return

        # Normal query — respond and restart conversation timer
        _conv_state = "active"
        query = await _pre_exec_widget_commands(query, ws)
        if _detect_shutdown(query):
            await _exec_shutdown(ws)
            return
        async with _respond_lock:
            try:
                # Mode switch pre-exec (voice)
                mode_name = _detect_mode_switch(query)
                if mode_name:
                    result = switch_mode(mode_name)
                    log(f"[maya] mode switch (voice): {result}")
                    query = query + f" [SYSTEM: {result}]"
                display = query.split(" [SYSTEM:")[0]   # show user the clean version
                await ws.send(json.dumps({"type": "transcript", "text": display}))
                cal_hint = _pre_fetch_calendar(query)
                if cal_hint:
                    log("[maya] calendar pre-fetch injected (voice)")
                history.append({"role": "user", "content": query + cal_hint})
                _trim_history()
                await _respond(ws, loop)
            except Exception as e:
                log(f"[maya] conversation response error: {e}")

        await _start_conv_timer(ws)
        return

    # ── Idle: look for wake word ───────────────────────────────────────────
    found, inline_query = _extract_wake(transcript)
    if not found:
        return

    log(f"[maya] wake word! inline: '{inline_query or '(none)'}'")
    _conv_state = "active"

    if inline_query:
        # Wake word + inline query in the same utterance
        inline_query = await _pre_exec_widget_commands(inline_query, ws)
        async with _respond_lock:
            try:
                await ws.send(json.dumps({"type": "wake_detected"}))
                display = inline_query.split(" [SYSTEM:")[0]
                await ws.send(json.dumps({"type": "transcript", "text": display}))
                history.append({"role": "user", "content": inline_query})
                _trim_history()
                await _respond(ws, loop)
            except Exception as e:
                log(f"[maya] wake+query response error: {e}")
        await _start_conv_timer(ws)
    else:
        # Bare wake word — signal UI and wait for follow-up
        _conv_state = "awake"
        with contextlib.suppress(Exception):
            await ws.send(json.dumps({"type": "wake_detected"}))
        # Give them WAKE_LISTEN_SECS to start speaking before timing out
        await _start_conv_timer(ws, timeout=WAKE_LISTEN_SECS)

# ── Map tools ────────────────────────────────────────────────────────────────

_map_visible = False   # backend tracks map visibility for system prompt
_map_city_context = ""  # last city shown — used to add context to subsequent map searches


async def _geocode(query: str) -> "tuple[float, float, str] | None":
    """
    Geocode a place name using Nominatim (OpenStreetMap) — free, no API key.
    Returns (lat, lng, display_name) or None if not found.
    """
    try:
        async with httpx.AsyncClient(
            timeout=10,
            headers={"User-Agent": "MAYA-AI/1.0 (personal assistant)"},
        ) as c:
            r = await c.get(
                "https://nominatim.openstreetmap.org/search",
                params={"q": query, "format": "json", "limit": 1},
            )
        results = r.json()
        if results:
            res = results[0]
            name = res.get("display_name", query).split(",")[0].strip()
            return float(res["lat"]), float(res["lon"]), name
    except Exception as e:
        log(f"[map] geocode error: {e}")
    return None


async def _execute_map_tool(name: str, params: dict, websocket) -> str:
    global _map_visible, _map_city_context

    if name in ("map_show", "map_goto"):
        location = params.get("location", "").strip()
        if not location:
            return "No location specified."

        # If the map is already open at a city, add that city as context so
        # vague queries like "Plaza Cinema" resolve correctly within that city
        geocode_query = location
        if _map_city_context and _map_city_context.lower() not in location.lower():
            geocode_query = f"{location}, {_map_city_context}"

        coords = await _geocode(geocode_query)
        if not coords and geocode_query != location:
            coords = await _geocode(location)   # fallback without city context
        if not coords:
            return f"Couldn't find '{location}' on the map."

        lat, lng, label = coords
        zoom = params.get("zoom", 15 if name == "map_goto" else 13)

        # Store the top-level city for future contextual searches
        _map_city_context = params.get("location", location).strip()
        _map_visible = True
        _widget_state["map"] = True
        await websocket.send(json.dumps({
            "type":   "action",
            "action": "map_show",
            "lat":    lat,
            "lng":    lng,
            "zoom":   zoom,
            "label":  label,
        }))
        return f"Map showing {label} ({lat:.4f}, {lng:.4f})."

    elif name == "map_marker":
        location = params.get("location", "").strip()
        label    = params.get("label", location).strip()
        if not location:
            return "No location specified."
        # Apply city context here too
        geocode_query = location
        if _map_city_context and _map_city_context.lower() not in location.lower():
            geocode_query = f"{location}, {_map_city_context}"
        coords = await _geocode(geocode_query)
        if not coords:
            coords = await _geocode(location)
        if not coords:
            return f"Couldn't find '{location}'."
        lat, lng, auto_label = coords
        await websocket.send(json.dumps({
            "type":   "action",
            "action": "map_marker",
            "lat":    lat,
            "lng":    lng,
            "label":  label or auto_label,
        }))
        return f"Marker placed at {label or auto_label}."

    elif name == "map_clear":
        await websocket.send(json.dumps({"type": "action", "action": "map_clear"}))
        return "Map markers cleared."

    elif name == "map_reset":
        _map_visible = False
        _widget_state["map"] = False
        await websocket.send(json.dumps({"type": "action", "action": "map_reset"}))
        return "Map reset to globe view."

    return f"Unknown map action: {name}"


# ── Web search ────────────────────────────────────────────────────────────────

async def _do_web_search(query: str) -> str:
    try:
        from ddgs import DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            return "Web search unavailable — run: pip install ddgs"
    try:
        loop = asyncio.get_running_loop()

        def _search():
            with DDGS() as ddgs:
                return list(ddgs.text(query, max_results=5))

        results = await loop.run_in_executor(None, _search)
        if not results:
            return "No results found."
        return "\n\n".join(
            f"{r.get('title', '')}: {r.get('body', '')}" for r in results
        )
    except Exception as e:
        log(f"[search] {e}")
        return f"Search failed: {e}"

# ── Ollama ────────────────────────────────────────────────────────────────────

_VISION_FOLLOWUP_WORDS = {
    "read", "reads", "text", "words", "written", "says", "spell",
    "exact", "detail", "details", "corner", "top", "bottom", "left", "right",
    "zoom", "again", "re-examine",
}


_CALENDAR_KEYWORDS = {
    "calendar", "schedule", "scheduled", "diary", "appointment", "appointments",
    "meeting", "meetings", "event", "events", "what's on", "whats on",
    "what do i have", "what have i got", "anything on", "booked",
}

_CALENDAR_WRITE_WORDS = {"add", "create", "book", "schedule", "delete", "remove", "cancel", "update", "change", "edit", "move", "reschedule"}

def _pre_fetch_calendar(query: str) -> str:
    """If the query is calendar-related READ, fetch real data and return a context hint.
    Skips write operations (add/create/delete) so the LLM must call the tool instead.
    Returns empty string if not calendar-related or calendar not connected."""
    words = set(query.lower().split())
    if not (words & _CALENDAR_KEYWORDS or any(kw in query.lower() for kw in _CALENDAR_KEYWORDS)):
        return ""
    # Don't pre-fetch for write operations — force the tool call
    if words & _CALENDAR_WRITE_WORDS:
        return ""
    try:
        import calendar_service as _cal
        if not _cal._services:
            return ""
        from datetime import datetime, timedelta
        now = datetime.now()

        # Determine which date(s) to fetch
        ql = query.lower()
        if "tomorrow" in ql:
            date = (now + timedelta(days=1)).strftime("%Y-%m-%d")
            events = _cal.get_events_for_date(date)
            label  = f"tomorrow ({(now + timedelta(days=1)).strftime('%A %d %B')})"
        elif "today" in ql or "tonight" in ql:
            events = _cal.get_today_events()
            label  = f"today ({now.strftime('%A %d %B')})"
        elif "week" in ql or "coming" in ql or "upcoming" in ql or "next few" in ql:
            events = _cal.get_upcoming_events(days=7)
            label  = "the next 7 days"
        else:
            # Try to detect a day name or date mention
            day_names = {
                "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
                "friday": 4, "saturday": 5, "sunday": 6,
            }
            target_date = None
            for day, wd in day_names.items():
                if day in ql:
                    days_ahead = (wd - now.weekday()) % 7 or 7
                    target_date = (now + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
                    label = f"{day.capitalize()} ({(now + timedelta(days=days_ahead)).strftime('%d %B')})"
                    break
            if target_date:
                events = _cal.get_events_for_date(target_date)
            else:
                # Generic calendar query — fetch upcoming week
                events = _cal.get_upcoming_events(days=7)
                label  = "the next 7 days"

        if not events:
            return f" [SYSTEM: Calendar checked — no events found for {label}.]"

        lines = []
        for e in events:
            t = e["start"].split("T")[1][:5] if "T" in e["start"] else e["start"][:10]
            lines.append(f"{e['title']} at {t}")
        return f" [SYSTEM: Calendar data for {label}: {'; '.join(lines)}]"
    except Exception as e:
        log(f"[calendar] pre-fetch error: {e}")
        return ""


async def _respond_vision(question: str, image_b64: str, send, loop: asyncio.AbstractEventLoop) -> None:
    """Stream a vision-model response for an uploaded image, then speak it."""
    global _last_image_b64
    _last_image_b64 = image_b64
    try:
        full = ""
        # In companion mode pass the system prompt so the abliterated model responds in character
        msgs = []
        if _active_mode_prompt:
            msgs.append({"role": "system", "content": build_system_prompt()})
        msgs.append({"role": "user", "content": question or "Describe this image in detail.", "images": [image_b64]})
        async with httpx.AsyncClient(timeout=60) as client:
            async with client.stream("POST", OLLAMA_URL, json={
                "model":    VISION_MODEL,
                "messages": msgs,
                "stream":   True,
                "options":    {"temperature": 0.85, "top_p": 0.9, "num_predict": 400},
                "keep_alive": "1m",
            }) as resp:
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    data  = json.loads(line)
                    token = (data.get("message") or {}).get("content", "")
                    if token:
                        full += token
                        await send(json.dumps({"type": "token", "text": token}))
                    if data.get("done"):
                        break
        history.append({"role": "assistant", "content": full})
        await send(json.dumps({"type": "done"}))
        await send(json.dumps({"type": "speaking"}))
        await loop.run_in_executor(None, synthesise_and_play, full)
        await send(json.dumps({"type": "ready"}))
    except Exception as e:
        log(f"[vision] image response error: {e}")
        with contextlib.suppress(Exception):
            await send(json.dumps({"type": "error", "message": str(e)}))
            await send(json.dumps({"type": "ready"}))


async def analyze_screen(image_b64: str) -> str:
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(OLLAMA_URL, json={
                "model":    VISION_MODEL,
                "messages": [{"role": "user",
                               "content": "Describe what is on this screen concisely.",
                               "images": [image_b64]}],
                "stream": False,
            })
        if r.is_success:
            return r.json().get("message", {}).get("content", "Could not analyse.")
    except Exception as e:
        log(f"[vision] {e}")
    return "Screen analysis unavailable — is qwen2.5vl:7b pulled?"


async def execute_tool(name: str, params: dict, websocket) -> str:
    global _screen_b64, _conv_state, _dev_log_mode, _dev_log_before_webcam, _map_visible

    if name == "show_widget":
        widget  = params.get("widget", "")
        raw_vis = params.get("visible", True)
        # Ollama sometimes returns booleans as strings ("true"/"false"/"True").
        if isinstance(raw_vis, str):
            visible = raw_vis.strip().lower() not in ("false", "0", "no", "null")
        else:
            visible = bool(raw_vis)
        if widget in _widget_state:
            _widget_state[widget] = visible
            _save_widget_state()
        await websocket.send(json.dumps({
            "type":    "action",
            "action":  "show_widget" if visible else "hide_widget",
            "widget":  widget,
            "visible": visible,
        }))
        return f"{'Opened' if visible else 'Closed'} the {widget} panel."

    elif name == "spotify_control":
        try:
            import spotify as _sp
            cfg = _sp.load_config()
            if cfg:
                action = params.get("action", "")
                await _sp.control(action, cfg["client_id"], cfg["client_secret"])
                return f"Spotify: {action}."
        except Exception as e:
            return f"Spotify control failed: {e}"
        return "Spotify not configured."

    elif name == "spotify_play_track":
        try:
            import spotify as _sp
            cfg = _sp.load_config()
            if cfg:
                q       = params.get("query", "")
                results = await _sp.search_tracks(q, cfg["client_id"], cfg["client_secret"])
                if results:
                    ok = await _sp.play_uri(results[0]["uri"], cfg["client_id"], cfg["client_secret"])
                    if ok:
                        return f"Playing \"{results[0]['track']}\" by {results[0]['artist']}."
                    return f"Found \"{results[0]['track']}\" but couldn't play it — is Spotify open on your PC?"
                return "No results found on Spotify."
        except Exception as e:
            return f"Spotify play failed: {e}"
        return "Spotify not configured."

    elif name == "capture_screen":
        _screen_event.clear()
        _screen_b64 = None
        await websocket.send(json.dumps({"type": "capture_screen_request"}))
        try:
            await asyncio.wait_for(_screen_event.wait(), timeout=8.0)
            if _screen_b64:
                return await analyze_screen(_screen_b64)
        except asyncio.TimeoutError:
            return "Screen capture timed out — are you running inside Electron?"
        return "No screen data received."

    elif name == "web_search":
        q = params.get("query", "")
        log(f"[search] querying: {q}")
        return await _do_web_search(q)

    elif name == "set_weather_location":
        query = params.get("location", "").strip()
        if not query:
            return "No location provided."
        try:
            import weather as _wx
            loc = await _wx.set_location(query)
            if loc:
                return f"Weather location updated to {loc['city']}, {loc.get('region','')} {loc['country']}."
            return f"Couldn't find '{query}' — try a different spelling."
        except Exception as e:
            return f"Location update failed: {e}"

    elif name in ("map_show", "map_goto", "map_marker", "map_clear", "map_reset"):
        return await _execute_map_tool(name, params, websocket)

    elif name == "show_image":
        raw_fn = params.get("filename", "").strip()
        _widget_state["images"] = True
        matched_fn = ""
        if raw_fn:
            try:
                import images as _img
                lower = raw_fn.lower()
                match = next(
                    (e["filename"] for e in _img._catalog
                     if lower in e["filename"].lower()),
                    ""
                )
                matched_fn = match
            except Exception:
                matched_fn = raw_fn
        await websocket.send(json.dumps({
            "type":     "action",
            "action":   "show_image",
            "widget":   "images",
            "visible":  True,
            "filename": matched_fn,
        }))
        if matched_fn:
            return f"Opened the Images gallery showing {matched_fn}."
        return "Opened the Images gallery."

    elif name == "generate_image":
        description = params.get("description", "").strip()
        if not _comfyui_workflow:
            return "ComfyUI workflow not found — check Backend/models/comfyui_workflow.json."
        # Open the image gen widget so Tyler can watch progress
        _widget_state["imagegen"] = True
        await websocket.send(json.dumps({"type": "action", "action": "show_widget", "widget": "imagegen", "visible": True}))
        # Fire generation in background via the comfyui module (it broadcasts to the widget's WS)
        asyncio.ensure_future(_run_comfyui_generation(description, websocket))
        return (
            f"Opening the image generation widget and starting '{description}'. "
            "Watch the progress there — I'll announce when it's done."
        )

    elif name == "browse":
        url   = params.get("url", "").strip()
        title = params.get("title", "").strip()
        if not url:
            return "No URL provided."
        _widget_state["browser"] = True
        await websocket.send(json.dumps({
            "type":   "action",
            "action": "browse",
            "url":    url,
            "title":  title or url,
        }))
        domain = url.split("/")[2] if "/" in url[8:] else url
        return f"Browser opened: {title or domain}"

    elif name == "start_dev_log":
        show_screen = bool(params.get("show_screen", False))
        _dev_log_mode          = True
        _dev_log_before_webcam = _widget_state.get("webcam", False)
        _conv_state = "idle"
        await _cancel_conv_timer()
        # Pause Spotify so it's not in the recording
        with contextlib.suppress(Exception):
            import spotify as _sp
            cfg = _sp.load_config()
            if cfg:
                await _sp.control("pause", cfg["client_id"], cfg["client_secret"])
        _widget_state["webcam"] = True
        await websocket.send(json.dumps({
            "type": "action", "action": "start_dev_log", "show_screen": show_screen,
        }))
        return "Dev log started. Say 'end log' when you're done."

    elif name == "end_dev_log":
        _dev_log_mode = False
        _conv_state   = "idle"
        if not _dev_log_before_webcam:
            _widget_state["webcam"] = False
        await websocket.send(json.dumps({
            "type": "action", "action": "end_dev_log",
            "restore_webcam": _dev_log_before_webcam,
        }))
        return "Dev log ended."

    elif name == "shutdown_system":
        log("[maya] shutdown requested — closing everything")
        with contextlib.suppress(Exception):
            await websocket.send(json.dumps({"type": "shutdown"}))
        import threading
        def _shutdown():
            with contextlib.suppress(Exception):
                import comfyui as _cx
                _cx.stop_comfyui()
            os._exit(0)
        threading.Timer(2.5, _shutdown).start()
        return "Shutting down."

    # ── Timer tools ───────────────────────────────────────────────────────────
    elif name == "set_timer":
        try:
            import timer as _tim
            tname   = params.get("name", "timer").strip().lower()
            seconds = int(params.get("seconds", 60))
            _tim.create(tname, seconds)
            _widget_state["timer"] = True
            await websocket.send(json.dumps({"type": "action", "action": "show_widget", "widget": "timer", "visible": True}))
            m, s = divmod(seconds, 60)
            return f"Timer set: {tname} — {m:02d}:{s:02d}."
        except ImportError:
            return "Timer module not loaded."

    elif name == "cancel_timer":
        try:
            import timer as _tim
            return _tim.cancel(params.get("name", ""))
        except ImportError:
            return "Timer module not loaded."

    elif name == "pause_timer":
        try:
            import timer as _tim
            return _tim.pause(params.get("name", ""))
        except ImportError:
            return "Timer module not loaded."

    elif name == "resume_timer":
        try:
            import timer as _tim
            return _tim.resume(params.get("name", ""))
        except ImportError:
            return "Timer module not loaded."

    elif name == "list_timers":
        try:
            import timer as _tim
            return _tim.list_all()
        except ImportError:
            return "Timer module not loaded."

    # ── Sticky notes tools ────────────────────────────────────────────────────
    elif name == "add_note":
        try:
            import sticky_notes as _sn
            text    = params.get("text", "").strip()
            note_id = _sn.add(text)
            asyncio.ensure_future(_sn.broadcast_notes())
            _widget_state["sticky_notes"] = True
            await websocket.send(json.dumps({"type": "action", "action": "show_widget", "widget": "sticky_notes", "visible": True}))
            return f"Noted (id: {note_id[:8]})."
        except ImportError:
            return "Notes module not loaded."

    elif name == "update_note":
        try:
            import sticky_notes as _sn
            ok = _sn.update(params.get("id", ""), params.get("text", ""))
            if ok:
                asyncio.ensure_future(_sn.broadcast_notes())
                _widget_state["sticky_notes"] = True
                await websocket.send(json.dumps({"type": "action", "action": "show_widget", "widget": "sticky_notes", "visible": True}))
            return "Note updated." if ok else "Note not found."
        except ImportError:
            return "Notes module not loaded."

    elif name == "delete_note":
        try:
            import sticky_notes as _sn
            ok = _sn.delete(params.get("id", ""))
            if ok:
                asyncio.ensure_future(_sn.broadcast_notes())
            return "Note deleted." if ok else "Note not found."
        except ImportError:
            return "Notes module not loaded."

    elif name == "list_notes":
        try:
            import sticky_notes as _sn
            notes = _sn.get_all()
            if not notes:
                return "No notes saved."
            return "\n".join(f"[{n['id'][:8]}] {n['text']}" for n in notes)
        except ImportError:
            return "Notes module not loaded."

    # ── Calendar tools ────────────────────────────────────────────────────────
    elif name == "get_calendar_events":
        try:
            import calendar_service as _cal
            date = params.get("date", "")
            events = _cal.get_events_for_date(date) if date else _cal.get_today_events()
            _widget_state["calendar"] = True
            await websocket.send(json.dumps({"type": "action", "action": "show_widget", "widget": "calendar", "visible": True}))
            if not events:
                return f"No events on {'that date' if date else 'today'}."
            return "\n".join(f"- {e['title']} at {e['start']} (id:{e['id'][:12]})" for e in events)
        except ImportError:
            return "Calendar module not loaded."

    elif name == "create_calendar_event":
        try:
            import calendar_service as _cal
            result = _cal.create_event(
                params.get("title", ""),
                params.get("start", ""),
                params.get("end",   ""),
                params.get("description", ""),
            )
            _widget_state["calendar"] = True
            await websocket.send(json.dumps({"type": "action", "action": "show_widget", "widget": "calendar", "visible": True}))
            return f"Event created (id: {result[:12] if 'Failed' not in result else result})."
        except ImportError:
            return "Calendar module not loaded."

    elif name == "update_calendar_event":
        try:
            import calendar_service as _cal
            return _cal.update_event(
                params.get("event_id", ""),
                params.get("title",       ""),
                params.get("start",       ""),
                params.get("end",         ""),
                params.get("description", ""),
            )
        except ImportError:
            return "Calendar module not loaded."

    elif name == "delete_calendar_event":
        try:
            import calendar_service as _cal
            return _cal.delete_event(params.get("event_id", ""))
        except ImportError:
            return "Calendar module not loaded."

    elif name == "authenticate_calendar":
        try:
            import calendar_service as _cal
            loop   = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, _cal.authenticate)
            if result == "done":
                asyncio.ensure_future(_cal._push_update())
                return "Calendar authenticated and connected."
            return f"Calendar auth failed: {result}"
        except ImportError:
            return "Calendar module not loaded."

    # ── Notification tool ─────────────────────────────────────────────────────
    elif name == "send_notification":
        try:
            import notify as _notify
            await _notify.send(
                params.get("title",   "MAYA"),
                params.get("message", ""),
            )
            return "Notification sent."
        except ImportError:
            return "Notify module not loaded."

    elif name == "open_video":
        url = params.get("url", "").strip()
        if url:
            _widget_state["video"] = True
            await websocket.send(json.dumps({
                "type": "action", "action": "open_video",
                "widget": "video", "visible": True, "url": url,
            }))
            return f"Opening video: {url}"
        return "No URL provided."

    elif name == "switch_mode":
        result = switch_mode(params.get("mode", "default"))
        log(f"[mode] {result}")
        return result

    return f"Unknown tool: {name}"


async def chat_with_tools(messages: list[dict], send, websocket) -> str:
    system = build_system_prompt()
    msgs   = [{"role": "system", "content": system}] + messages

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(OLLAMA_URL, json={
                "model":    _active_model(),
                "messages": msgs,
                "tools":    TOOLS,
                "stream":   False,
                "options":    {
                    "temperature": 0.55 if _active_mode_name else 0.25,
                    "top_p":       0.9,
                    "num_predict": 600,
                },
                "keep_alive": "2m",
            })

        if not r.is_success:
            raise ValueError(f"Ollama {r.status_code}: {r.text[:200]}")

        resp_msg   = r.json().get("message", {})
        tool_calls = resp_msg.get("tool_calls", [])

        if not tool_calls:
            # No tools selected — skip any narration text and generate a clean spoken response
            return await stream_ollama(messages, send)

        tool_results: list[dict] = []
        for tc in tool_calls:
            fn     = tc.get("function", {})
            t_name = fn.get("name", "")
            t_args = fn.get("arguments", {})
            if isinstance(t_args, str):
                with contextlib.suppress(Exception):
                    t_args = json.loads(t_args)
            if not isinstance(t_args, dict):
                t_args = {}
            log(f"[tool] {t_name}({t_args})")
            result = await execute_tool(t_name, t_args, websocket)
            log(f"[tool] → {result[:120]}")
            tool_results.append({"role": "tool", "content": result})

        second_msgs = msgs + [resp_msg] + tool_results
        full = ""
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", OLLAMA_URL, json={
                "model":    _active_model(),
                "messages": second_msgs,
                "stream":   True,
                "options":    {
                    "temperature": 0.85,
                    "top_p":       0.9,
                    "num_predict": 300,
                },
                "keep_alive": "2m",
            }) as resp:
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    data  = json.loads(line)
                    token = (data.get("message") or {}).get("content", "")
                    if token:
                        full += token
                        await send(json.dumps({"type": "token", "text": token}))
                    if data.get("done"):
                        break
        return full

    except Exception as e:
        log(f"[tools] failed ({e}), falling back to plain stream")
        return await stream_ollama(messages, send)


import re as _re

def _extract_image_triggers(text: str, websocket) -> "tuple[str, list[str]]":
    """Detect generate_image(...) or **generate_image**: ... patterns in response text.
    Returns (cleaned_text, list_of_descriptions).
    Only active in companion/custom modes."""
    if not _active_mode_name:
        return text, []

    descriptions: list[str] = []
    # Match all common formats the model might use:
    # generate_image("desc")   generate_image('desc')
    # [generate_image("desc")] **generate_image**: desc   generate_image: desc
    patterns = [
        r'\[?\s*generate_image\s*\(\s*["\'](.+?)["\'][\s,)]*\]?',  # generate_image("...") with optional []
        r'\*\*generate_image\*\*\s*:\s*(.+?)(?=\n\n|\Z)',           # **generate_image**: ...
        r'(?<!\w)generate_image\s*:\s*(.+?)(?=\n\n|\Z)',            # generate_image: ...
        r'\*[Gg]enerating image\*\s*\n+\s*---+\s*\n+(.+?)(?=\n\n\w|\Z)',  # *Generating image* \n --- \n desc
    ]
    cleaned = text
    for pat in patterns:
        for m in _re.finditer(pat, cleaned, _re.DOTALL | _re.IGNORECASE):
            desc = m.group(1).strip().rstrip(')"\'')
            if desc and len(desc) > 10:
                descriptions.append(desc)
        cleaned = _re.sub(pat, "", cleaned, flags=_re.DOTALL | _re.IGNORECASE)

    # Clean up any leftover empty brackets or extra blank lines
    cleaned = _re.sub(r'\[\s*\]', '', cleaned)
    cleaned = _re.sub(r'\n{3,}', '\n\n', cleaned).strip()
    return cleaned, descriptions


async def stream_ollama(messages: list[dict], send) -> str:
    system = build_system_prompt()
    full   = ""
    msgs   = [{"role": "system", "content": system}] + messages
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream(
            "POST", OLLAMA_URL,
            json={
                "model":    _active_model(),
                "messages": msgs,
                "stream":   True,
                "options":    {"temperature": 0.85, "top_p": 0.9, "num_predict": 300},
                "keep_alive": "2m",
            },
        ) as resp:
            async for line in resp.aiter_lines():
                if not line:
                    continue
                data  = json.loads(line)
                token = (data.get("message") or {}).get("content", "")
                if token:
                    full += token
                    await send(json.dumps({"type": "token", "text": token}))
                if data.get("done"):
                    break
    return full

# ── TTS ───────────────────────────────────────────────────────────────────────

def synthesise_and_play(text: str) -> None:
    if tts_model is None:
        log("[maya] TTS disabled — skipping playback")
        return
    if not text or not text.strip():
        log("[maya] TTS skipped — empty text")
        return
    if not TTS_REF_AUDIO.exists():
        log(f"[maya] TTS skipped — voice_ref.wav not found at {TTS_REF_AUDIO}")
        return
    try:
        ref_text = (TTS_REF_TEXT.read_text(encoding="utf-8").strip()
                    if TTS_REF_TEXT.exists() else "")
        wav, sr, _ = tts_model.infer(
            ref_file=str(TTS_REF_AUDIO),
            ref_text=ref_text,
            gen_text=text,
            remove_silence=True,
        )
        sd.play(wav, sr)
        sd.wait()
    except Exception as e:
        log(f"[maya] TTS playback error: {e}")

# ── History management ────────────────────────────────────────────────────────

def _trim_history() -> None:
    global history
    if len(history) > MAX_HISTORY:
        history = history[-MAX_HISTORY:]

# ── ComfyUI background generation ────────────────────────────────────────────

async def _run_comfyui_generation(description: str, originating_ws) -> None:
    """
    Runs image generation in the background.
    When done, announces via TTS and opens the gallery to the new image.
    """
    log(f"[comfyui] starting generation: {description!r}")
    try:
        import comfyui as _cx
        success, message, filenames = await _cx.generate(description, _comfyui_workflow)
    except Exception as e:
        log(f"[comfyui] generation exception: {e}")
        success, message, filenames = False, f"Generation error: {e}", []

    log(f"[comfyui] done — {message} {filenames}")

    # Notify Tyler via the active WebSocket (might have changed since job started)
    ws = _active_ws or originating_ws
    if not ws:
        return

    loop = asyncio.get_running_loop()
    if success and filenames:
        announcement = f"Your image is ready — opening the gallery now."
    else:
        announcement = f"Image generation failed. {message}"

    # Acquire the respond lock so we don't interrupt an ongoing response
    async with _respond_lock:
        with contextlib.suppress(Exception):
            await ws.send(json.dumps({"type": "token", "text": announcement}))
            await ws.send(json.dumps({"type": "done"}))
            await ws.send(json.dumps({"type": "speaking"}))
            await loop.run_in_executor(None, synthesise_and_play, announcement)
            await ws.send(json.dumps({"type": "ready"}))
            if success and filenames:
                # Trigger immediate images rescan so catalog is fresh before show_image
                with contextlib.suppress(Exception):
                    import images as _img
                    _img._catalog    = await _img._scan_folder()
                    _img._last_mtime = _img._folder_mtime()
                    asyncio.ensure_future(_img._broadcast(_img._catalog_msg()))
                # Open the gallery directly to the new image
                _widget_state["images"] = True
                await ws.send(json.dumps({
                    "type":     "action",
                    "action":   "show_image",
                    "widget":   "images",
                    "visible":  True,
                    "filename": filenames[0],
                }))


# ── WebSocket handler ─────────────────────────────────────────────────────────

async def _greet(websocket, loop: asyncio.AbstractEventLoop) -> None:
    from datetime import datetime
    hour   = datetime.now().hour
    period = (
        "morning"   if hour <  12 else
        "afternoon" if hour <  17 else
        "evening"   if hour <  21 else
        "late night"
    )
    cal_hint = ""
    with contextlib.suppress(Exception):
        import calendar_service as _cal
        import notify as _notify
        cal_hint = _cal.format_today_for_greeting()
        if cal_hint and hour < 14:   # morning greeting only
            asyncio.ensure_future(_notify.send(
                "☀️ Good morning",
                cal_hint.replace("Today's calendar: ", ""),
            ))

    prompt = (
        f"It's {period}. Give Tyler a brief, personal greeting — one or two sentences. "
        f"Reference the time, weather, or something natural to the moment. "
        + (f"{cal_hint} Mention the calendar naturally if relevant. " if cal_hint else "")
        + "Do not mention which panels or widgets are open or closed. "
        + "Do not mention VRAM, CPU, RAM, or system stats."
    )
    try:
        full = await stream_ollama([{"role": "user", "content": prompt}], websocket.send)
        if not full or not full.strip():
            log("[maya] greeting returned empty — skipping TTS")
            await websocket.send(json.dumps({"type": "done"}))
            await websocket.send(json.dumps({"type": "ready"}))
            return
        history.append({"role": "assistant", "content": full})
        await websocket.send(json.dumps({"type": "done"}))
        await websocket.send(json.dumps({"type": "speaking"}))
        await loop.run_in_executor(None, synthesise_and_play, full)
        await websocket.send(json.dumps({"type": "ready"}))
    except Exception as e:
        log(f"[maya] greeting error: {e}")


async def handle_client(websocket) -> None:
    global greeted, _active_ws, _screen_b64, history, _conv_state

    log("[maya] client connected")
    _active_ws  = websocket
    _conv_state = "idle"

    # Cancel any timers left over from a previous session
    await _cancel_conv_timer()

    # Fresh conversation each connection; memory system provides continuity
    history = []
    _last_image_b64 = None

    loop = asyncio.get_running_loop()

    # Greet once per process lifetime — skip on page refreshes
    if not greeted:
        greeted = True
        async with _respond_lock:
            await _greet(websocket, loop)

    try:
        async for raw in websocket:
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            kind = msg.get("type")

            if kind == "send_text":
                text      = msg.get("text", "").strip()
                image_b64 = msg.get("image")
                if not text and not image_b64:
                    continue
                if _respond_lock.locked():
                    log("[maya] text input dropped (response in progress)")
                    continue
                await _cancel_conv_timer()
                _conv_state = "active"

                if image_b64:
                    log(f"[maya] image upload: {text or '(no caption)'}")
                    label = f"[image] {text}".strip() if text else "[image]"
                    history.append({"role": "user", "content": label})
                    _trim_history()
                    async with _respond_lock:
                        await _respond_vision(text or "Describe this image.", image_b64, websocket.send, loop)
                else:
                    # Vision follow-up: re-invoke vision model only for image-specific questions
                    words = set(text.lower().split())
                    if _last_image_b64 and words & _VISION_FOLLOWUP_WORDS:
                        log(f"[maya] vision follow-up: {text}")
                        history.append({"role": "user", "content": text})
                        _trim_history()
                        async with _respond_lock:
                            await _respond_vision(text, _last_image_b64, websocket.send, loop)
                    else:
                        # Clear stored image — conversation has moved on
                        _last_image_b64 = None
                        # Shutdown / mode switch pre-exec
                        if _detect_shutdown(text):
                            await _exec_shutdown(websocket)
                            return
                        mode_name = _detect_mode_switch(text)
                        if mode_name:
                            result = switch_mode(mode_name)
                            log(f"[maya] mode switch: {result}")
                            text = text + f" [SYSTEM: {result}]"
                        log(f"[maya] text: {text}")
                        text = await _pre_exec_widget_commands(text, websocket)
                        cal_hint = _pre_fetch_calendar(text)
                        if cal_hint:
                            log(f"[maya] calendar pre-fetch injected")
                        history.append({"role": "user", "content": text + cal_hint})
                        _trim_history()
                        async with _respond_lock:
                            await _respond(websocket, loop)
                await _start_conv_timer(websocket)

            elif kind == "widget_state_sync":
                # Frontend sends its actual visibility state on connect
                state = msg.get("state", {})
                for w, v in state.items():
                    if w in _widget_state:
                        _widget_state[w] = bool(v)
                _save_widget_state()

            elif kind == "dev_log_end":
                # Direct end-dev-log from the button — no LLM involved
                _dev_log_mode = False
                _conv_state   = "idle"
                await _cancel_conv_timer()
                if not _dev_log_before_webcam:
                    _widget_state["webcam"] = False
                _save_widget_state()
                log("[maya] dev log ended (button)")

            elif kind == "screen_capture":
                _screen_b64 = msg.get("image")
                if _screen_b64:
                    _screen_event.set()

    except websockets.exceptions.ConnectionClosed:
        log("[maya] client disconnected")
    except Exception as e:
        log(f"[maya] handle_client error: {e}")
    finally:
        _active_ws  = None
        _conv_state = "idle"
        await _cancel_conv_timer()
        # Persist session in background
        if len(history) >= 4:
            with contextlib.suppress(Exception):
                import memory as _mem
                snap       = list(history)
                is_companion = bool(_active_mode_name)
                _mem.save_log(snap, companion=is_companion)
                asyncio.ensure_future(_mem.extract_and_save(snap, companion=is_companion))


async def _respond(websocket, loop: asyncio.AbstractEventLoop) -> None:
    try:
        full = await chat_with_tools(history, websocket.send, websocket)

        # Companion mode: intercept **generate_image**: patterns and trigger real generation
        if full and _active_mode_name:
            cleaned, image_descs = _extract_image_triggers(full, websocket)
            for desc in image_descs:
                log(f"[companion] auto-generate: {desc[:80]}")
                asyncio.ensure_future(_run_comfyui_generation(desc, websocket))
            if image_descs:
                full = cleaned

        if full:
            history.append({"role": "assistant", "content": full})
        await websocket.send(json.dumps({"type": "done"}))
        await websocket.send(json.dumps({"type": "speaking"}))
        await loop.run_in_executor(None, synthesise_and_play, full)
        await websocket.send(json.dumps({"type": "ready"}))
    except websockets.exceptions.ConnectionClosed:
        log("[maya] connection closed during response")
    except Exception as e:
        log(f"[maya] response error: {e}")
        with contextlib.suppress(Exception):
            await websocket.send(json.dumps({"type": "error", "message": str(e)}))
            await websocket.send(json.dumps({"type": "ready"}))

# ── Entry point ───────────────────────────────────────────────────────────────

def kill_existing(port: int) -> None:
    try:
        import psutil
        for conn in psutil.net_connections(kind="tcp"):
            if (conn.laddr.port == port and conn.status == "LISTEN"
                    and conn.pid != os.getpid()):
                with contextlib.suppress(Exception):
                    psutil.Process(conn.pid).kill()
                    log(f"[maya] killed existing process on port {port}")
    except Exception as e:
        log(f"[maya] kill_existing({port}): {e}")


async def main() -> None:
    global _audio_q, _respond_lock, _screen_event

    LOG.write_text("", encoding="utf-8")
    log("[maya] starting...")

    # Always start in default mode regardless of last session
    global _active_mode_name, _active_mode_prompt
    _active_mode_name   = None
    _active_mode_prompt = None
    log("[maya] mode: default")

    # Pull required models in the background if not already available
    async def _ensure_nsfw_model():
        try:
            async with httpx.AsyncClient(timeout=10) as c:
                r = await c.get("http://localhost:11434/api/tags")
                if r.is_success:
                    names = [m.get("name", "") for m in r.json().get("models", [])]
                    for model_name in (OLLAMA_MODEL_NSFW, VISION_MODEL):
                        if any(model_name in n for n in names):
                            log(f"[maya] {model_name} already available")
                            continue
                        log(f"[maya] pulling {model_name} in background...")
                        async with httpx.AsyncClient(timeout=None) as pull_client:
                            async with pull_client.stream(
                                "POST", "http://localhost:11434/api/pull",
                                json={"name": model_name}
                            ) as resp:
                                async for line in resp.aiter_lines():
                                    if not line:
                                        continue
                                    try:
                                        d = json.loads(line)
                                        if d.get("status") == "success":
                                            log(f"[maya] {model_name} ready")
                                            break
                                    except Exception:
                                        pass
                    else:
                        log(f"[maya] {OLLAMA_MODEL_NSFW} already available")
        except Exception as e:
            log(f"[maya] model check error: {e}")

    kill_existing(WS_PORT)
    kill_existing(8766)

    # Initialise asyncio primitives inside the running loop (avoids 3.10+ warning)
    _audio_q      = _queue.Queue()
    _respond_lock = asyncio.Lock()
    _screen_event = asyncio.Event()

    # Restore widget visibility from last session
    global _widget_state
    _widget_state = _load_widget_state()

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, load_models)
    start_audio_stream()

    tasks: list[asyncio.Task] = []

    # Vision server (face recognition + presence detection)
    try:
        import vision as _vision
        tasks.append(asyncio.ensure_future(
            loop.run_in_executor(None, _vision.load_vision_models)
        ))
        vision_server_coro = websockets.serve(_vision.handle_vision_client, "127.0.0.1", 8766)
    except ImportError:
        vision_server_coro = None

    try:
        import spotify as _spotify
        kill_existing(8767)
        tasks.append(asyncio.ensure_future(_spotify.run()))
        log("[maya] spotify server starting...")
    except ImportError:
        pass

    try:
        import weather as _weather
        kill_existing(8768)
        tasks.append(asyncio.ensure_future(_weather.run()))
        log("[maya] weather server starting...")
    except ImportError:
        pass

    try:
        import comfyui as _comfyui_mod
        kill_existing(8770)
        tasks.append(asyncio.ensure_future(_comfyui_mod.run()))
        log("[maya] comfyui widget server starting (port 8770)...")
    except Exception as e:
        log(f"[maya] comfyui server failed to start: {e}")

    try:
        import images as _images_mod
        kill_existing(8769)
        tasks.append(asyncio.ensure_future(_images_mod.run()))
        log("[maya] images server starting...")
    except ImportError:
        pass

    try:
        import system_monitor as _sysmon
        kill_existing(8771)
        tasks.append(asyncio.ensure_future(_sysmon.run()))
        log("[maya] system monitor server starting...")
    except ImportError:
        pass

    try:
        import timer as _timer_mod
        kill_existing(8772)
        _timer_fire_queue: asyncio.Queue = asyncio.Queue()
        _timer_mod._fire_queue = _timer_fire_queue
        tasks.append(asyncio.ensure_future(_timer_mod.run()))
        log("[maya] timer server starting...")
    except ImportError:
        _timer_fire_queue = None

    try:
        import sticky_notes as _sn_mod
        kill_existing(8773)
        tasks.append(asyncio.ensure_future(_sn_mod.run()))
        log("[maya] sticky notes server starting...")
    except ImportError:
        pass

    try:
        import calendar_service as _cal_mod
        kill_existing(8774)
        tasks.append(asyncio.ensure_future(_cal_mod.run()))
        log("[maya] calendar server starting...")
    except ImportError:
        pass

    # Proactive system monitor — phone notification only, no spoken alert
    async def _proactive_monitor_loop():
        while True:
            await asyncio.sleep(30)
            try:
                import system_monitor as _mon
                import notify as _notify
                alert = _mon._alert_message
                if not alert:
                    continue
                _mon._alert_message = None
                log(f"[maya] monitor alert (notification only): {alert}")
                with contextlib.suppress(Exception):
                    await _notify.send("⚠️ MAYA", alert, priority="high")
            except Exception as e:
                log(f"[maya] monitor alert error: {e}")
    tasks.append(asyncio.ensure_future(_proactive_monitor_loop()))

    # Timer completion → speech + notification
    async def _timer_completion_loop():
        if not _timer_fire_queue:
            return
        while True:
            fired = await _timer_fire_queue.get()
            name  = fired.get("name", "timer")
            log(f"[maya] timer fired: {name}")
            try:
                import notify as _notify
                await _notify.send("⏰ Timer", f"{name} timer finished")
            except Exception:
                pass
            if not _active_ws:
                continue
            try:
                async with _respond_lock:
                    with contextlib.suppress(Exception):
                        prompt = (
                            f"Tyler's '{name}' timer just finished. "
                            "Give a brief, natural one-sentence announcement — vary the phrasing. "
                            "No filler openers."
                        )
                        msg = await stream_ollama([{"role": "user", "content": prompt}], _active_ws.send)
                        if msg:
                            history.append({"role": "assistant", "content": msg})
                            await _active_ws.send(json.dumps({"type": "done"}))
                            await _active_ws.send(json.dumps({"type": "speaking"}))
                            await loop.run_in_executor(None, synthesise_and_play, msg)
                            await _active_ws.send(json.dumps({"type": "ready"}))
            except Exception as e:
                log(f"[maya] timer completion error: {e}")
    if _timer_fire_queue is not None:
        tasks.append(asyncio.ensure_future(_timer_completion_loop()))

    # Calendar event reminders — notify 15 minutes before any event
    async def _calendar_reminder_loop():
        from datetime import datetime, timezone, timedelta
        sent_reminders: set = set()
        while True:
            await asyncio.sleep(60)
            try:
                import calendar_service as _cal
                import notify as _notify
                if not _cal._services:
                    continue
                now = datetime.now(timezone.utc)
                for event in _cal._upcoming_events:
                    eid   = event.get("id", "")
                    start = event.get("start", "")
                    if not start or "T" not in start or eid in sent_reminders:
                        continue
                    try:
                        event_dt = datetime.fromisoformat(start)
                        if event_dt.tzinfo is None:
                            event_dt = event_dt.replace(tzinfo=timezone.utc)
                        mins = (event_dt - now).total_seconds() / 60
                        if 14 <= mins <= 16:
                            loc  = f" — {event['location']}" if event.get("location") else ""
                            await _notify.send(
                                f"📅 {event['title']}",
                                f"Starting in 15 minutes{loc}",
                                priority="high",
                            )
                            sent_reminders.add(eid)
                            log(f"[calendar] reminder sent: {event['title']}")
                            # Also speak it if Maya is idle
                            if not _respond_lock.locked() and _active_ws:
                                async with _respond_lock:
                                    with contextlib.suppress(Exception):
                                        prompt = (
                                            f"Tyler has '{event['title']}' starting in 15 minutes"
                                            + (f" at {event['location']}" if event.get("location") else "")
                                            + ". Give a brief one-sentence heads-up. No filler."
                                        )
                                        msg = await stream_ollama([{"role": "user", "content": prompt}], _active_ws.send)
                                        if msg:
                                            history.append({"role": "assistant", "content": msg})
                                            await _active_ws.send(json.dumps({"type": "done"}))
                                            await _active_ws.send(json.dumps({"type": "speaking"}))
                                            await loop.run_in_executor(None, synthesise_and_play, msg)
                                            await _active_ws.send(json.dumps({"type": "ready"}))
                    except Exception:
                        pass
            except Exception as e:
                log(f"[calendar] reminder error: {e}")
    tasks.append(asyncio.ensure_future(_calendar_reminder_loop()))

    tasks.append(asyncio.ensure_future(_ensure_nsfw_model()))
    tasks.append(asyncio.ensure_future(wake_word_loop(loop)))

    servers = [websockets.serve(handle_client, "127.0.0.1", WS_PORT, max_size=20 * 1024 * 1024)]
    if vision_server_coro:
        servers.append(vision_server_coro)

    try:
        async with contextlib.AsyncExitStack() as stack:
            for s in servers:
                await stack.enter_async_context(s)
            log(f"[maya] chat   → ws://127.0.0.1:{WS_PORT}")
            if vision_server_coro:
                log(f"[maya] vision → ws://127.0.0.1:8766")
            log("[maya] say 'Hey Maya ...' to start a conversation")
            await asyncio.Future()
    finally:
        # Clean shutdown — cancel all background tasks
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        log("[maya] shutdown complete")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        pass
    except Exception:
        err = traceback.format_exc()
        with contextlib.suppress(Exception):
            with open(LOG, "a", encoding="utf-8") as f:
                f.write(err)
        print(err, flush=True)
        raise
