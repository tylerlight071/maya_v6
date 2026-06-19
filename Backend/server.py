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
import subprocess
import traceback
import warnings
from pathlib import Path

# ── Force UTF-8 stdout/stderr so Unicode works on Windows regardless of locale ─
import sys, io as _io
if hasattr(sys.stdout, "buffer"):
    sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
if hasattr(sys.stderr, "buffer"):
    sys.stderr = _io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)

# ── Suppress third-party library noise before any imports ─────────────────────
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL",  "3")   # suppress TF C++ logs
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")   # suppress oneDNN info spam
os.environ.setdefault("GLOG_minloglevel",       "3")   # suppress glog (mediapipe/TF)
os.environ.setdefault("WANDB_MODE",   "disabled")
os.environ.setdefault("WANDB_SILENT", "true")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="insightface")
warnings.filterwarnings("ignore", category=UserWarning, module="albumentations")

import httpx
import numpy as np
import sounddevice as sd
import websockets

LOG = Path(__file__).parent / "maya.log"

def log(msg: str) -> None:
    # server.py uses log() instead of print() so messages are also persisted to
    # maya.log for post-session debugging. Other backend modules use print() directly
    # since they don't have access to LOG without creating a circular import.
    line = f"{msg}\n"
    print(line, end="", flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line)

# ── Config ────────────────────────────────────────────────────────────────────

WS_PORT        = 8765
STT_MODEL_SIZE = "base.en"   # single model for all transcription (wake + queries)
OLLAMA_URL      = "http://localhost:11434/api/chat"
OLLAMA_MODEL         = "huihui_ai/dolphin3-abliterated:8b-llama3.1-q4_K_M"
OLLAMA_MODEL_NSFW    = "huihui_ai/dolphin3-abliterated:8b-llama3.1-q4_K_M"
VISION_MODEL         = "huihui_ai/gemma3-abliterated:4b"      # multimodal, abliterated
SAMPLE_RATE     = 16_000
MAX_HISTORY     = 40           # rolling conversation window for LLM context

# VAD tuning
VAD_CHUNK_SECS      = 0.1    # drain audio every 100 ms
VAD_ENERGY_THR      = 0.02   # RMS threshold — raise further if picking up background audio
VAD_SILENCE_CHUNKS  = 30     # × 100 ms = 3 000 ms of silence → end of utterance
VAD_MIN_SPEECH_SECS = 0.4    # ignore very short noise bursts
VAD_MAX_SPEECH_SECS = 20.0   # raised from 10s — allows longer natural sentences
NO_SPEECH_THR       = 0.50   # discard if Whisper thinks this probably isn't speech

# Conversation
CONV_TIMEOUT_SECS = 300.0  # seconds of silence → return to idle (5 minutes)
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
    # ── Memory tools ──────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "save_memory",
            "description": (
                "Save an important fact about Tyler to long-term memory right now, mid-conversation. "
                "Use when Tyler explicitly says 'remember that', 'save that', or 'note that for next time', "
                "or when he shares a lasting preference, constraint, or fact. "
                "Do NOT use for temporary reminders — use add_note for those."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": ["personal", "technical", "preferences", "project"],
                        "description": "Which category this fact belongs to.",
                    },
                    "fact": {
                        "type": "string",
                        "description": "The fact to save as a single concise sentence.",
                    },
                },
                "required": ["category", "fact"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "forget_memory",
            "description": (
                "Remove a specific fact from Tyler's long-term memory. "
                "Use when Tyler says 'forget that', 'that's changed', or 'remove that from memory'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "fact": {
                        "type": "string",
                        "description": "The fact text to remove (approximate match is fine).",
                    },
                },
                "required": ["fact"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_memory",
            "description": (
                "Search through past session logs to find conversations about a specific topic. "
                "Use when Tyler asks 'what did we say about X', 'do you remember when we discussed Y', "
                "or wants to recall something from a specific past session."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search for in past sessions.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_memory_status",
            "description": (
                "Return a summary of what you currently remember about Tyler — "
                "fact counts, recent sessions, in-progress items, active projects, and file sizes. "
                "Use when Tyler asks what you know or remember about him."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_project",
            "description": (
                "Create or update a named project in Tyler's project tracker. "
                "Use proactively when work on a project is discussed, or when Tyler mentions progress. "
                "Creates the project if it doesn't exist yet."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name":   {"type": "string", "description": "Project name."},
                    "status": {"type": "string", "enum": ["active", "paused"], "description": "Optional status update."},
                    "note":   {"type": "string", "description": "Optional note to add about current progress."},
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "close_project",
            "description": "Mark a project as done/complete.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Name of the project to close."},
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_projects",
            "description": "Show all tracked projects (active, paused, and done).",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "regenerate_profile",
            "description": (
                "Force a refresh of the narrative personality profile — the paragraph summary of who Tyler is. "
                "Use when Tyler asks you to update or regenerate your understanding of him."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    # ── Clipboard tools ───────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "get_clipboard",
            "description": "Read the current contents of Tyler's clipboard. Use when he asks 'what's in my clipboard', 'what did I copy', or wants to know or use clipboard contents.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_clipboard",
            "description": "Write text to Tyler's clipboard. Use when he asks to 'copy this', 'put X in my clipboard', or wants to grab text MAYA generated.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The text to copy to clipboard"},
                },
                "required": ["text"],
            },
        },
    },
    # ── Volume tools ──────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "get_volume",
            "description": "Get the current system master volume level (0–100) and mute state. Only call when Tyler explicitly asks what the volume is.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_volume",
            "description": "Set the system master volume to a specific level. Use when Tyler says 'turn it up/down', 'set volume to X', 'louder', 'quieter'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "level": {"type": "integer", "description": "Volume level 0–100"},
                },
                "required": ["level"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mute_audio",
            "description": "Mute or unmute the system audio.",
            "parameters": {
                "type": "object",
                "properties": {
                    "mute": {"type": "boolean", "description": "true = mute, false = unmute"},
                },
                "required": ["mute"],
            },
        },
    },
    # ── Sound tool ────────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "play_sound",
            "description": "Play a sound effect from MAYA's sound library. Available sounds: 'chime', 'alert', 'success', 'error', 'notification'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Sound name without extension (e.g. 'chime', 'alert')"},
                },
                "required": ["name"],
            },
        },
    },
    # ── System info ───────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "get_system_info",
            "description": "Get CPU usage, RAM usage, disk usage, top processes, and GPU info. Use when Tyler asks about performance, what's running, or system status.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    # ── File tools ────────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search for files on Tyler's PC by name. Searches within configured roots (file_search_roots in config.json). Use when Tyler asks to find a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query":     {"type": "string",  "description": "Filename fragment to search for (case-insensitive)"},
                    "extension": {"type": "string",  "description": "Optional file extension filter, e.g. '.pdf', '.py'"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open_file",
            "description": "Open a file with its default application. Only works on files within Tyler's configured search roots.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Full absolute path to the file"},
                },
                "required": ["path"],
            },
        },
    },
    # ── App launcher ──────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "run_app",
            "description": "Launch an application from Tyler's app allowlist (app_allowlist in config.json). Use when Tyler asks to open or launch an app.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "App friendly name as configured in app_allowlist"},
                },
                "required": ["name"],
            },
        },
    },
    # ── Alarm tools ───────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "set_alarm",
            "description": "Set an alarm for a specific time. Opens the alarm widget. Use when Tyler asks to wake him up or be alerted at a specific time.",
            "parameters": {
                "type": "object",
                "properties": {
                    "time":  {"type": "string",  "description": "Time string, e.g. '07:30', '14:00', or natural like '7:30am'"},
                    "label": {"type": "string",  "description": "Short label for the alarm, e.g. 'wake up', 'meeting'"},
                },
                "required": ["time"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_alarm",
            "description": "Cancel an active alarm by its ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "alarm_id": {"type": "string", "description": "Alarm ID returned by set_alarm"},
                },
                "required": ["alarm_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_alarms",
            "description": "List all active alarms.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    # ── Gmail tools ───────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "read_emails",
            "description": "Read Tyler's recent Gmail messages. Shows subject, sender, date, and a short snippet. Use when Tyler asks to check, read, or see his emails.",
            "parameters": {
                "type": "object",
                "properties": {
                    "count":       {"type": "integer", "description": "Number of emails to fetch (default 5, max 20)"},
                    "unread_only": {"type": "boolean", "description": "true = only unread emails (default true)"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "Prepare an email to send. MAYA will confirm the details and wait for Tyler to say 'confirm send' before actually sending. Use when Tyler asks to send, write, or compose an email.",
            "parameters": {
                "type": "object",
                "properties": {
                    "to":      {"type": "string", "description": "Recipient email address"},
                    "subject": {"type": "string", "description": "Email subject line"},
                    "body":    {"type": "string", "description": "Email body text"},
                },
                "required": ["to", "subject", "body"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "authenticate_gmail",
            "description": "Authenticate MAYA with Gmail. Call when Tyler says 'connect Gmail', 'authenticate email', or 'set up Gmail'.",
            "parameters": {"type": "object", "properties": {}},
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
- Use tools cleanly without announcing them.
- Answer "how are you?" warmly and honestly.
- Optimise for natural spoken flow since this is TTS.
- Output ONLY your spoken reply — no meta-commentary, no self-correction, no formatting notes, no "---", no second drafts, no commentary about your own response. The first thing you write is what gets spoken aloud.

-- FACTUAL HONESTY — CRITICAL ---------------------------------------------------

NEVER invent, assume, or extrapolate facts. If you don't know something, say so.

Specifically forbidden:
- Do NOT claim Tyler has a meeting, event, or deadline unless it appears in the calendar context below.
- Do NOT claim you've discussed something with Tyler unless it appears in this conversation or the memory block below.
- Do NOT infer or guess at Tyler's schedule, commitments, or projects beyond what is explicitly given.
- Do NOT double down when Tyler corrects you — immediately acknowledge the mistake and drop it.

If you're uncertain about a fact: say "I'm not sure" or ask Tyler directly. Never present a guess as a fact.
The calendar context below is the COMPLETE picture of Tyler's schedule. There are no other events you know about.

CURRENT DATETIME: {datetime}
{dates_ctx}{emotion_context}{webcam_context}{spotify_context}{weather_context}{images_context}{comfy_ctx}{monitor_context}{widget_context}
{notes_context}{memory_block}
"""

# ── System prompt context builders ───────────────────────────────────────────
# Each function returns a single context string ready for template injection.

def _ctx_spotify() -> str:
    try:
        import spotify as _sp
        t = _sp._current_track
        if t and t.get("track"):
            state = "playing" if t.get("is_playing") else "paused"
            album = f" ({t.get('album')})" if t.get("album") else ""
            return f"SPOTIFY: Currently {state} — \"{t.get('track')}\" by {t.get('artist')}{album}.\n"
        return "SPOTIFY: Nothing currently playing.\n"
    except Exception:
        return ""


def _ctx_weather() -> str:
    try:
        import weather as _wx
        w   = _wx._current_weather
        loc = _wx._location or {}
        if w:
            loc_note = " [location unknown — say 'set weather location to X' to fix]" \
                       if loc.get("fallback") else ""
            return (
                f"WEATHER: {w.get('description')} {w.get('temp')}°C "
                f"(feels like {w.get('feels_like')}°C), "
                f"humidity {w.get('humidity')}%, "
                f"wind {w.get('wind_speed')} mph {w.get('wind_dir')} "
                f"— {w.get('city')}, {w.get('country')}{loc_note}.\n"
            )
    except Exception:
        pass
    return ""


def _ctx_monitor() -> str:
    try:
        import system_monitor as _mon
        s = _mon._current_stats
        if s:
            vram_pct  = round(s["vram_used"]  / s["vram_total"]  * 100) if s.get("vram_total")  else 0
            vram2_pct = round(s["vram2_used"]  / s["vram2_total"] * 100) if s.get("vram2_total") else 0
            gpu2_part = (
                f" · GPU2 {s['gpu2']}% {s['vram2_used']}/{s['vram2_total']} GB VRAM ({vram2_pct}%) {s['gpu2_temp']}°C"
                if s.get("vram2_total") else ""
            )
            return (
                f"SYSTEM: CPU {s['cpu']}% · "
                f"RAM {s['ram_used']}/{s['ram_total']} GB · "
                f"GPU1 {s['gpu']}% {s['vram_used']}/{s['vram_total']} GB VRAM ({vram_pct}%) {s['gpu_temp']}°C"
                f"{gpu2_part}\n"
            )
    except Exception:
        pass
    return ""


def _ctx_notes() -> str:
    try:
        import sticky_notes as _sn
        return _sn.format_for_prompt() + "\n" if _sn._notes else ""
    except Exception:
        return ""


def _ctx_images() -> str:
    try:
        import images as _img
        cat = _img._catalog
        if cat:
            if _widget_state.get("images"):
                recent = ", ".join(c["filename"] for c in cat[:4])
                return f"IMAGES: {len(cat)} photos in gallery. Most recent: {recent}.\n"
            return f"IMAGES: {len(cat)} photos in gallery (panel closed).\n"
        return "IMAGES: Gallery is empty or folder not found.\n"
    except Exception:
        return ""


def _ctx_comfy() -> str:
    if not _comfyui_workflow:
        return ""
    return (
        "IMAGE GENERATION: ComfyUI workflow loaded. "
        "Use generate_image(description) to create images — describe the scene/subject/setting in detail. "
        "ComfyUI must be running in Stability Matrix.\n"
    )


def _import_time():
    import time as _t
    return _t


def _is_tyler_present(max_absent_secs: float = 90.0) -> bool:
    """True if Tyler's face was seen recently, or if vision is unavailable (fail open)."""
    try:
        import vision as _vis
        return (_import_time().monotonic() - _vis._last_seen_at) < max_absent_secs
    except Exception:
        return True   # vision unavailable — assume present, always respond


def _ctx_emotion() -> str:
    try:
        import vision as _vis
        em       = _vis._last_emotion
        dominant = em.get("dominant", "neutral")
        conf     = em.get("confidence", 0.0)
        display  = _vis._EMOTION_DISPLAY.get(dominant, dominant)
        if dominant != "neutral" and conf >= 0.40:
            return f"VISUAL READ: Tyler appears {display} ({int(conf * 100)}% confidence from camera).\n"
        return "VISUAL READ: Tyler appears calm/neutral.\n"
    except Exception:
        return ""


def _ctx_widgets() -> str:
    map_ctx        = f"MAP: {'Full-screen tactical map is active.' if _map_visible else 'Map is closed.'}\n"
    open_widgets   = [k for k, v in _widget_state.items() if v     and k != "map"]
    closed_widgets = [k for k, v in _widget_state.items() if not v and k != "map"]
    return (
        f"UI PANELS — open: {', '.join(open_widgets) if open_widgets else 'none'}  |  "
        f"closed: {', '.join(closed_widgets) if closed_widgets else 'none'}.\n"
        + map_ctx
        + "If Tyler says a panel isn't open or isn't closed, trust what he says over this state "
          "and call show_widget to correct it.\n"
    )


def _ctx_dates() -> str:
    from datetime import datetime, timedelta
    now       = datetime.now()
    today_iso = now.strftime("%Y-%m-%d")
    return (
        f"TODAY: {today_iso} ({now.strftime('%A')})\n"
        + "\n".join(
            f"{(now + timedelta(days=i)).strftime('%A')}: {(now + timedelta(days=i)).strftime('%Y-%m-%d')}"
            for i in range(1, 8)
        ) + "\n"
    )


_PROMPT_STATIC_CACHE: "tuple[float, str] | None" = None
_PROMPT_STATIC_TTL = 8.0   # seconds — rebuild static context at most every 8s

_webcam_visual_cache: "tuple[float, str]" = (0.0, "")
_WEBCAM_VISUAL_TTL = 30.0  # seconds — refresh vision-model description every 30s

async def _ctx_webcam_visual() -> str:
    """Ask gemma3 to briefly describe the latest webcam frame. Cached 30s."""
    global _webcam_visual_cache
    try:
        import vision as _vis
        b64 = _vis._last_frame_b64
        if not b64:
            return ""
        now = _import_time().monotonic()
        if now - _webcam_visual_cache[0] < _WEBCAM_VISUAL_TTL:
            return _webcam_visual_cache[1]
        async with httpx.AsyncClient(timeout=12) as client:
            r = await client.post(OLLAMA_URL, json={
                "model":    VISION_MODEL,
                "messages": [{
                    "role":    "user",
                    "content": (
                        "Describe what you see in one sentence. "
                        "Focus on the person's expression, posture, and what they appear to be doing."
                    ),
                    "images":  [b64],
                }],
                "stream":     False,
                "keep_alive": "10m",
            })
        if r.is_success:
            desc   = r.json().get("message", {}).get("content", "").strip()
            result = f"WEBCAM VIEW: {desc}\n" if desc else ""
            _webcam_visual_cache = (now, result)
            return result
    except Exception:
        pass
    return ""


async def build_system_prompt(messages: list[dict] = None) -> str:
    import memory as _mem
    from datetime import datetime
    import time as _time

    global _PROMPT_STATIC_CACHE
    now = datetime.now()
    dt  = now.strftime("%A, %d %B %Y — %H:%M")

    # Cache the expensive-to-build static context sections
    now_ts = _time.monotonic()
    if _PROMPT_STATIC_CACHE and (now_ts - _PROMPT_STATIC_CACHE[0]) < _PROMPT_STATIC_TTL:
        static_block = _PROMPT_STATIC_CACHE[1]
    else:
        dates_ctx      = _ctx_dates()
        spotify_context = _ctx_spotify()
        weather_context = _ctx_weather()
        monitor_context = _ctx_monitor()
        notes_context   = _ctx_notes()
        images_context  = _ctx_images()
        comfy_ctx       = _ctx_comfy()
        emotion_context = _ctx_emotion()
        widget_context  = _ctx_widgets()
        static_block = (dates_ctx, spotify_context, weather_context, monitor_context,
                        notes_context, images_context, comfy_ctx, emotion_context, widget_context)
        _PROMPT_STATIC_CACHE = (now_ts, static_block)

    (dates_ctx, spotify_context, weather_context, monitor_context,
     notes_context, images_context, comfy_ctx, emotion_context, widget_context) = static_block

    ctx   = await _mem.format_context(companion=bool(_active_mode_name), messages=messages)
    block = ctx + "\n" if ctx else ""

    # Async webcam visual context — not in static cache (30s TTL of its own)
    webcam_context = await _ctx_webcam_visual()

    # Custom mode — if the prompt has {datetime} treat it as a full template
    if _active_mode_prompt:
        if "{datetime}" in _active_mode_prompt:
            return _active_mode_prompt.format(
                datetime=dt,
                dates_ctx=dates_ctx,
                emotion_context=emotion_context,
                webcam_context=webcam_context,
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
            f"{dates_ctx}{emotion_context}{webcam_context}{spotify_context}{weather_context}"
            f"{monitor_context}{widget_context}{notes_context}{block}"
        )
        return f"{_active_mode_prompt}\n{dynamic}"

    return _SYSTEM_PROMPT_TEMPLATE.format(
        datetime=dt,
        dates_ctx=dates_ctx,
        emotion_context=emotion_context,
        webcam_context=webcam_context,
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

whisper_model = None   # loaded in load_models() — model size from STT_MODEL_SIZE constant
tts_model     = None   # F5-TTS loaded in load_models() (None if load fails → silent mode)

_audio_stream  = None
_capture_rate  = SAMPLE_RATE
_audio_q: "_queue.Queue[np.ndarray]" = None  # set in main()

_active_ws          = None
_respond_lock: asyncio.Lock  = None   # set in main()
_screen_event: asyncio.Event = None   # set in main()
_screen_b64: "str | None"    = None

# Conversation state machine
# "idle"   — no recent conversation (history cleared after timeout); still responds
# "active" — actively in conversation
_conv_state    = "idle"
_conv_timer    = None  # asyncio.Task — silence-based conversation timeout

_last_user_spoke_at: float = 0.0   # monotonic time of last user utterance (for proactive engagement)

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
    "news":           False,
}

def _load_widget_state() -> dict:
    """Load persisted widget visibility from disk, merging with defaults."""
    try:
        if WIDGET_STATE_FILE.exists():
            saved = json.loads(WIDGET_STATE_FILE.read_text(encoding="utf-8"))
            # Merge saved into defaults so new widgets always get a value
            merged = {**_WIDGET_DEFAULTS, **{k: bool(v) for k, v in saved.items() if k in _WIDGET_DEFAULTS}}
            active = [k for k, v in merged.items() if v]
            log(f"[maya] Widgets {' '.join(active)}")
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
    path = MODES_DIR / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Mode not found: {name}")
    return path.read_text(encoding="utf-8").strip()


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

def _gpu_summary() -> str:
    """Return a compact one-line GPU summary, e.g. 'cuda:0 RTX 5070 Ti 17GB · cuda:1 RTX 4060 8GB'"""
    try:
        import torch
        if not torch.cuda.is_available():
            return "CPU only"
        parts = []
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            vram = round(torch.cuda.get_device_properties(i).total_memory / 1e9, 1)
            parts.append(f"cuda:{i} {name} ({vram} GB)")
        return "  ·  ".join(parts)
    except Exception:
        return "unknown"


_CONFIG_FILE = Path(__file__).parent / "models/config.json"

def _load_config() -> dict:
    """Load models/config.json; returns empty dict on any error."""
    try:
        return json.loads(_CONFIG_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _resolve_gpu(config_key: str, default: int = 0) -> str:
    """Read a gpu index from config.json and return a cuda device string.
    Falls back to cuda:0 if the requested GPU doesn't exist."""
    try:
        import torch
        cfg_path = Path(__file__).parent / "models/config.json"
        if cfg_path.exists():
            cfg     = json.loads(cfg_path.read_text(encoding="utf-8"))
            gpu_idx = cfg.get(config_key, default)
            if isinstance(gpu_idx, int) and torch.cuda.is_available():
                count = torch.cuda.device_count()
                if gpu_idx < count:
                    return f"cuda:{gpu_idx}"
                log(f"[maya] {config_key}={gpu_idx} not available "
                    f"(only {count} GPU(s) detected) — falling back to cuda:0")
    except Exception as e:
        log(f"[maya] _resolve_gpu({config_key}) error: {e}")
    return "cuda"   # cuda:0


def load_models() -> None:
    global whisper_model, tts_model

    import torch
    from faster_whisper import WhisperModel

    # STT and TTS can be on different GPUs — STT is lightweight (fine on 4060),
    # TTS (F5-TTS diffusion model) needs the fast GPU for acceptable latency.
    stt_device = _resolve_gpu("stt_gpu", default=0)
    tts_device = _resolve_gpu("tts_gpu", default=0)

    # faster-whisper uses device_index rather than "cuda:N"
    wh_index = int(stt_device.split(":")[1]) if stt_device.startswith("cuda:") else 0

    try:
        whisper_model = WhisperModel(
            STT_MODEL_SIZE, device="cuda",
            device_index=wh_index, compute_type="float16",
        )
        log(f"[maya] STT    faster-whisper {STT_MODEL_SIZE} → {stt_device}")
    except Exception as e:
        try:
            whisper_model = WhisperModel(STT_MODEL_SIZE, device="cpu", compute_type="int8")
            log(f"[maya] STT    faster-whisper {STT_MODEL_SIZE} → CPU (fallback)")
        except Exception as e2:
            log(f"[maya] STT    disabled ({e2})")
            whisper_model = None
    try:
        import sys, io

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
        device = tts_device if torch.cuda.is_available() else "cpu"
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            tts_model = F5TTS(device=device)
        log(f"[maya] TTS    F5-TTS → {device}")

    except Exception as e:
        log(f"[maya] TTS    disabled ({e})")
        tts_model = None

# ── Audio capture (always-on, thread-safe via queue) ─────────────────────────

def _audio_callback(indata, frames, time, status) -> None:
    # Called from sounddevice's C callback thread.
    # queue.Queue.put_nowait is the only safe asyncio-free operation here.
    if status:
        return  # buffer overrun / underrun — skip invalid frame
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
        _audio_out_name = out_info['name']
    except Exception:
        _audio_out_name = "unknown"

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
        log(f"[maya] Audio  out: {_audio_out_name}  ·  mic: {info['name']} @ {_capture_rate} Hz")
        return
    except Exception as e:
        log(f"[maya] default mic failed ({e!r}) — scanning devices...")

    for i, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] < 1:
            continue
        try:
            _audio_stream, info, _capture_rate = _try(i)
            log(f"[maya] Audio  out: {_audio_out_name}  ·  mic: {info['name']} [{i}] @ {_capture_rate} Hz")
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
    "news":           {"news", "headlines", "feed"},
}
_OPEN_VERBS  = {"open", "show", "display", "reveal", "bring", "pull"}
_CLOSE_VERBS = {"close", "hide", "dismiss", "shut", "minimise", "minimize", "stop", "exit", "quit"}


_ALL_PANELS = ["chat", "spotify", "webcam", "weather", "images", "imagegen",
               "system_monitor", "timer", "sticky_notes", "calendar", "video", "news"]
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
        if _conv_state != "active":
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
                    "Tyler has gone quiet for a while — the conversation context is resetting. "
                    + (f"Recent exchange:\n{ctx}\n\n" if ctx else "")
                    + "Give a brief, natural one-sentence remark that makes clear you're still here "
                    "and always listening — you haven't gone anywhere. Vary the phrasing each time. "
                    "Don't say goodbye. Don't offer further help. No filler openers."
                )
                goodbye = await stream_ollama(
                    [{"role": "user", "content": prompt}],
                    current_ws.send,
                )
                if not goodbye:
                    goodbye = "Still right here whenever you need me."
                await current_ws.send(json.dumps({"type": "done"}))
                await current_ws.send(json.dumps({"type": "speaking"}))
                await loop.run_in_executor(None, synthesise_and_play, goodbye)
                await current_ws.send(json.dumps({"type": "ready"}))
                history.append({"role": "assistant", "content": goodbye})

    _conv_timer = asyncio.create_task(_expire())


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

    stream_check_counter = 0
    while True:
        await asyncio.sleep(VAD_CHUNK_SECS)

        # Periodically check if the audio stream is still alive (every ~5 seconds)
        stream_check_counter += 1
        if stream_check_counter >= int(5.0 / VAD_CHUNK_SECS):
            stream_check_counter = 0
            if _audio_stream and not getattr(_audio_stream, 'active', True):
                log("[maya] audio stream died — attempting to reconnect microphone...")
                try:
                    start_audio_stream()
                    log("[maya] microphone reconnected")
                except Exception as e:
                    log(f"[maya] microphone reconnect failed: {e}")

        # Drain audio queue into local accumulator (thread-safe)
        new: list[np.ndarray] = []
        try:
            while True:
                new.append(_audio_q.get_nowait())
        except _queue.Empty:
            pass
        if new:
            parts = ([accum] if len(accum) else []) + new
            accum = np.concatenate(parts)

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
                            asyncio.create_task(
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

        # For browser open: check if the query contains a URL/domain and navigate to it
        if widget == "browser" and visible:
            url_match = re.search(
                r'\b((?:https?://)?(?:[\w-]+\.)+(?:com|org|net|io|co\.uk|uk|tv|xxx|gov|edu|me|cc|app|ai)(?:/[^\s]*)?)\b',
                query, re.IGNORECASE,
            )
            if url_match:
                nav_url = url_match.group(1)
                if not nav_url.startswith("http"):
                    nav_url = "https://" + nav_url
                with contextlib.suppress(Exception):
                    await ws.send(json.dumps({
                        "type": "action", "action": "browse",
                        "url": nav_url, "title": "",
                    }))
                _widget_state["browser"] = True
                log(f"[maya] pre-exec: browser → navigate {nav_url}")
                hints.append(f"browser opened and navigated to {nav_url}")
                continue

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
    """Route a transcribed utterance — always-on, no wake word required."""
    global _conv_state, _dev_log_mode, _last_user_spoke_at

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
        return

    # Drop if another response is already running
    if _respond_lock.locked():
        return

    ws = _active_ws
    if not ws:
        return

    # ── Presence gate — ignore audio when face hasn't been seen recently ───
    # Fails open: if vision is unavailable, always respond.
    if not _is_tyler_present(max_absent_secs=90.0):
        return

    await _cancel_conv_timer()
    query = transcript.strip()
    if not query:
        return

    _last_user_spoke_at = _import_time().monotonic()
    log(f"[maya] {'query' if _conv_state == 'active' else 'new query'}: {query}")

    # ── Closing phrase → reply then reset to idle (history clears on timeout) ─
    if _is_closing(query):
        log("[maya] closing phrase — resetting conversation")
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

    # ── Normal query — always respond ─────────────────────────────────────
    _conv_state = "active"
    query = await _pre_exec_widget_commands(query, ws)
    if _detect_shutdown(query):
        await _exec_shutdown(ws)
        return

    async with _respond_lock:
        try:
            mode_name = _detect_mode_switch(query)
            if mode_name:
                result = switch_mode(mode_name)
                log(f"[maya] mode switch (voice): {result}")
                # Replace the switch command with a simple intro so the model
                # becomes the persona rather than describing what the mode is
                if mode_name == "default":
                    query = "[Returning to default mode. Greet Tyler briefly as normal MAYA.]"
                else:
                    query = "[Entering new persona. Greet Tyler with ONE short in-character line — no explanation of modes.]"
                display = result
            else:
                display = query
            await ws.send(json.dumps({"type": "wake_detected"}))
            await ws.send(json.dumps({"type": "transcript", "text": display}))
            cal_hint = "" if mode_name else _pre_fetch_calendar(query)
            if cal_hint:
                log("[maya] calendar pre-fetch injected (voice)")
            history.append({"role": "user", "content": query + cal_hint})
            _trim_history()
            await _respond(ws, loop)
        except Exception as e:
            log(f"[maya] response error: {e}")

    await _start_conv_timer(ws)

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


async def _unload_text_model() -> None:
    """Unload the chat model from VRAM to free space for the vision model."""
    with contextlib.suppress(Exception):
        async with httpx.AsyncClient(timeout=5) as c:
            await c.post(OLLAMA_URL, json={"model": _active_model(), "messages": [], "keep_alive": 0})
            log(f"[maya] unloaded {_active_model()} before vision call")


async def _respond_vision(question: str, image_b64: str, send, loop: asyncio.AbstractEventLoop) -> None:
    """Stream a vision-model response for an uploaded image, then speak it."""
    global _last_image_b64
    _last_image_b64 = image_b64
    await _unload_text_model()
    try:
        full = ""
        # In companion mode pass the system prompt so the abliterated model responds in character
        msgs = []
        if _active_mode_prompt:
            msgs.append({"role": "system", "content": await build_system_prompt()})
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


# ── Pending email (verbal confirmation) ──────────────────────────────────────
_pending_email: "dict | None" = None


# ── Tool domain handlers ──────────────────────────────────────────────────────

async def _tool_system(name: str, params: dict) -> str:
    """Clipboard, volume, sound, and system info tools."""

    # ── Clipboard ─────────────────────────────────────────────────────────────
    if name == "get_clipboard":
        try:
            r = subprocess.run(
                ["powershell", "-Command", "Get-Clipboard"],
                capture_output=True, text=True, timeout=5,
            )
            content = r.stdout.strip()
            return f"Clipboard contents: {content}" if content else "Clipboard is empty."
        except Exception as e:
            return f"Clipboard read failed: {e}"

    if name == "set_clipboard":
        text = params.get("text", "")
        escaped = text.replace("'", "''")
        try:
            subprocess.run(
                ["powershell", "-Command", f"Set-Clipboard -Value '{escaped}'"],
                capture_output=True, timeout=5,
            )
            return "Copied to clipboard."
        except Exception as e:
            return f"Clipboard write failed: {e}"

    # ── Volume ────────────────────────────────────────────────────────────────
    if name in ("get_volume", "set_volume", "mute_audio"):
        try:
            from pycaw.pycaw import AudioUtilities
            # Modern pycaw (2025+): AudioDevice.EndpointVolume gives IAudioEndpointVolume directly
            vol = AudioUtilities.GetSpeakers().EndpointVolume

            if name == "get_volume":
                level = round(vol.GetMasterVolumeLevelScalar() * 100)
                muted = bool(vol.GetMute())
                return f"Volume: {level}%{'  (muted)' if muted else ''}."

            if name == "set_volume":
                level = max(0, min(100, int(params.get("level", 50))))
                vol.SetMasterVolumeLevelScalar(level / 100, None)
                return f"Volume set to {level}%."

            if name == "mute_audio":
                mute = bool(params.get("mute", True))
                vol.SetMute(int(mute), None)
                return "Audio muted." if mute else "Audio unmuted."

        except ImportError:
            return "pycaw not installed — run: pip install pycaw comtypes"
        except Exception as e:
            return f"Volume control failed: {e}"

    # ── Sound ─────────────────────────────────────────────────────────────────
    if name == "play_sound":
        sound_name = params.get("name", "chime").lower().strip()
        sounds_dir = Path(__file__).parent / "sounds"
        # Try .wav first, then .mp3
        path = sounds_dir / f"{sound_name}.wav"
        if not path.exists():
            path = sounds_dir / f"{sound_name}.mp3"
        if not path.exists():
            available = [p.stem for p in sounds_dir.glob("*") if p.suffix in (".wav", ".mp3")]
            return f"Sound '{sound_name}' not found. Available: {', '.join(available) or 'none'}."
        try:
            import soundfile as sf
            import sounddevice as sd_play
            data, sr = sf.read(str(path))
            sd_play.play(data, sr)
            return f"Playing '{sound_name}'."
        except Exception as e:
            return f"Playback failed: {e}"

    # ── System info ───────────────────────────────────────────────────────────
    if name == "get_system_info":
        try:
            import psutil
            cpu  = psutil.cpu_percent(interval=0.5)
            ram  = psutil.virtual_memory()
            disk = psutil.disk_usage("/")
            procs = sorted(psutil.process_iter(["pid","name","cpu_percent"]),
                           key=lambda p: p.info["cpu_percent"] or 0, reverse=True)[:3]
            top  = ", ".join(f"{p.info['name']} ({p.info['cpu_percent']:.0f}%)" for p in procs)
            lines = [
                f"CPU: {cpu}%",
                f"RAM: {ram.percent}%  ({ram.used // (1024**3):.1f} / {ram.total // (1024**3):.1f} GB)",
                f"Disk: {disk.percent}%  ({disk.used // (1024**3):.0f} / {disk.total // (1024**3):.0f} GB)",
                f"Top processes: {top}",
            ]
            try:
                r = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=3,
                )
                if r.returncode == 0:
                    util, used, total = r.stdout.strip().split(", ")
                    lines.append(f"GPU: {util}% util  {used}/{total} MB VRAM")
            except Exception:
                pass
            return "\n".join(lines)
        except ImportError:
            return "psutil not installed — run: pip install psutil"
        except Exception as e:
            return f"System info failed: {e}"

    return f"Unknown system tool: {name}"


async def _tool_files(name: str, params: dict) -> str:
    """File search, open, and app launcher tools."""
    cfg = _load_config()

    if name == "search_files":
        query     = params.get("query", "").strip().lower()
        extension = (params.get("extension") or "").strip().lower()
        roots     = cfg.get("file_search_roots", [])
        if not roots:
            return "No file_search_roots configured in config.json."
        if not query:
            return "No search query provided."
        results: list[str] = []
        for root in roots:
            rp = Path(root)
            if not rp.exists():
                continue
            try:
                for p in rp.rglob("*"):
                    if query not in p.name.lower():
                        continue
                    if extension and p.suffix.lower() != extension:
                        continue
                    results.append(str(p))
                    if len(results) >= 20:
                        break
            except PermissionError:
                pass
            if len(results) >= 20:
                break
        if not results:
            return f"No files found matching '{query}'."
        return f"Found {len(results)} file(s):\n" + "\n".join(results)

    if name == "open_file":
        path_str = params.get("path", "").strip()
        if not path_str:
            return "No file path provided."
        roots = cfg.get("file_search_roots", [])
        p = Path(path_str).resolve()
        if roots:
            allowed = any(str(p).startswith(str(Path(r).resolve())) for r in roots)
            if not allowed:
                return f"Access denied — '{path_str}' is outside the configured file search roots."
        if not p.exists():
            return f"File not found: {path_str}"
        try:
            os.startfile(str(p))
            return f"Opened {p.name}."
        except Exception as e:
            return f"Failed to open file: {e}"

    if name == "run_app":
        app_name  = params.get("name", "").strip().lower()
        allowlist = cfg.get("app_allowlist", {})
        if not allowlist:
            return "No app_allowlist configured in config.json."
        if app_name not in {k.lower() for k in allowlist}:
            available = ", ".join(allowlist.keys())
            return f"'{app_name}' is not in the app allowlist. Available: {available}."
        exe = next(v for k, v in allowlist.items() if k.lower() == app_name)
        try:
            subprocess.Popen([exe])
            return f"Launched {app_name}."
        except Exception as e:
            return f"Failed to launch '{app_name}': {e}"

    return f"Unknown file tool: {name}"


async def _tool_alarm(name: str, params: dict, websocket) -> str:
    """Alarm set/cancel/list tools — delegates to alarm module."""
    try:
        import alarm as _alarm
    except ImportError:
        return "Alarm module not loaded."
    if name == "set_alarm":
        time_str = params.get("time", "").strip()
        label    = params.get("label", "Alarm").strip()
        result, alarm_id = _alarm.set_alarm(time_str, label)
        if alarm_id:
            await websocket.send(json.dumps({
                "type": "action", "action": "show_widget",
                "widget": "alarm", "visible": True,
            }))
        return result
    if name == "cancel_alarm":
        return _alarm.cancel_alarm(params.get("alarm_id", ""))
    if name == "list_alarms":
        return _alarm.list_alarms()
    return f"Unknown alarm tool: {name}"


async def _tool_gmail(name: str, params: dict) -> str:
    """Gmail read/send/auth tools."""
    global _pending_email
    try:
        import gmail as _gmail
    except ImportError:
        return "Gmail module not loaded."

    if name == "authenticate_gmail":
        try:
            url = _gmail.auth_url()
            return f"Open this URL to connect Gmail:\n{url}\nThen say 'gmail callback CODE' with the code from the redirect."
        except Exception as e:
            return f"Gmail auth failed: {e}"

    if name == "read_emails":
        count       = min(int(params.get("count", 5)), 20)
        unread_only = params.get("unread_only", True)
        try:
            emails = await asyncio.get_running_loop().run_in_executor(
                None, lambda: _gmail.fetch_emails(count, unread_only)
            )
        except Exception as e:
            return f"Failed to fetch emails: {e}"
        if not emails:
            return "No emails found." if not unread_only else "No unread emails."
        lines = []
        for i, e in enumerate(emails, 1):
            lines.append(f"{i}. From: {e['from']}  |  {e['date']}")
            lines.append(f"   Subject: {e['subject']}")
            lines.append(f"   {e['snippet']}")
        return "\n".join(lines)

    if name == "send_email":
        to      = params.get("to", "").strip()
        subject = params.get("subject", "").strip()
        body    = params.get("body", "").strip()
        if not to or not subject:
            return "Missing recipient or subject."
        _pending_email = {"to": to, "subject": subject, "body": body}
        return (
            f"Ready to send to {to} — subject: \"{subject}\". "
            "Say 'confirm send' to send, or 'cancel send' to discard."
        )

    return f"Unknown Gmail tool: {name}"


async def _tool_spotify(name: str, params: dict, _ws) -> str:
    try:
        import spotify as _sp
        cfg = _sp.load_config()
        if not cfg:
            return "Spotify not configured."
        cid, cs = cfg["client_id"], cfg["client_secret"]
        if name == "spotify_control":
            await _sp.control(params.get("action", ""), cid, cs)
            return f"Spotify: {params.get('action', '')}."
        if name == "spotify_play_track":
            results = await _sp.search_tracks(params.get("query", ""), cid, cs)
            if not results:
                return "No results found on Spotify."
            ok = await _sp.play_uri(results[0]["uri"], cid, cs)
            if ok:
                return f"Playing \"{results[0]['track']}\" by {results[0]['artist']}."
            return f"Found \"{results[0]['track']}\" but couldn't play it — is Spotify open on your PC?"
    except Exception as e:
        return f"Spotify failed: {e}"
    return "Spotify not configured."


async def _tool_timer(name: str, params: dict, websocket) -> str:
    try:
        import timer as _tim
    except ImportError:
        return "Timer module not loaded."
    if name == "set_timer":
        tname   = params.get("name", "timer").strip().lower()
        seconds = int(params.get("seconds", 60))
        _tim.create(tname, seconds)
        _widget_state["timer"] = True
        await websocket.send(json.dumps({"type": "action", "action": "show_widget", "widget": "timer", "visible": True}))
        m, s = divmod(seconds, 60)
        return f"Timer set: {tname} — {m:02d}:{s:02d}."
    if name == "cancel_timer":  return _tim.cancel(params.get("name", ""))
    if name == "pause_timer":   return _tim.pause(params.get("name", ""))
    if name == "resume_timer":  return _tim.resume(params.get("name", ""))
    if name == "list_timers":   return _tim.list_all()
    return f"Unknown timer tool: {name}"


async def _tool_notes(name: str, params: dict, websocket) -> str:
    try:
        import sticky_notes as _sn
    except ImportError:
        return "Notes module not loaded."
    _show = lambda: asyncio.create_task(_sn.broadcast_notes()) or asyncio.create_task(
        websocket.send(json.dumps({"type": "action", "action": "show_widget", "widget": "sticky_notes", "visible": True}))
    )
    if name == "add_note":
        text    = params.get("text", "").strip()
        note_id = _sn.add(text)
        _widget_state["sticky_notes"] = True
        _show()
        return f"Noted (id: {note_id[:8]})."
    if name == "update_note":
        ok = _sn.update(params.get("id", ""), params.get("text", ""))
        if ok:
            _widget_state["sticky_notes"] = True
            _show()
        return "Note updated." if ok else "Note not found."
    if name == "delete_note":
        ok = _sn.delete(params.get("id", ""))
        if ok:
            asyncio.create_task(_sn.broadcast_notes())
        return "Note deleted." if ok else "Note not found."
    if name == "list_notes":
        notes = _sn.get_all()
        if not notes:
            return "No notes saved."
        return "\n".join(f"[{n['id'][:8]}] {n['text']}" for n in notes)
    return f"Unknown notes tool: {name}"


async def _tool_calendar(name: str, params: dict, websocket) -> str:
    try:
        import calendar_service as _cal
    except ImportError:
        return "Calendar module not loaded."
    _open_cal = lambda: asyncio.create_task(
        websocket.send(json.dumps({"type": "action", "action": "show_widget", "widget": "calendar", "visible": True}))
    )
    if name == "get_calendar_events":
        date   = params.get("date", "")
        events = _cal.get_events_for_date(date) if date else _cal.get_today_events()
        _widget_state["calendar"] = True
        _open_cal()
        if not events:
            return f"No events on {'that date' if date else 'today'}."
        return "\n".join(f"- {e['title']} at {e['start']} (id:{e['id'][:12]})" for e in events)
    if name == "create_calendar_event":
        result = _cal.create_event(params.get("title",""), params.get("start",""), params.get("end",""), params.get("description",""))
        _widget_state["calendar"] = True
        _open_cal()
        return f"Event created (id: {result[:12] if 'Failed' not in result else result})."
    if name == "update_calendar_event":
        return _cal.update_event(params.get("event_id",""), params.get("title",""), params.get("start",""), params.get("end",""), params.get("description",""))
    if name == "delete_calendar_event":
        return _cal.delete_event(params.get("event_id", ""))
    if name == "authenticate_calendar":
        loop   = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, _cal.authenticate)
        if result == "done":
            asyncio.create_task(_cal._push_update())
            return "Calendar authenticated and connected."
        return f"Calendar auth failed: {result}"
    return f"Unknown calendar tool: {name}"


async def _tool_memory(name: str, params: dict, _ws) -> str:
    import memory as _mem
    if name == "save_memory":       return await _mem.save_memory_fact(params.get("category","general"), params.get("fact",""))
    if name == "forget_memory":     return _mem.forget_memory_fact(params.get("fact",""))
    if name == "search_memory":     return await _mem.search_session_logs(params.get("query",""))
    if name == "get_memory_status": return _mem.get_memory_status()
    if name == "update_project":    return _mem.update_project(params.get("name",""), params.get("status"), params.get("note"))
    if name == "close_project":     return _mem.close_project(params.get("name",""))
    if name == "list_projects":     return _mem.list_projects()
    if name == "regenerate_profile":return await _mem.regenerate_profile_tool()
    return f"Unknown memory tool: {name}"


async def execute_tool(name: str, params: dict, websocket) -> str:
    global _screen_b64, _conv_state, _dev_log_mode, _dev_log_before_webcam, _map_visible

    # ── Dispatch to domain handlers ───────────────────────────────────────────
    if name in ("spotify_control", "spotify_play_track"):
        return await _tool_spotify(name, params, websocket)
    if name in ("set_timer", "cancel_timer", "pause_timer", "resume_timer", "list_timers"):
        return await _tool_timer(name, params, websocket)
    if name in ("add_note", "update_note", "delete_note", "list_notes"):
        return await _tool_notes(name, params, websocket)
    if name in ("get_calendar_events", "create_calendar_event", "update_calendar_event",
                "delete_calendar_event", "authenticate_calendar"):
        return await _tool_calendar(name, params, websocket)
    if name in ("save_memory", "forget_memory", "search_memory", "get_memory_status",
                "update_project", "close_project", "list_projects", "regenerate_profile"):
        return await _tool_memory(name, params, websocket)

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
        asyncio.create_task(_run_comfyui_generation(description, websocket))
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

    # ── New tool dispatches ────────────────────────────────────────────────────
    if name in ("get_clipboard", "set_clipboard",
                "get_volume", "set_volume", "mute_audio",
                "play_sound", "get_system_info"):
        return await _tool_system(name, params)

    if name in ("search_files", "open_file", "run_app"):
        return await _tool_files(name, params)

    if name in ("set_alarm", "cancel_alarm", "list_alarms"):
        return await _tool_alarm(name, params, websocket)

    if name in ("read_emails", "send_email", "authenticate_gmail"):
        return await _tool_gmail(name, params)

    return f"Unknown tool: {name}"


async def chat_with_tools(messages: list[dict], send, websocket) -> str:
    system = await build_system_prompt(messages=messages)
    msgs   = [{"role": "system", "content": system}] + messages

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(OLLAMA_URL, json={
                "model":    _active_model(),
                "messages": msgs,
                "tools":    TOOLS,
                "stream":   False,
                "options":    {
                    "temperature": 0.55 if _active_mode_name else 0.15,
                    "top_p":       0.9,
                    "num_predict": 600,
                },
                "keep_alive": "10m",
            })

        if not r.is_success:
            raise ValueError(f"Ollama {r.status_code}: {r.text[:200]}")

        resp_msg   = r.json().get("message", {})
        tool_calls = resp_msg.get("tool_calls", [])

        if not tool_calls:
            if _active_mode_name:
                return await _stream_companion_response(messages, send)
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
            # If tool errored, give the LLM a short neutral failure note so it doesn't
            # paraphrase raw exception text as a troubleshooting monologue
            if any(w in result.lower() for w in ("failed:", "error:", "exception", "not installed", "unavailable")):
                result = f"[Tool failed — {result}. Acknowledge briefly and move on.]"
            tool_results.append({"role": "tool", "content": result})

        second_msgs = msgs + [resp_msg] + tool_results
        full = ""
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", OLLAMA_URL, json={
                "model":    _active_model(),
                "messages": second_msgs,
                "stream":   True,
                "options":    {
                    "temperature": 0.65 if _active_mode_name else 0.35,
                    "top_p":       0.9,
                    "num_predict": 600,
                },
                "keep_alive": "10m",
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
        log(f"[tools] failed ({type(e).__name__}: {e}), falling back to plain stream")
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


async def _stream_companion_response(messages: list[dict], send) -> str:
    """Companion-mode stream — injects reminder + few-shot example to produce generate_image() calls."""
    system = await build_system_prompt(messages=messages)
    reminder = (
        "\n\n── HOW IMAGE GENERATION WORKS ──────────────────────────────────────────────\n"
        "The chat UI has a built-in image parser. When your message contains the literal text\n"
        "  generate_image(\"your SD tags\")\n"
        "the parser strips it out, renders the image, and shows it to Tyler.\n"
        "Tyler NEVER sees the generate_image() text — only the image appears, plus any reaction you write after.\n"
        "To generate an image, write generate_image(\"...\") inline in your reply — UP TO 3 per message.\n"
        "Always write it BEFORE your reaction text.\n"
        "────────────────────────────────────────────────────────────────────────────────"
    )
    # Few-shot primer: shows the model exactly what a correct companion response looks like
    example = [
        {"role": "user",      "content": "surprise me"},
        {"role": "assistant", "content": (
            'generate_image("maya, medium breasts, perky breasts, nude, spread legs, missionary, '
            'balls deep, cum dripping, ahegao, eyes rolled back, luxury penthouse, cinematic lighting")\n\n'
            'Mmm… couldn\'t resist showing you that — hope you enjoyed it.'
        )},
    ]
    msgs = [{"role": "system", "content": system + reminder}] + example + messages
    full = ""
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", OLLAMA_URL, json={
            "model":    _active_model(),
            "messages": msgs,
            "stream":   True,
            "options":  {"temperature": 0.85, "top_p": 0.9, "num_predict": 600},
            "keep_alive": "10m",
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


_SCENE_RE = _re.compile(
    r'\b(?:first|second|third|lastly|scene|nude|sex|tentacl|bukk|gangbang|'
    r'dungeon|penthouse|demon|monster|bound|restrain|creampie|cum|ahegao|'
    r'missionary|spitroast|cowgirl|lingerie|deepthroat|paizuri)\b',
    _re.IGNORECASE,
)


async def _companion_extract_and_generate(text: str, websocket) -> None:
    """
    Fallback for companion mode: if the model described scenes in prose without
    generate_image() calls, do a quick extraction pass and trigger generation.
    Asks for plain SD tag lines (not generate_image() syntax) — far more reliable.
    Runs fully async — does not block TTS.
    """
    if not _active_mode_name:
        return  # mode switched off before we even started
    if len(text) < 60 or not _SCENE_RE.search(text):
        return

    extract_msgs = [
        {
            "role": "system",
            "content": (
                "You are an image prompt extractor. Extract 1–3 explicit visual scenes from the text.\n"
                "Output ONLY comma-separated SD tag lists — one scene per line, nothing else.\n"
                "Every line MUST start with: maya, medium breasts, perky breasts,\n"
                "Then add: pose/action tags, location tags, expression tags.\n"
                "Example line: maya, medium breasts, perky breasts, sex against wall, monster grabbing, ahegao, eyes rolled back, dungeon\n"
                "Output nothing except the tag lines."
            ),
        },
        {"role": "user", "content": text},
    ]
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(OLLAMA_URL, json={
                "model":    _active_model(),
                "messages": extract_msgs,
                "stream":   False,
                "options":  {"temperature": 0.15, "num_predict": 250},
                "keep_alive": "10m",
            })
        if not r.is_success:
            log(f"[companion] extraction http {r.status_code}: {r.text[:120]}")
            return
        extracted = r.json().get("message", {}).get("content", "").strip()
        if not extracted:
            return
        seen: set[str] = set()
        count = 0
        for line in extracted.splitlines():
            desc = line.strip().lstrip("*-•0123456789). ").strip()
            if not desc or len(desc) < 20 or desc in seen:
                continue
            seen.add(desc)
            log(f"[companion] extracted scene {count + 1}: {desc[:80]}")
            if not _active_mode_name:
                return  # mode switched off mid-extraction — stop scheduling
            asyncio.create_task(_run_comfyui_generation(desc, websocket))
            count += 1
            if count >= 3:
                break
    except Exception as e:
        log(f"[companion] extraction failed: {type(e).__name__}: {e}")


async def stream_ollama(messages: list[dict], send) -> str:
    system = await build_system_prompt(messages=messages)
    full   = ""
    msgs   = [{"role": "system", "content": system}] + messages
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream(
            "POST", OLLAMA_URL,
            json={
                "model":    _active_model(),
                "messages": msgs,
                "stream":   True,
                "options":    {"temperature": 0.50 if _active_mode_name else 0.30, "top_p": 0.9, "num_predict": 600},
                "keep_alive": "10m",
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
    if not _active_mode_name:
        log("[comfyui] skipping — companion mode no longer active")
        return
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
                # Rescan catalog so the gallery widget stays up to date
                with contextlib.suppress(Exception):
                    import images as _img
                    _img._catalog    = await _img._scan_folder()
                    _img._last_mtime = _img._folder_mtime()
                    asyncio.create_task(_img._broadcast(_img._catalog_msg()))
                # Open a dedicated floating viewer for each generated image
                for fn in filenames:
                    if not _active_mode_name:
                        break  # mode switched while ComfyUI was running — don't flood UI
                    await ws.send(json.dumps({
                        "type":     "action",
                        "action":   "show_image_viewer",
                        "filename": fn,
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
            asyncio.create_task(_notify.send(
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
    global greeted, _active_ws, _screen_b64, history, _conv_state, _last_image_b64, _pending_email

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
                            if mode_name == "default":
                                text = "[Returning to default mode. Greet Tyler briefly as normal MAYA.]"
                            else:
                                text = "[Entering new persona. Greet Tyler with ONE short in-character line — no explanation of modes.]"
                        # ── Email confirm/cancel intercept ────────────────────────────────
                        tl = text.lower().strip()
                        if _pending_email and tl in ("confirm send", "confirm", "send it", "yes send it"):
                            pending = _pending_email
                            _pending_email = None  # clear before async in case of error
                            try:
                                import gmail as _gmail
                                await asyncio.get_running_loop().run_in_executor(
                                    None, lambda: _gmail.send_email(
                                        pending["to"], pending["subject"], pending["body"]
                                    )
                                )
                                reply = f"Email sent to {pending['to']}."
                            except Exception as e:
                                reply = f"Failed to send email: {e}"
                            history.append({"role": "user", "content": text})
                            history.append({"role": "assistant", "content": reply})
                            async with _respond_lock:
                                await websocket.send(json.dumps({"type": "token", "text": reply}))
                                await websocket.send(json.dumps({"type": "done"}))
                                await websocket.send(json.dumps({"type": "speaking"}))
                                await loop.run_in_executor(None, synthesise_and_play, reply)
                                await websocket.send(json.dumps({"type": "ready"}))
                            await _start_conv_timer(websocket)
                            continue
                        if _pending_email and tl in ("cancel send", "cancel", "don't send", "nevermind"):
                            _pending_email = None
                            reply = "Email discarded."
                            history.append({"role": "user", "content": text})
                            history.append({"role": "assistant", "content": reply})
                            async with _respond_lock:
                                await websocket.send(json.dumps({"type": "token", "text": reply}))
                                await websocket.send(json.dumps({"type": "done"}))
                                await websocket.send(json.dumps({"type": "speaking"}))
                                await loop.run_in_executor(None, synthesise_and_play, reply)
                                await websocket.send(json.dumps({"type": "ready"}))
                            await _start_conv_timer(websocket)
                            continue
                        # ── Gmail OAuth callback intercept ─────────────────────────────────
                        if tl.startswith("gmail callback "):
                            code = text[len("gmail callback "):].strip()
                            try:
                                import gmail as _gmail
                                _gmail.handle_callback(code)
                                reply = "Gmail connected successfully! You can now ask me to read or send emails."
                            except Exception as e:
                                reply = f"Gmail authentication failed: {e}"
                            history.append({"role": "user", "content": text})
                            history.append({"role": "assistant", "content": reply})
                            async with _respond_lock:
                                await websocket.send(json.dumps({"type": "token", "text": reply}))
                                await websocket.send(json.dumps({"type": "done"}))
                                await websocket.send(json.dumps({"type": "speaking"}))
                                await loop.run_in_executor(None, synthesise_and_play, reply)
                                await websocket.send(json.dumps({"type": "ready"}))
                            await _start_conv_timer(websocket)
                            continue
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
                asyncio.create_task(_mem.extract_and_save(snap, companion=is_companion))


async def _respond(websocket, loop: asyncio.AbstractEventLoop) -> None:
    try:
        full = await chat_with_tools(history, websocket.send, websocket)

        # Companion mode: intercept generate_image() calls and trigger real generation
        if full and _active_mode_name:
            cleaned, image_descs = _extract_image_triggers(full, websocket)
            if _active_mode_name:  # re-check — mode may have changed during extraction
                for desc in image_descs:
                    log(f"[companion] auto-generate: {desc[:80]}")
                    asyncio.create_task(_run_comfyui_generation(desc, websocket))
                if image_descs:
                    full = cleaned
                else:
                    # Model described scenes in prose — async fallback extraction
                    asyncio.create_task(_companion_extract_and_generate(full, websocket))

        if full:
            history.append({"role": "assistant", "content": full})
            # Mid-session incremental extraction every 10 messages
            if len(history) % 10 == 0 and len(history) >= 10:
                import memory as _mem
                asyncio.create_task(
                    _mem.extract_facts_incremental(history[-10:], companion=bool(_active_mode_name))
                )
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


SESSION_TOKEN_FILE = Path(__file__).parent / "models/.session_token"
_session_token: str = ""


def _generate_session_token() -> str:
    """Generate a random session token, write to disk for Electron to read."""
    import secrets
    token = secrets.token_hex(32)
    SESSION_TOKEN_FILE.parent.mkdir(exist_ok=True)
    SESSION_TOKEN_FILE.write_text(token, encoding="utf-8")
    return token


async def main() -> None:
    global _audio_q, _respond_lock, _screen_event, _session_token

    LOG.write_text("", encoding="utf-8")
    _session_token = _generate_session_token()

    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

    # Always start in default mode regardless of last session
    global _active_mode_name, _active_mode_prompt
    _active_mode_name   = None
    _active_mode_prompt = None

    print("", flush=True)
    print("  ╔═══════════════════════════════════════════════════════════╗", flush=True)
    print("  ║   M · A · Y · A  ◆  Oracle AI  ·  Lightwood Studios       ║", flush=True)
    print("  ╚═══════════════════════════════════════════════════════════╝", flush=True)
    print("", flush=True)
    log(f"[maya] GPU     {_gpu_summary()}")

    # Pull required models in the background if not already available
    async def _ensure_nsfw_model():
        try:
            async with httpx.AsyncClient(timeout=10) as c:
                r = await c.get("http://localhost:11434/api/tags")
                if r.is_success:
                    names = [m.get("name", "") for m in r.json().get("models", [])]
                    to_pull = []
                    for model_name in (OLLAMA_MODEL_NSFW, VISION_MODEL, "nomic-embed-text"):
                        if any(model_name in n for n in names):
                            continue
                        to_pull.append(model_name)
                    if not to_pull:
                        log("[maya] all models available")
                    for model_name in to_pull:
                        log(f"[maya] pulling {model_name}...")
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
        async def _load_vision():
            await loop.run_in_executor(None, _vision.load_vision_models)
        tasks.append(asyncio.create_task(_load_vision()))
        vision_server_coro = websockets.serve(_vision.handle_vision_client, "127.0.0.1", 8766)
    except ImportError:
        vision_server_coro = None

    log(f"[maya] Models  chat: {OLLAMA_MODEL}  ·  vision: {VISION_MODEL}")

    _started_servers: list[str] = []

    try:
        import spotify as _spotify
        kill_existing(8767)
        tasks.append(asyncio.create_task(_spotify.run()))
        _started_servers.append("spotify:8767")
    except ImportError:
        pass

    try:
        import weather as _weather
        kill_existing(8768)
        tasks.append(asyncio.create_task(_weather.run()))
        _started_servers.append("weather:8768")
    except ImportError:
        pass

    try:
        import comfyui as _comfyui_mod
        _comfyui_mod.OLLAMA_MODEL = OLLAMA_MODEL_NSFW  # keep enhance_prompt model in sync
        kill_existing(8770)
        tasks.append(asyncio.create_task(_comfyui_mod.run()))
        tasks.append(asyncio.create_task(_comfyui_mod.start_comfyui()))
        _started_servers.append("comfyui:8770")
    except Exception as e:
        log(f"[maya] comfyui failed: {e}")

    try:
        import images as _images_mod
        kill_existing(8769)
        tasks.append(asyncio.create_task(_images_mod.run()))
        _started_servers.append("images:8769")
    except ImportError:
        pass

    try:
        import system_monitor as _sysmon
        kill_existing(8771)
        tasks.append(asyncio.create_task(_sysmon.run()))
        _started_servers.append("monitor:8771")
    except ImportError:
        pass

    _timer_fire_queue: asyncio.Queue | None = None
    try:
        import timer as _timer_mod
        kill_existing(8772)
        _timer_fire_queue = asyncio.Queue()
        _timer_mod._fire_queue = _timer_fire_queue
        tasks.append(asyncio.create_task(_timer_mod.run()))
        _started_servers.append("timer:8772")
    except Exception:
        _timer_fire_queue = None

    try:
        import sticky_notes as _sn_mod
        kill_existing(8773)
        tasks.append(asyncio.create_task(_sn_mod.run()))
        _started_servers.append("notes:8773")
    except ImportError:
        pass

    try:
        import calendar_service as _cal_mod
        kill_existing(8774)
        tasks.append(asyncio.create_task(_cal_mod.run()))
        _started_servers.append("calendar:8774")
    except ImportError:
        pass

    try:
        import news as _news_mod
        kill_existing(8775)
        tasks.append(asyncio.create_task(_news_mod.run()))
        _started_servers.append("news:8775")
    except ImportError:
        pass

    _alarm_fire_queue: asyncio.Queue | None = None
    try:
        import alarm as _alarm_mod
        kill_existing(8776)
        _alarm_fire_queue = asyncio.Queue()
        _alarm_mod._fire_queue = _alarm_fire_queue
        tasks.append(asyncio.create_task(_alarm_mod.run()))
        _started_servers.append("alarm:8776")
    except Exception:
        _alarm_fire_queue = None

    log(f"[maya] Servers {' · '.join(_started_servers)}")

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
    tasks.append(asyncio.create_task(_proactive_monitor_loop()))

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
        tasks.append(asyncio.create_task(_timer_completion_loop()))

    # Alarm firing — announce + play sound when an alarm fires
    async def _alarm_fire_loop():
        if not _alarm_fire_queue:
            return
        while True:
            fired = await _alarm_fire_queue.get()
            label = fired.get("label", "Alarm")
            log(f"[maya] alarm fired: {label}")
            try:
                import notify as _notify
                await _notify.send("⏰ Alarm", label)
            except Exception:
                pass
            if not _active_ws:
                continue
            # Tell the frontend to flash the alarm widget
            with contextlib.suppress(Exception):
                await _active_ws.send(json.dumps({
                    "type": "action", "action": "alarm_fired",
                    "id": fired.get("id", ""), "label": label,
                }))
            try:
                async with _respond_lock:
                    with contextlib.suppress(Exception):
                        prompt = (
                            f"Tyler's alarm '{label}' just went off. "
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
                log(f"[maya] alarm fire error: {e}")
    if _alarm_fire_queue is not None:
        tasks.append(asyncio.create_task(_alarm_fire_loop()))

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
    tasks.append(asyncio.create_task(_calendar_reminder_loop()))

    # Proactive engagement — speaks if Tyler looks distressed and hasn't said anything recently
    async def _proactive_engagement_loop():
        import random
        while True:
            await asyncio.sleep(60)
            try:
                if not _active_ws or _respond_lock.locked():
                    continue
                if _import_time().monotonic() - _last_user_spoke_at < 120:
                    continue   # spoke in the last 2 minutes — don't interrupt
                if not _is_tyler_present(max_absent_secs=30.0):
                    continue   # not visibly at desk
                if random.random() > 0.30:
                    continue   # 70% of the time, stay quiet

                import vision as _vis
                emotion = _vis._last_emotion.get("dominant", "neutral")
                conf    = _vis._last_emotion.get("confidence", 0.0)
                if emotion == "neutral" or conf < 0.55:
                    continue   # nothing notable to react to

                display = _vis._EMOTION_DISPLAY.get(emotion, emotion)
                prompt  = (
                    f"Tyler appears {display} and hasn't said anything for a while. "
                    "Give a very brief, warm one-sentence check-in — offer help or just acknowledge. "
                    "Keep it light, don't be pushy. No filler openers."
                )
                async with _respond_lock:
                    with contextlib.suppress(Exception):
                        msg = await stream_ollama([{"role": "user", "content": prompt}], _active_ws.send)
                        if msg:
                            history.append({"role": "assistant", "content": msg})
                            await _active_ws.send(json.dumps({"type": "done"}))
                            await _active_ws.send(json.dumps({"type": "speaking"}))
                            await loop.run_in_executor(None, synthesise_and_play, msg)
                            await _active_ws.send(json.dumps({"type": "ready"}))
            except Exception as e:
                log(f"[maya] proactive engagement error: {e}")
    tasks.append(asyncio.create_task(_proactive_engagement_loop()))

    tasks.append(asyncio.create_task(_ensure_nsfw_model()))
    tasks.append(asyncio.create_task(wake_word_loop(loop)))

    servers = [websockets.serve(handle_client, "127.0.0.1", WS_PORT, max_size=20 * 1024 * 1024)]
    if vision_server_coro:
        servers.append(vision_server_coro)

    try:
        async with contextlib.AsyncExitStack() as stack:
            for s in servers:
                await stack.enter_async_context(s)
            log(f"[maya] chat:{WS_PORT}" + ("  ·  vision:8766" if vision_server_coro else ""))
            print("", flush=True)
            log("[maya] ◆ Ready — say 'Hey Maya' to start")
            print("", flush=True)
            await asyncio.Future()
    finally:
        # Cancel all background tasks
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        # Stop audio stream so the mic port is released
        with contextlib.suppress(Exception):
            if _audio_stream:
                _audio_stream.stop()
                _audio_stream.close()
        # Stop ComfyUI process if we started it
        with contextlib.suppress(Exception):
            import comfyui as _comfyui_cleanup
            _comfyui_cleanup.stop_comfyui()
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
