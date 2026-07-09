"""MAYA Sticky Notes — persistent CRUD note storage."""
import asyncio
import contextlib
import json
import time
import uuid
from pathlib import Path

import websockets

NOTES_PORT = 8773
NOTES_FILE = Path(__file__).parent / "memory/notes.json"

_clients: set  = set()
_notes:   list = []   # [{"id": str, "text": str, "created": float, "updated": float}]


# ── Persistence ───────────────────────────────────────────────────────────────

def _load() -> None:
    global _notes
    try:
        if NOTES_FILE.exists():
            data   = json.loads(NOTES_FILE.read_text(encoding="utf-8"))
            _notes = data if isinstance(data, list) else []
    except Exception:
        _notes = []


def _save() -> None:
    try:
        NOTES_FILE.parent.mkdir(parents=True, exist_ok=True)
        NOTES_FILE.write_text(
            json.dumps(_notes, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except Exception as e:
        print(f"[notes] save error: {e}", flush=True)


# ── Sync CRUD helpers (called from server.py execute_tool) ───────────────────

def add(text: str) -> str:
    """Create a note and return its id."""
    now  = time.time()
    note = {"id": str(uuid.uuid4()), "text": text, "created": now, "updated": now}
    _notes.append(note)
    _save()
    return note["id"]


def update(note_id: str, text: str) -> bool:
    for note in _notes:
        if note["id"] == note_id:
            note["text"]    = text
            note["updated"] = time.time()
            _save()
            return True
    return False


def delete(note_id: str) -> bool:
    global _notes
    before = len(_notes)
    _notes  = [n for n in _notes if n["id"] != note_id]
    if len(_notes) < before:
        _save()
        return True
    return False


def get_all() -> list:
    return list(_notes)


def format_for_prompt() -> str:
    if not _notes:
        return ""
    lines = [f"- [{n['id'][:8]}] {n['text']}" for n in _notes]
    return "STICKY NOTES (id prefix shown for delete/update):\n" + "\n".join(lines)


# ── Broadcast ─────────────────────────────────────────────────────────────────

async def _broadcast(msg: dict) -> None:
    raw = json.dumps(msg)
    for ws in list(_clients):
        with contextlib.suppress(Exception):
            await ws.send(raw)


async def broadcast_notes() -> None:
    await _broadcast({"type": "notes_data", "notes": _notes})


# ── WebSocket handler ─────────────────────────────────────────────────────────

async def handle_client(websocket) -> None:
    import auth
    if not await auth.check(websocket):
        return
    _clients.add(websocket)
    try:
        await websocket.send(json.dumps({"type": "notes_data", "notes": _notes}))
        async for raw in websocket:
            try:
                msg  = json.loads(raw)
                kind = msg.get("type", "")
                if kind == "update_note":
                    update(msg["id"], msg["text"])
                    await _broadcast({"type": "notes_data", "notes": _notes})
                elif kind == "delete_note":
                    delete(msg["id"])
                    await _broadcast({"type": "notes_data", "notes": _notes})
            except Exception:
                pass
    except Exception:
        pass
    finally:
        _clients.discard(websocket)


async def run() -> None:
    _load()
    print(f"[notes] {len(_notes)} note(s) loaded", flush=True)
    async with websockets.serve(handle_client, "127.0.0.1", NOTES_PORT):
        print(f"[notes] ws://127.0.0.1:{NOTES_PORT}", flush=True)
        await asyncio.Future()
