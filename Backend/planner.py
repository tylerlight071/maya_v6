"""
MAYA Planner — local tasks and calendar events.
Stored in Backend/models/tasks.json and Backend/models/events.json
WebSocket on port 8771; broadcasts updates to all connected widget clients.
"""
import asyncio
import contextlib
import json
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path

import websockets

PLANNER_PORT = 8771
MODELS_DIR   = Path(__file__).parent / "models"
TASKS_FILE   = MODELS_DIR / "tasks.json"
EVENTS_FILE  = MODELS_DIR / "events.json"

_clients: set = set()
_tasks:   list = []
_events:  list = []


# ── Persistence ───────────────────────────────────────────────────────────────

def _load_tasks() -> list:
    try:
        if TASKS_FILE.exists():
            return json.loads(TASKS_FILE.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[planner] tasks load error: {e}", flush=True)
    return []


def _load_events() -> list:
    try:
        if EVENTS_FILE.exists():
            return json.loads(EVENTS_FILE.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[planner] events load error: {e}", flush=True)
    return []


def _save_tasks() -> None:
    try:
        MODELS_DIR.mkdir(exist_ok=True)
        TASKS_FILE.write_text(json.dumps(_tasks, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        print(f"[planner] tasks save error: {e}", flush=True)


def _save_events() -> None:
    try:
        MODELS_DIR.mkdir(exist_ok=True)
        EVENTS_FILE.write_text(json.dumps(_events, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        print(f"[planner] events save error: {e}", flush=True)


# ── Time normalisation ───────────────────────────────────────────────────────

def _normalize_time(t: str) -> str:
    """Ensure time strings are zero-padded HH:MM for correct lexicographic sorting."""
    if not t:
        return t
    t = t.strip()
    parts = t.split(":")
    if len(parts) == 2:
        try:
            return f"{int(parts[0]):02d}:{parts[1]}"
        except ValueError:
            pass
    return t


# ── Natural language date parser ──────────────────────────────────────────────

def parse_date(s: str) -> str:
    """Convert natural language or formatted date to YYYY-MM-DD. Falls back to today."""
    if not s:
        return date.today().isoformat()

    low   = s.lower().strip()
    today = date.today()

    # Natural language shortcuts
    if low in ("today", ""):
        return today.isoformat()
    if low == "tomorrow":
        return (today + timedelta(days=1)).isoformat()
    if low == "yesterday":
        return (today - timedelta(days=1)).isoformat()

    # "next monday" / "monday" / "this friday" — always means the coming occurrence
    weekdays = ["monday","tuesday","wednesday","thursday","friday","saturday","sunday"]
    for i, day in enumerate(weekdays):
        if low.startswith(f"next {day}") or low.startswith(f"this {day}") or low == day:
            diff = i - today.weekday()
            if diff <= 0:
                diff += 7
            return (today + timedelta(days=diff)).isoformat()

    # "in N days" / "in N weeks"
    import re as _re
    m = _re.match(r"in\s+(\d+)\s+(day|days|week|weeks)", low)
    if m:
        n    = int(m.group(1))
        unit = m.group(2)
        delta = timedelta(weeks=n) if "week" in unit else timedelta(days=n)
        return (today + delta).isoformat()

    # Everything else: let dateutil parse it (handles ISO, RFC, month names, etc.)
    try:
        from dateutil import parser as _duparser
        dt = _duparser.parse(s, default=datetime(today.year, today.month, today.day))
        return dt.date().isoformat()
    except Exception:
        pass

    return today.isoformat()


# ── Broadcast ─────────────────────────────────────────────────────────────────

async def _broadcast_tasks() -> None:
    if not _clients:
        return
    raw  = json.dumps({"type": "tasks_update", "tasks": _tasks})
    dead = set()
    for ws in list(_clients):
        try:
            await ws.send(raw)
        except Exception:
            dead.add(ws)
    _clients.difference_update(dead)


async def _broadcast_events() -> None:
    if not _clients:
        return
    raw  = json.dumps({"type": "events_update", "events": _events})
    dead = set()
    for ws in list(_clients):
        try:
            await ws.send(raw)
        except Exception:
            dead.add(ws)
    _clients.difference_update(dead)


# ── Public API (called by server.py tool handlers) ────────────────────────────

async def add_task(title: str, priority: str = "normal", due: str = "") -> dict:
    task = {
        "id":      uuid.uuid4().hex[:8],
        "title":   title.strip(),
        "priority": priority if priority in ("low","normal","high") else "normal",
        "due":     parse_date(due) if due else None,
        "done":    False,
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    _tasks.insert(0, task)
    _save_tasks()
    await _broadcast_tasks()
    return task


async def complete_task(query: str) -> bool:
    q = query.lower()
    for t in _tasks:
        if t["id"] == query or q in t["title"].lower():
            t["done"]         = True
            t["completed_at"] = datetime.now().isoformat(timespec="seconds")
            _save_tasks()
            await _broadcast_tasks()
            return True
    return False


async def remove_task(query: str) -> bool:
    global _tasks
    q   = query.lower()
    old = len(_tasks)
    _tasks = [t for t in _tasks if t["id"] != query and q not in t["title"].lower()]
    if len(_tasks) < old:
        _save_tasks()
        await _broadcast_tasks()
        return True
    return False


async def add_event(
    title: str, date_str: str,
    time_str: str = "", end_time: str = "", description: str = "",
) -> dict:
    event = {
        "id":          uuid.uuid4().hex[:8],
        "title":       title.strip(),
        "date":        parse_date(date_str),
        "time":        _normalize_time(time_str.strip()) or None,
        "end_time":    _normalize_time(end_time.strip()) or None,
        "description": description.strip(),
        "created":     datetime.now().isoformat(timespec="seconds"),
    }
    _events.append(event)
    _events.sort(key=lambda e: (e["date"], e.get("time") or ""))
    _save_events()
    await _broadcast_events()
    return event


async def remove_event(query: str, date_str: str = "") -> bool:
    global _events
    q    = query.lower()
    d    = parse_date(date_str) if date_str else None
    old  = len(_events)
    def keep(e: dict) -> bool:
        if e["id"] == query:
            return False
        if q in e["title"].lower():
            if d and e["date"] != d:
                return True   # title matches but wrong date — keep
            return False
        return True
    _events = [e for e in _events if keep(e)]
    if len(_events) < old:
        _save_events()
        await _broadcast_events()
        return True
    return False


def get_upcoming_events(days: int = 7) -> list:
    today = date.today().isoformat()
    end   = (date.today() + timedelta(days=days)).isoformat()
    return [e for e in _events if today <= e["date"] <= end]


def get_active_tasks() -> list:
    return [t for t in _tasks if not t.get("done")]


# ── WebSocket handler ─────────────────────────────────────────────────────────

async def handle_client(websocket) -> None:
    _clients.add(websocket)
    try:
        # Push current state on connect
        await websocket.send(json.dumps({"type": "tasks_update",  "tasks":  _tasks}))
        await websocket.send(json.dumps({"type": "events_update", "events": _events}))

        async for raw in websocket:
            try:
                msg = json.loads(raw)
            except Exception:
                continue

            t = msg.get("type")

            if t == "add_task":
                await add_task(
                    msg.get("title", ""),
                    msg.get("priority", "normal"),
                    msg.get("due", ""),
                )
            elif t == "complete_task":
                await complete_task(msg.get("id", ""))
            elif t == "remove_task":
                await remove_task(msg.get("id", ""))
            elif t == "add_event":
                await add_event(
                    msg.get("title", ""),
                    msg.get("date", ""),
                    msg.get("time", ""),
                    msg.get("end_time", ""),
                    msg.get("description", ""),
                )
            elif t == "remove_event":
                await remove_event(msg.get("id", ""), msg.get("date", ""))
            elif t == "refresh":
                await websocket.send(json.dumps({"type": "tasks_update",  "tasks":  _tasks}))
                await websocket.send(json.dumps({"type": "events_update", "events": _events}))

    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        _clients.discard(websocket)


# ── Entry point ───────────────────────────────────────────────────────────────

async def run() -> None:
    global _tasks, _events

    MODELS_DIR.mkdir(exist_ok=True)
    _tasks  = _load_tasks()
    _events = _load_events()
    print(f"[planner] {len(_tasks)} tasks, {len(_events)} events loaded", flush=True)

    async with websockets.serve(handle_client, "127.0.0.1", PLANNER_PORT):
        print(f"[planner] ws://127.0.0.1:{PLANNER_PORT}", flush=True)
        await asyncio.Future()
