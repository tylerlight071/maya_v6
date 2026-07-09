"""MAYA Alarm — set alarms that fire at specific times of day."""
import asyncio
import contextlib
import json
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import websockets

ALARM_PORT = 8776

_clients:    set  = set()
_alarms:     dict = {}   # id → {label, fire_at_str, fire_dt, task}
_fire_queue: "asyncio.Queue | None" = None  # injected by server.py


def _parse_time(time_str: str) -> "datetime | None":
    """Parse a time string like '07:30', '7:30am', '14:00', '2pm' into today's datetime."""
    now = datetime.now()
    time_str = time_str.strip().lower().replace(".", ":")

    # Try common formats
    for fmt in ("%H:%M", "%I:%M%p", "%I:%M %p", "%I%p", "%I %p"):
        try:
            t = datetime.strptime(time_str, fmt)
            dt = now.replace(hour=t.hour, minute=t.minute, second=0, microsecond=0)
            if dt <= now:
                dt += timedelta(days=1)  # push to tomorrow if time has passed
            return dt
        except ValueError:
            pass

    # Try plain hour like "7" or "14"
    try:
        hour = int(time_str.replace("h", "").strip())
        if 0 <= hour <= 23:
            dt = now.replace(hour=hour, minute=0, second=0, microsecond=0)
            if dt <= now:
                dt += timedelta(days=1)
            return dt
    except ValueError:
        pass

    return None


async def _alarm_task(alarm_id: str, fire_dt: datetime) -> None:
    """Sleep until fire_dt, then fire the alarm."""
    delay = (fire_dt - datetime.now()).total_seconds()
    if delay > 0:
        await asyncio.sleep(delay)

    alarm = _alarms.pop(alarm_id, None)
    if alarm is None:
        return  # was cancelled

    label = alarm.get("label", "Alarm")
    print(f"[alarm] firing: {label}", flush=True)

    # Push to server.py fire queue
    if _fire_queue is not None:
        await _fire_queue.put({"id": alarm_id, "label": label, "fire_at": alarm.get("fire_at_str", "")})

    # Broadcast to widget
    await _broadcast({"type": "alarm_fired", "id": alarm_id, "label": label})
    await _broadcast_list()

    # Play alarm sound
    try:
        import soundfile as sf
        import sounddevice as sd
        sounds_dir = Path(__file__).parent / "sounds"
        for name in ("alarm.wav", "alert.wav", "chime.wav"):
            p = sounds_dir / name
            if p.exists():
                data, sr = sf.read(str(p))
                sd.play(data, sr)
                break
    except Exception as e:
        print(f"[alarm] sound failed: {e}", flush=True)


def set_alarm(time_str: str, label: str = "Alarm") -> "tuple[str, str]":
    """Set an alarm. Returns (description_string, alarm_id) or (error_string, '')."""
    fire_dt = _parse_time(time_str)
    if fire_dt is None:
        return f"Couldn't parse time '{time_str}'. Try '07:30', '2pm', or '14:00'.", ""

    alarm_id   = str(uuid.uuid4())[:8]
    fire_str   = fire_dt.strftime("%H:%M")
    day_label  = "today" if fire_dt.date() == datetime.now().date() else "tomorrow"

    task = asyncio.create_task(_alarm_task(alarm_id, fire_dt))

    _alarms[alarm_id] = {
        "label":       label,
        "fire_at_str": f"{fire_str} {day_label}",
        "fire_dt":     fire_dt,
        "task":        task,
    }

    asyncio.create_task(_broadcast_list())
    return f"Alarm set for {fire_str} {day_label} — '{label}'. ID: {alarm_id}.", alarm_id


def cancel_alarm(alarm_id: str) -> str:
    """Cancel an alarm by its ID."""
    alarm_id = alarm_id.strip()
    # Support prefix match (first 8 chars)
    match = next((k for k in _alarms if k.startswith(alarm_id) or alarm_id.startswith(k)), None)
    if not match:
        return f"No alarm with ID '{alarm_id}' found."
    alarm = _alarms.pop(match)
    alarm.get("task", None).cancel() if alarm.get("task") else None
    asyncio.create_task(_broadcast_list())
    return f"Alarm '{alarm['label']}' cancelled."


def list_alarms() -> str:
    """List all active alarms."""
    if not _alarms:
        return "No active alarms."
    lines = []
    for aid, a in _alarms.items():
        lines.append(f"[{aid}] {a['label']} — {a['fire_at_str']}")
    return "\n".join(lines)


# ── WebSocket server for the AlarmWidget ─────────────────────────────────────

async def _broadcast(msg: dict) -> None:
    raw = json.dumps(msg)
    for ws in list(_clients):
        with contextlib.suppress(Exception):
            await ws.send(raw)


async def _broadcast_list() -> None:
    alarms_list = [
        {"id": aid, "label": a["label"], "fire_at": a["fire_at_str"]}
        for aid, a in _alarms.items()
    ]
    await _broadcast({"type": "alarms", "alarms": alarms_list})


async def _handler(websocket) -> None:
    import auth
    if not await auth.check(websocket):
        return
    _clients.add(websocket)
    try:
        await _broadcast_list()  # send current state on connect
        async for raw in websocket:
            try:
                msg = json.loads(raw)
            except Exception:
                continue
            if msg.get("type") == "cancel":
                alarm_id = msg.get("id", "")
                result   = cancel_alarm(alarm_id)
                await _broadcast_list()
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        _clients.discard(websocket)


async def run() -> None:
    async with websockets.serve(_handler, "127.0.0.1", ALARM_PORT):
        print(f"[alarm] WebSocket server running on port {ALARM_PORT}", flush=True)
        await asyncio.Future()  # run forever
