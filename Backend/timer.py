"""MAYA Timer — multiple named countdown timers."""
import asyncio
import contextlib
import json
import time

import websockets

TIMER_PORT = 8772

_clients:    set  = set()
_timers:     dict = {}   # name → {duration, remaining, state}
_fire_queue: "asyncio.Queue | None" = None   # injected by server.py before run()
_last_tick:  float = 0.0


# ── Sync helpers (called from server.py execute_tool) ────────────────────────

def create(name: str, seconds: int) -> None:
    _timers[name.lower()] = {
        "duration":  seconds,
        "remaining": float(seconds),
        "state":     "running",
    }


def cancel(name: str) -> str:
    key = name.lower()
    if key == "all":
        removed = list(_timers.keys())
        _timers.clear()
        return f"Cancelled all timers ({', '.join(removed)})." if removed else "No active timers."
    if key in _timers:
        del _timers[key]
        return f"Cancelled the {name} timer."
    return f"No timer named '{name}'."


def pause(name: str) -> str:
    key = name.lower()
    if key in _timers and _timers[key]["state"] == "running":
        _timers[key]["state"] = "paused"
        return f"Paused the {name} timer."
    return f"No running timer named '{name}'."


def resume(name: str) -> str:
    key = name.lower()
    if key in _timers and _timers[key]["state"] == "paused":
        _timers[key]["state"] = "running"
        return f"Resumed the {name} timer."
    return f"No paused timer named '{name}'."


def list_all() -> str:
    if not _timers:
        return "No active timers."
    lines = []
    for name, t in _timers.items():
        rem = int(max(0, t["remaining"]))
        m, s = divmod(rem, 60)
        lines.append(f"{name}: {m:02d}:{s:02d} ({t['state']})")
    return "\n".join(lines)


# ── Broadcast helpers ─────────────────────────────────────────────────────────

def _payload() -> list:
    return [
        {
            "name":      name,
            "duration":  t["duration"],
            "remaining": max(0.0, t["remaining"]),
            "state":     t["state"],
        }
        for name, t in _timers.items()
    ]


async def _broadcast(msg: dict) -> None:
    raw = json.dumps(msg)
    for ws in list(_clients):
        with contextlib.suppress(Exception):
            await ws.send(raw)


# ── Tick loop ─────────────────────────────────────────────────────────────────

async def _tick_loop() -> None:
    global _last_tick
    _last_tick = time.monotonic()
    while True:
        await asyncio.sleep(0.25)
        now        = time.monotonic()
        dt         = now - _last_tick
        _last_tick = now

        fired = []
        for name, t in list(_timers.items()):
            if t["state"] != "running":
                continue
            t["remaining"] -= dt
            if t["remaining"] <= 0:
                t["remaining"] = 0.0
                t["state"]     = "done"
                fired.append(name)

        if _timers:
            await _broadcast({"type": "timers_update", "timers": _payload()})

        for name in fired:
            if _fire_queue is not None:
                await _fire_queue.put({"name": name})
            # Keep done timer visible for 5 s then remove it
            await asyncio.sleep(5)
            _timers.pop(name, None)
        if fired:
            await _broadcast({"type": "timers_update", "timers": _payload()})


# ── WebSocket handler ─────────────────────────────────────────────────────────

async def handle_client(websocket) -> None:
    _clients.add(websocket)
    try:
        await websocket.send(json.dumps({"type": "timers_update", "timers": _payload()}))
        async for raw in websocket:
            try:
                msg  = json.loads(raw)
                kind = msg.get("type", "")
                name = msg.get("name", "")
                if   kind == "create": create(name, int(msg.get("seconds", 60)))
                elif kind == "pause":  pause(name)
                elif kind == "resume": resume(name)
                elif kind == "cancel": cancel(name)
                await _broadcast({"type": "timers_update", "timers": _payload()})
            except Exception:
                pass
    except Exception:
        pass
    finally:
        _clients.discard(websocket)


async def run() -> None:
    asyncio.ensure_future(_tick_loop())
    async with websockets.serve(handle_client, "127.0.0.1", TIMER_PORT):
        print(f"[timer] ws://127.0.0.1:{TIMER_PORT}", flush=True)
        await asyncio.Future()
