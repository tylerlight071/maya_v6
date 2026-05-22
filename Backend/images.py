"""
MAYA Images — local image gallery from a configured folder.
Generates base64 thumbnails via Pillow, serves over WebSocket.
Watches for folder changes every 30 seconds.
"""
import asyncio
import base64
import contextlib
import io
import json
import time
from pathlib import Path

import websockets

IMAGES_PORT   = 8769
IMAGES_DIR    = Path(r"C:\Users\tyler\Pictures\Images")
THUMB_SIZE    = (200, 200)
THUMB_QUALITY = 72
POLL_INTERVAL = 30
EXTENSIONS    = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}

_clients:    set        = set()
_catalog:    list[dict] = []
_last_mtime: float      = 0.0


# ── Thumbnail generation ──────────────────────────────────────────────────────

def _make_thumb(path: Path) -> "tuple[str | None, int, int]":
    """
    Generate a base64 JPEG thumbnail.
    Returns (data_uri, orig_width, orig_height) or (None, 0, 0) on failure.
    """
    try:
        from PIL import Image
        img = Image.open(path)

        # Handle GIF — use first frame only
        if getattr(img, "is_animated", False) or img.format == "GIF":
            img.seek(0)

        orig_w, orig_h = img.size

        # Convert palette / RGBA → RGB (JPEG encoder requires RGB)
        if img.mode in ("P", "RGBA", "LA", "L"):
            background = Image.new("RGB", img.size, (13, 10, 24))  # MAYA dark bg
            if img.mode in ("RGBA", "LA"):
                background.paste(img, mask=img.split()[-1])
            else:
                background.paste(img.convert("RGB"))
            img = background
        elif img.mode != "RGB":
            img = img.convert("RGB")

        img.thumbnail(THUMB_SIZE, Image.LANCZOS)

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=THUMB_QUALITY, optimize=True)
        b64 = base64.b64encode(buf.getvalue()).decode()
        return f"data:image/jpeg;base64,{b64}", orig_w, orig_h

    except Exception as e:
        print(f"[images] thumb failed for {path.name}: {e}", flush=True)
        return None, 0, 0


# ── Folder scan ───────────────────────────────────────────────────────────────

async def _scan_folder() -> list[dict]:
    """
    Scan IMAGES_DIR, generate thumbnails in thread pool.
    Returns list sorted by modification time descending (newest first).
    """
    if not IMAGES_DIR.exists():
        print(f"[images] folder not found: {IMAGES_DIR}", flush=True)
        return []

    loop = asyncio.get_running_loop()
    entries = sorted(
        (f for f in IMAGES_DIR.iterdir() if f.suffix.lower() in EXTENSIONS),
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )

    # Generate thumbnails concurrently (up to 8 at a time) instead of one-by-one
    sem = asyncio.Semaphore(8)

    async def process(f: Path) -> "dict | None":
        async with sem:
            try:
                stat        = f.stat()
                thumb, w, h = await loop.run_in_executor(None, _make_thumb, f)
                if thumb is None:
                    return None
                return {
                    "filename":   f.name,
                    "thumb_b64":  thumb,
                    "width":      w,
                    "height":     h,
                    "mtime":      int(stat.st_mtime * 1000),
                    "size_bytes": stat.st_size,
                }
            except Exception as e:
                print(f"[images] skipping {f.name}: {e}", flush=True)
                return None

    results = await asyncio.gather(*(process(f) for f in entries))
    # Filter None and preserve newest-first order
    return [r for r in results if r is not None]


def _folder_mtime() -> float:
    try:
        return IMAGES_DIR.stat().st_mtime
    except Exception:
        return 0.0


# ── Broadcast ─────────────────────────────────────────────────────────────────

async def _broadcast(msg: dict) -> None:
    if not _clients:
        return
    raw  = json.dumps(msg)
    dead = set()
    for ws in list(_clients):
        try:
            await ws.send(raw)
        except Exception:
            dead.add(ws)
    _clients.difference_update(dead)


def _catalog_msg() -> dict:
    return {
        "type":      "images_catalog",
        "images":    _catalog,
        "total":     len(_catalog),
        "folder":    str(IMAGES_DIR),
        "timestamp": int(time.time() * 1000),
    }


# ── Poll loop ─────────────────────────────────────────────────────────────────

async def _poll() -> None:
    global _catalog, _last_mtime
    while True:
        await asyncio.sleep(POLL_INTERVAL)
        try:
            mtime = _folder_mtime()
            if mtime != _last_mtime:
                print("[images] folder changed — rescanning...", flush=True)
                _catalog    = await _scan_folder()
                _last_mtime = mtime
                await _broadcast(_catalog_msg())
                print(f"[images] rescanned: {len(_catalog)} images", flush=True)
        except Exception as e:
            print(f"[images] poll error: {e}", flush=True)


# ── WebSocket handler ─────────────────────────────────────────────────────────

async def handle_client(websocket) -> None:
    global _catalog, _last_mtime   # declared at top so all uses in this function are valid
    _clients.add(websocket)
    try:
        if _catalog:
            await websocket.send(json.dumps(_catalog_msg()))
        else:
            await websocket.send(json.dumps({"type": "images_loading"}))

        async for raw in websocket:
            try:
                msg = json.loads(raw)
            except Exception:
                continue

            if msg.get("type") == "request_refresh":
                _catalog    = await _scan_folder()
                _last_mtime = _folder_mtime()
                await websocket.send(json.dumps(_catalog_msg()))

    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        _clients.discard(websocket)


# ── Entry point ───────────────────────────────────────────────────────────────

async def run() -> None:
    global _catalog, _last_mtime

    print(f"[images] scanning {IMAGES_DIR} ...", flush=True)
    _catalog    = await _scan_folder()
    _last_mtime = _folder_mtime()
    print(f"[images] {len(_catalog)} images loaded", flush=True)

    poll_task = None
    try:
        async with websockets.serve(handle_client, "127.0.0.1", IMAGES_PORT):
            print(f"[images] ws://127.0.0.1:{IMAGES_PORT}", flush=True)
            poll_task = asyncio.create_task(_poll())
            await asyncio.Future()
    finally:
        if poll_task and not poll_task.done():
            poll_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await poll_task
