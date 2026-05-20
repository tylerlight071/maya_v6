"""
MAYA Spotify integration.

Setup:
  1. https://developer.spotify.com/dashboard — create an app
  2. Add http://127.0.0.1:8888/callback as Redirect URI
  3. Fill in Backend/models/spotify_config.json
  4. Delete spotify_tokens.json if it exists (re-auth needed for new scopes)
"""
import asyncio
import base64
import contextlib
import json
import time
import urllib.parse
import webbrowser
from pathlib import Path

import httpx
import websockets

SPOTIFY_PORT  = 8767
CALLBACK_PORT = 8888
REDIRECT_URI  = f"http://127.0.0.1:{CALLBACK_PORT}/callback"
SCOPES = (
    "user-read-playback-state user-modify-playback-state "
    "user-read-currently-playing user-library-read user-library-modify "
    "user-read-recently-played playlist-read-private"
)

CONFIG_FILE = Path(__file__).parent / "models/spotify_config.json"
TOKENS_FILE = Path(__file__).parent / "models/spotify_tokens.json"

API_TIMEOUT = httpx.Timeout(10.0)   # all Spotify API calls time out after 10 s
POLL_INTERVAL = 2.0                  # seconds between playback state polls

_access_token  = None
_refresh_token = None
_token_expiry  = 0.0
_token_lock: "asyncio.Lock | None" = None   # initialised in run()
_current_track = None
_clients: set  = set()
_prev_track_id = None


# ── Config / tokens ───────────────────────────────────────────────────────────

def load_config() -> "dict | None":
    if not CONFIG_FILE.exists():
        return None
    try:
        return json.loads(CONFIG_FILE.read_text())
    except Exception:
        return None


def load_tokens() -> bool:
    global _access_token, _refresh_token, _token_expiry
    if not TOKENS_FILE.exists():
        return False
    try:
        t = json.loads(TOKENS_FILE.read_text())
        _access_token  = t.get("access_token")
        _refresh_token = t.get("refresh_token")
        _token_expiry  = t.get("expiry", 0)
        return bool(_access_token and _refresh_token)
    except Exception:
        return False


def save_tokens() -> None:
    TOKENS_FILE.parent.mkdir(parents=True, exist_ok=True)
    TOKENS_FILE.write_text(json.dumps({
        "access_token":  _access_token,
        "refresh_token": _refresh_token,
        "expiry":        _token_expiry,
    }))


# ── OAuth ─────────────────────────────────────────────────────────────────────

def auth_url(client_id: str) -> str:
    return "https://accounts.spotify.com/authorize?" + urllib.parse.urlencode({
        "client_id":     client_id,
        "response_type": "code",
        "redirect_uri":  REDIRECT_URI,
        "scope":         SCOPES,
    })


async def exchange_code(code: str, client_id: str, client_secret: str) -> None:
    global _access_token, _refresh_token, _token_expiry
    auth = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    async with httpx.AsyncClient(timeout=API_TIMEOUT) as c:
        r = await c.post(
            "https://accounts.spotify.com/api/token",
            headers={"Authorization": f"Basic {auth}",
                     "Content-Type": "application/x-www-form-urlencoded"},
            data={"grant_type": "authorization_code",
                  "code": code, "redirect_uri": REDIRECT_URI},
        )
    d = r.json()
    if not d.get("access_token"):
        print(f"[spotify] auth failed: {d.get('error', d)}", flush=True)
        return
    _access_token  = d["access_token"]
    _refresh_token = d.get("refresh_token", _refresh_token)
    _token_expiry  = time.time() + d.get("expires_in", 3600) - 60
    save_tokens()
    print("[spotify] authenticated", flush=True)


async def _do_refresh(client_id: str, client_secret: str) -> None:
    global _access_token, _token_expiry
    auth = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    async with httpx.AsyncClient(timeout=API_TIMEOUT) as c:
        r = await c.post(
            "https://accounts.spotify.com/api/token",
            headers={"Authorization": f"Basic {auth}",
                     "Content-Type": "application/x-www-form-urlencoded"},
            data={"grant_type": "refresh_token", "refresh_token": _refresh_token},
        )
    d = r.json()
    if not d.get("access_token"):
        print(f"[spotify] token refresh failed: {d.get('error', d)}", flush=True)
        return
    _access_token = d["access_token"]
    _token_expiry = time.time() + d.get("expires_in", 3600) - 60
    save_tokens()


# ── OAuth callback server ─────────────────────────────────────────────────────

_code_future: "asyncio.Future | None" = None


async def _oauth_cb(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        data = await reader.read(2048)
        req  = data.decode(errors="ignore")
        code = None
        if "GET /callback" in req:
            qs = req.split("GET /callback")[1].split(" ")[0]
            for part in qs.lstrip("?").split("&"):
                if part.startswith("code="):
                    code = urllib.parse.unquote(part[5:])
        html = ("<html><body style='font-family:sans-serif;padding:40px;"
                "background:#0d0a18;color:#e0d0ff'>"
                "<h2>MAYA // Spotify connected.</h2>"
                "<p>You can close this tab.</p></body></html>")
        writer.write(
            f"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n"
            f"Content-Length: {len(html)}\r\n\r\n{html}".encode()
        )
        await writer.drain()
        if code and _code_future and not _code_future.done():
            _code_future.set_result(code)
    finally:
        writer.close()


# ── Spotify API helpers ───────────────────────────────────────────────────────

async def _tok(client_id: str, client_secret: str) -> str:
    """Return a valid access token, refreshing under a lock to prevent races."""
    async with _token_lock:
        if time.time() > _token_expiry:
            await _do_refresh(client_id, client_secret)
    return _access_token


def _h(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def _pick_art(images: list) -> "str | None":
    """Return a medium-quality image URL (≈300 px), avoiding tiny thumbnails."""
    if not images:
        return None
    # Spotify returns images largest→smallest; index 1 is usually ~300px
    return images[min(1, len(images) - 1)]["url"]


def _parse_item(item: dict, d: dict) -> "dict | None":
    """Parse a player state 'item' (track or episode) into our track dict."""
    kind = item.get("type", "track")
    ts   = int(time.time() * 1000)

    if kind == "episode":
        show   = item.get("show", {})
        images = item.get("images", []) or show.get("images", [])
        return {
            "type":          "track_update",
            "track":         item["name"],
            "artist":        show.get("name", "Podcast"),
            "album":         show.get("publisher", ""),
            "album_art":     _pick_art(images),
            "duration_ms":   item["duration_ms"],
            "progress_ms":   d.get("progress_ms", 0),
            "is_playing":    d.get("is_playing", False),
            "shuffle_state": d.get("shuffle_state", False),
            "repeat_state":  d.get("repeat_state", "off"),
            "volume":        (d.get("device") or {}).get("volume_percent", 100),
            "track_id":      item["id"],
            "is_episode":    True,
            "timestamp":     ts,
        }

    imgs = item["album"]["images"]
    return {
        "type":          "track_update",
        "track":         item["name"],
        "artist":        ", ".join(a["name"] for a in item["artists"]),
        "album":         item["album"]["name"],
        "album_art":     _pick_art(imgs),
        "duration_ms":   item["duration_ms"],
        "progress_ms":   d.get("progress_ms", 0),
        "is_playing":    d.get("is_playing", False),
        "shuffle_state": d.get("shuffle_state", False),
        "repeat_state":  d.get("repeat_state", "off"),
        "volume":        (d.get("device") or {}).get("volume_percent", 100),
        "track_id":      item["id"],
        "is_episode":    False,
        "timestamp":     ts,
    }


async def fetch_track(cid: str, cs: str) -> "dict | None":
    tok = await _tok(cid, cs)
    async with httpx.AsyncClient(timeout=API_TIMEOUT) as c:
        r = await c.get("https://api.spotify.com/v1/me/player", headers=_h(tok))
    if r.status_code == 429:
        raise httpx.HTTPStatusError("rate limited", request=r.request, response=r)
    if r.status_code in (204, 401, 403) or not r.content:
        return None
    d    = r.json()
    item = d.get("item")
    if not item:
        return None
    return _parse_item(item, d)


async def check_liked(track_id: str, cid: str, cs: str) -> bool:
    tok = await _tok(cid, cs)
    async with httpx.AsyncClient(timeout=API_TIMEOUT) as c:
        r = await c.get(
            f"https://api.spotify.com/v1/me/tracks/contains?ids={track_id}",
            headers=_h(tok),
        )
    return bool(r.is_success and r.json() and r.json()[0])


async def toggle_like(track_id: str, currently_liked: bool, cid: str, cs: str) -> bool:
    tok = await _tok(cid, cs)
    url = f"https://api.spotify.com/v1/me/tracks?ids={track_id}"
    async with httpx.AsyncClient(timeout=API_TIMEOUT) as c:
        if currently_liked:
            await c.delete(url, headers=_h(tok))
            return False
        else:
            await c.put(url, headers=_h(tok))
            return True


async def control(action: str, cid: str, cs: str) -> None:
    tok       = await _tok(cid, cs)
    h         = _h(tok)
    device_id = await _best_device_id(h)
    params    = {"device_id": device_id} if device_id else {}
    async with httpx.AsyncClient(timeout=API_TIMEOUT) as c:
        if action == "play":
            await c.put("https://api.spotify.com/v1/me/player/play", headers=h, params=params)
        elif action == "pause":
            await c.put("https://api.spotify.com/v1/me/player/pause", headers=h, params=params)
        elif action == "next":
            await c.post("https://api.spotify.com/v1/me/player/next", headers=h, params=params)
        elif action == "prev":
            await c.post("https://api.spotify.com/v1/me/player/previous", headers=h, params=params)


async def set_volume(pct: int, cid: str, cs: str) -> None:
    tok = await _tok(cid, cs)
    async with httpx.AsyncClient(timeout=API_TIMEOUT) as c:
        await c.put(
            f"https://api.spotify.com/v1/me/player/volume?volume_percent={pct}",
            headers=_h(tok),
        )


async def seek(pos_ms: int, cid: str, cs: str) -> None:
    tok = await _tok(cid, cs)
    async with httpx.AsyncClient(timeout=API_TIMEOUT) as c:
        await c.put(
            f"https://api.spotify.com/v1/me/player/seek?position_ms={pos_ms}",
            headers=_h(tok),
        )


async def set_shuffle(state: bool, cid: str, cs: str) -> None:
    tok = await _tok(cid, cs)
    async with httpx.AsyncClient(timeout=API_TIMEOUT) as c:
        await c.put(
            f"https://api.spotify.com/v1/me/player/shuffle?state={str(state).lower()}",
            headers=_h(tok),
        )


async def set_repeat(state: str, cid: str, cs: str) -> None:
    tok = await _tok(cid, cs)
    async with httpx.AsyncClient(timeout=API_TIMEOUT) as c:
        await c.put(
            f"https://api.spotify.com/v1/me/player/repeat?state={state}",
            headers=_h(tok),
        )


async def get_queue(cid: str, cs: str) -> list:
    tok = await _tok(cid, cs)
    async with httpx.AsyncClient(timeout=API_TIMEOUT) as c:
        r = await c.get("https://api.spotify.com/v1/me/player/queue", headers=_h(tok))
    if not r.is_success:
        return []
    items = []
    for t in r.json().get("queue", [])[:20]:
        try:
            items.append({
                "uri":        t["uri"],
                "track":      t["name"],
                "artist":     ", ".join(a["name"] for a in t["artists"]),
                "album_art":  _pick_art(t["album"]["images"]),
                "duration_ms": t["duration_ms"],
            })
        except (KeyError, TypeError):
            continue
    return items


async def add_to_queue(uri: str, cid: str, cs: str) -> None:
    tok = await _tok(cid, cs)
    async with httpx.AsyncClient(timeout=API_TIMEOUT) as c:
        await c.post(f"https://api.spotify.com/v1/me/player/queue?uri={uri}", headers=_h(tok))


async def get_playlists(cid: str, cs: str) -> list:
    tok = await _tok(cid, cs)
    async with httpx.AsyncClient(timeout=API_TIMEOUT) as c:
        r = await c.get("https://api.spotify.com/v1/me/playlists?limit=50", headers=_h(tok))
    if not r.is_success:
        return []
    return [
        {
            "uri":   p["uri"],
            "name":  p["name"],
            "count": p["tracks"]["total"],
            "image": _pick_art(p.get("images", [])),
        }
        for p in r.json().get("items", []) if p
    ]


async def play_context(uri: str, cid: str, cs: str) -> None:
    tok       = await _tok(cid, cs)
    h         = _h(tok)
    device_id = await _best_device_id(h)
    params    = {"device_id": device_id} if device_id else {}
    async with httpx.AsyncClient(timeout=API_TIMEOUT) as c:
        r = await c.put(
            "https://api.spotify.com/v1/me/player/play",
            headers=h, params=params, json={"context_uri": uri},
        )
        print(f"[spotify] play_context → {r.status_code}", flush=True)


async def get_recently_played(cid: str, cs: str) -> list:
    tok = await _tok(cid, cs)
    async with httpx.AsyncClient(timeout=API_TIMEOUT) as c:
        r = await c.get(
            "https://api.spotify.com/v1/me/player/recently-played?limit=15",
            headers=_h(tok),
        )
    if not r.is_success:
        return []
    seen, results = set(), []
    for item in r.json().get("items", []):
        t = item.get("track")
        if not t or t["uri"] in seen:
            continue
        seen.add(t["uri"])
        try:
            results.append({
                "uri":        t["uri"],
                "track":      t["name"],
                "artist":     ", ".join(a["name"] for a in t["artists"]),
                "album_art":  _pick_art(t["album"]["images"]),
                "duration_ms": t["duration_ms"],
            })
        except (KeyError, TypeError):
            continue
    return results


async def search_tracks(query: str, cid: str, cs: str) -> list:
    tok = await _tok(cid, cs)
    async with httpx.AsyncClient(timeout=API_TIMEOUT) as c:
        r = await c.get(
            "https://api.spotify.com/v1/search",
            params={"q": query, "type": "track", "limit": 10},
            headers=_h(tok),
        )
    if not r.is_success:
        return []
    results = []
    for i in r.json().get("tracks", {}).get("items", []):
        if not i:
            continue
        try:
            results.append({
                "uri":        i["uri"],
                "track":      i["name"],
                "artist":     ", ".join(a["name"] for a in i["artists"]),
                "album_art":  _pick_art(i["album"]["images"]),
                "duration_ms": i["duration_ms"],
            })
        except (KeyError, TypeError):
            continue
    return results


def _is_local_device(d: dict) -> bool:
    """Prefer local computer/phone devices over third-party (Amazon, Sonos, etc.)."""
    name = d.get("name", "").lower()
    dev_type = d.get("type", "").lower()
    # Amazon Echo / Alexa IDs contain _amzn; Sonos contains _sonos
    dev_id = d.get("id", "")
    if "_amzn_" in dev_id or "_sonos_" in dev_id:
        return False
    if dev_type in ("speaker", "avr", "stb"):
        return False
    return True


async def _best_device_id(h: dict) -> "str | None":
    async with httpx.AsyncClient(timeout=API_TIMEOUT) as c:
        r = await c.get("https://api.spotify.com/v1/me/player/devices", headers=h)
    if not r.is_success:
        return None
    devices = r.json().get("devices", [])
    if not devices:
        return None

    # Prefer: active local device → any active device → any local device → any device
    local = [d for d in devices if _is_local_device(d)]
    active_local  = next((d for d in local    if d.get("is_active")), None)
    active_any    = next((d for d in devices  if d.get("is_active")), None)
    return (active_local or active_any or local[0] if local else devices[0])["id"]


async def play_uri(uri: str, cid: str, cs: str) -> bool:
    """Play a URI. Returns True on success, False on failure."""
    tok       = await _tok(cid, cs)
    h         = _h(tok)
    device_id = await _best_device_id(h)
    params    = {"device_id": device_id} if device_id else {}
    async with httpx.AsyncClient(timeout=API_TIMEOUT) as c:
        r = await c.put(
            "https://api.spotify.com/v1/me/player/play",
            headers=h, params=params, json={"uris": [uri]},
        )
    print(f"[spotify] play_uri → {r.status_code} (device={device_id})", flush=True)
    if r.status_code in (200, 204):
        return True
    print(f"[spotify] error body: {r.text[:300]}", flush=True)
    return False


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
    # Use difference_update (in-place) rather than -= to avoid Python treating
    # _clients as a local variable (augmented assignment makes it local, causing
    # UnboundLocalError on the `if not _clients` read above).
    _clients.difference_update(dead)


# ── Poll loop ─────────────────────────────────────────────────────────────────

def _significant_change(a: "dict | None", b: "dict | None") -> bool:
    if (a is None) != (b is None):
        return True
    if a is None:
        return False
    watch = {"track", "artist", "album", "album_art", "is_playing",
             "shuffle_state", "repeat_state", "volume", "track_id", "is_liked"}
    return any(a.get(k) != b.get(k) for k in watch)


async def _poll(cid: str, cs: str) -> None:
    global _current_track, _prev_track_id
    backoff = POLL_INTERVAL
    first   = True

    while True:
        try:
            track = await fetch_track(cid, cs)
            if first:
                first = False
                status = f"{track['track']} — {track['artist']}" if track else "nothing playing"
                print(f"[spotify] first poll: {status}", flush=True)

            if track:
                tid = track["track_id"]
                if tid != _prev_track_id:
                    track["is_liked"] = await check_liked(tid, cid, cs)
                    _prev_track_id = tid
                elif _current_track:
                    track["is_liked"] = _current_track.get("is_liked", False)

            if _significant_change(track, _current_track):
                if track and _current_track and track.get("track_id") != _current_track.get("track_id"):
                    print(f"[spotify] ♪ {track['track']} — {track['artist']}", flush=True)
                _current_track = track
                await _broadcast(track or {"type": "track_update", "track": None})
            elif track and _current_track:
                # No significant change — but always keep progress + timestamp fresh so
                # new clients connecting mid-song get an accurate position, not a stale one
                # that makes the frontend compute the wrong progress via extrapolation.
                _current_track["progress_ms"] = track["progress_ms"]
                _current_track["timestamp"]   = track["timestamp"]
                _current_track["is_playing"]  = track["is_playing"]

            backoff = POLL_INTERVAL  # reset on success

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                wait = int(e.response.headers.get("Retry-After", 30))
                print(f"[spotify] rate limited — waiting {wait}s", flush=True)
                await asyncio.sleep(wait)
                continue
            backoff = min(backoff * 2, 30)
            print(f"[spotify] HTTP error {e.response.status_code}, backing off {backoff:.0f}s", flush=True)

        except Exception as e:
            backoff = min(backoff * 1.5, 30)
            print(f"[spotify] poll error ({e}), backing off {backoff:.0f}s", flush=True)

        await asyncio.sleep(backoff)


# ── WebSocket handler ─────────────────────────────────────────────────────────

async def handle_client(websocket) -> None:
    _clients.add(websocket)
    cfg = load_config()

    # Push current state on connect
    try:
        if not cfg:
            await websocket.send(json.dumps({"type": "setup_required"}))
        elif not _access_token:
            await websocket.send(json.dumps({"type": "auth_required"}))
        elif _current_track:
            await websocket.send(json.dumps(_current_track))
        else:
            await websocket.send(json.dumps({"type": "track_update", "track": None}))
    except Exception:
        _clients.discard(websocket)
        return

    try:
        async for raw in websocket:
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            if not cfg:
                continue
            cid, cs = cfg["client_id"], cfg["client_secret"]
            t = msg.get("type")

            try:
                if   t == "get_current":
                    await websocket.send(json.dumps(
                        _current_track or {"type": "track_update", "track": None}
                    ))
                elif t == "control":      await control(msg["action"], cid, cs)
                elif t == "volume":       await set_volume(int(msg["value"]), cid, cs)
                elif t == "seek":         await seek(int(msg["position_ms"]), cid, cs)
                elif t == "shuffle":      await set_shuffle(bool(msg["state"]), cid, cs)
                elif t == "repeat":       await set_repeat(msg["state"], cid, cs)
                elif t == "play_uri":     await play_uri(msg["uri"], cid, cs)
                elif t == "play_context": await play_context(msg["uri"], cid, cs)
                elif t == "add_queue":    await add_to_queue(msg["uri"], cid, cs)
                elif t == "toggle_like":
                    new_state = await toggle_like(msg["track_id"], msg["liked"], cid, cs)
                    if _current_track:
                        _current_track["is_liked"] = new_state
                    await websocket.send(json.dumps({"type": "like_update", "is_liked": new_state}))
                elif t == "get_queue":
                    q = await get_queue(cid, cs)
                    await websocket.send(json.dumps({"type": "queue_data", "items": q}))
                elif t == "get_playlists":
                    pl = await get_playlists(cid, cs)
                    await websocket.send(json.dumps({"type": "playlists_data", "items": pl}))
                elif t == "get_recently_played":
                    rp = await get_recently_played(cid, cs)
                    await websocket.send(json.dumps({"type": "recently_played", "items": rp}))
                elif t == "search":
                    results = await search_tracks(msg.get("query", ""), cid, cs)
                    await websocket.send(json.dumps({"type": "search_results", "results": results}))
                elif t == "request_auth":
                    url = auth_url(cid)
                    await websocket.send(json.dumps({"type": "auth_url", "url": url}))
                    webbrowser.open(url)
            except websockets.exceptions.ConnectionClosed:
                break
            except Exception as e:
                print(f"[spotify] handler error ({t}): {e}", flush=True)

    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        _clients.discard(websocket)


# ── Entry point ───────────────────────────────────────────────────────────────

async def run() -> None:
    global _code_future, _token_lock
    _token_lock = asyncio.Lock()

    cfg = load_config()
    if not cfg:
        print(f"[spotify] no config at {CONFIG_FILE}", flush=True)
        async with websockets.serve(handle_client, "127.0.0.1", SPOTIFY_PORT):
            print(f"[spotify] ws://127.0.0.1:{SPOTIFY_PORT} (setup mode)", flush=True)
            await asyncio.Future()
        return

    cid, cs = cfg["client_id"], cfg["client_secret"]

    if not load_tokens():
        print("[spotify] starting OAuth...", flush=True)
        _code_future = asyncio.get_running_loop().create_future()
        srv = await asyncio.start_server(_oauth_cb, "127.0.0.1", CALLBACK_PORT)
        webbrowser.open(auth_url(cid))
        print(f"[spotify] waiting for callback on port {CALLBACK_PORT}", flush=True)
        code = await _code_future
        srv.close()
        await exchange_code(code, cid, cs)

    poll_task = None
    try:
        async with websockets.serve(handle_client, "127.0.0.1", SPOTIFY_PORT):
            print(f"[spotify] ws://127.0.0.1:{SPOTIFY_PORT}", flush=True)
            poll_task = asyncio.ensure_future(_poll(cid, cs))
            await asyncio.Future()
    finally:
        if poll_task and not poll_task.done():
            poll_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await poll_task
