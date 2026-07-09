"""
Shared WebSocket authentication for MAYA's local servers.

Every backend server binds to 127.0.0.1, but localhost alone is not a security
boundary: any process on the machine — including a web page open in a browser,
since WebSockets are exempt from the same-origin policy — can otherwise connect
and drive the assistant. This module gates every connection on a per-session
token that only the Electron shell can read (it lives in models/.session_token,
written by server.py at startup and exposed to the renderer via the preload).

Usage in a handler (first two lines):

    import auth
    async def handle_client(websocket):
        if not await auth.check(websocket):
            return
        ...

The client passes the token as a query parameter: ws://127.0.0.1:PORT?token=...
"""
import secrets
from pathlib import Path
from urllib.parse import urlparse, parse_qs

_TOKEN_FILE = Path(__file__).parent / "models" / ".session_token"


def read_token() -> str:
    """Read the current session token from disk. Empty string if missing."""
    try:
        return _TOKEN_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _request_path(websocket) -> str:
    """Best-effort extraction of the request path across websockets versions."""
    req = getattr(websocket, "request", None)
    if req is not None and getattr(req, "path", None):
        return req.path
    # Fallback for older websockets releases
    return getattr(websocket, "path", "") or ""


def _provided_token(websocket) -> str:
    path = _request_path(websocket)
    if not path:
        return ""
    values = parse_qs(urlparse(path).query).get("token", [])
    return values[0] if values else ""


async def check(websocket) -> bool:
    """
    Validate the connection's session token. On failure, close the socket with
    code 4401 (application-defined "unauthorized") and return False so the
    handler can bail out immediately. Fails closed if no token file exists.
    """
    expected = read_token()
    provided = _provided_token(websocket)

    if expected and provided and secrets.compare_digest(provided, expected):
        return True

    try:
        await websocket.close(code=4401, reason="unauthorized")
    except Exception:
        pass
    return False
