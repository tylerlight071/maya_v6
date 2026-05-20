"""MAYA Calendar — Google Calendar via OAuth2, supporting multiple accounts."""
import asyncio
import contextlib
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

import websockets

CALENDAR_PORT = 8774
CREDS_FILE    = Path(__file__).parent / "models/google_credentials.json"
SCOPES        = ["https://www.googleapis.com/auth/calendar"]
LOCAL_TZ      = "Europe/London"

# One token file per account
TOKENS_FILES  = [
    Path(__file__).parent / "models/google_tokens_1.json",
    Path(__file__).parent / "models/google_tokens_2.json",
]

_clients:         set  = set()
_services:        list = []   # one google API service per authenticated account
_today_events:    list = []
_upcoming_events: list = []


# ── Auth ──────────────────────────────────────────────────────────────────────

def _build_service_from_token(token_file: Path):
    """Load a service from a saved token. Returns service or None."""
    try:
        from google.oauth2.credentials      import Credentials
        from google.auth.transport.requests import Request
        from googleapiclient.discovery      import build

        if not token_file.exists():
            return None

        creds = Credentials.from_authorized_user_file(str(token_file), SCOPES)
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
            token_file.write_text(creds.to_json(), encoding="utf-8")

        if creds and creds.valid:
            return build("calendar", "v3", credentials=creds)
    except Exception as e:
        print(f"[calendar] load {token_file.name} error: {e}", flush=True)
    return None


def _build_services() -> None:
    global _services
    _services = []
    for i, tf in enumerate(TOKENS_FILES, start=1):
        svc = _build_service_from_token(tf)
        if svc:
            _services.append(svc)
            print(f"[calendar] account {i} loaded from {tf.name}", flush=True)
    if not _services:
        print('[calendar] no accounts authenticated — run auth_calendar.py', flush=True)


def authenticate(account: int = 1) -> str:
    """Run the OAuth2 flow for the given account number (1 or 2).
    Blocking — call from executor."""
    idx = account - 1
    if idx < 0 or idx >= len(TOKENS_FILES):
        return f"Invalid account number: {account}"
    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build

        if not CREDS_FILE.exists():
            return "google_credentials.json not found in Backend/models/"

        flow  = InstalledAppFlow.from_client_secrets_file(str(CREDS_FILE), SCOPES)
        creds = flow.run_local_server(port=0, open_browser=True)
        TOKENS_FILES[idx].write_text(creds.to_json(), encoding="utf-8")
        svc = build("calendar", "v3", credentials=creds)

        if idx < len(_services):
            _services[idx] = svc
        else:
            while len(_services) <= idx:
                _services.append(None)
            _services[idx] = svc

        _refresh_cache()
        return f"Account {account} authenticated."
    except Exception as e:
        return f"auth error: {e}"


# ── Data helpers ──────────────────────────────────────────────────────────────

def _fmt(e: dict) -> dict:
    start = e.get("start", {})
    end   = e.get("end",   {})
    return {
        "id":          e.get("id",       ""),
        "title":       e.get("summary",  "(no title)"),
        "start":       start.get("dateTime", start.get("date", "")),
        "end":         end.get("dateTime",   end.get("date",   "")),
        "location":    e.get("location",    ""),
        "description": e.get("description", ""),
    }


def _query_all(time_min: str, time_max: str, max_per_cal: int = 50) -> list:
    """Fetch events from all authenticated accounts, merge and sort."""
    all_events: list = []
    for svc in _services:
        if not svc:
            continue
        try:
            # Get all non-holiday calendars for this account
            cal_list   = svc.calendarList().list().execute()
            cal_ids    = [c["id"] for c in cal_list.get("items", [])
                          if "holiday" not in c.get("id", "").lower()
                          or not any("holiday" in x.get("id","").lower()
                                     for x in cal_list.get("items",[])
                                     if x["id"] != c["id"])]
            seen_hol   = False
            for cal_id in cal_ids:
                if "holiday" in cal_id.lower():
                    if seen_hol:
                        continue
                    seen_hol = True
                try:
                    result = svc.events().list(
                        calendarId=cal_id, timeMin=time_min, timeMax=time_max,
                        singleEvents=True, orderBy="startTime", maxResults=max_per_cal,
                    ).execute()
                    all_events.extend(_fmt(e) for e in result.get("items", []))
                except Exception:
                    pass
        except Exception as e:
            print(f"[calendar] query error: {e}", flush=True)

    # Sort merged results by start time, deduplicate by id
    seen: set = set()
    unique    = []
    for e in sorted(all_events, key=lambda x: x["start"]):
        if e["id"] not in seen:
            seen.add(e["id"])
            unique.append(e)
    return unique


def get_today_events() -> list:
    now   = datetime.now(timezone.utc)
    start = now.replace(hour=0,  minute=0,  second=0,  microsecond=0).isoformat()
    end   = now.replace(hour=23, minute=59, second=59, microsecond=0).isoformat()
    return _query_all(start, end)


def get_upcoming_events(days: int = 7) -> list:
    now = datetime.now(timezone.utc)
    end = now + timedelta(days=days)
    return _query_all(now.isoformat(), end.isoformat(), max_per_cal=25)


def get_events_for_date(date_str: str) -> list:
    day   = datetime.fromisoformat(date_str).replace(tzinfo=timezone.utc)
    start = day.replace(hour=0,  minute=0,  second=0).isoformat()
    end   = day.replace(hour=23, minute=59, second=59).isoformat()
    return _query_all(start, end)


def create_event(title: str, start: str, end: str, description: str = "") -> str:
    svc = _services[0] if _services else None
    if not svc:
        return "Calendar not connected."
    try:
        body: dict = {
            "summary": title,
            "start":   {"dateTime": start, "timeZone": LOCAL_TZ},
            "end":     {"dateTime": end,   "timeZone": LOCAL_TZ},
        }
        if description:
            body["description"] = description
        result = svc.events().insert(calendarId="primary", body=body).execute()
        _refresh_cache()
        return result.get("id", "created")
    except Exception as e:
        return f"Failed: {e}"


def update_event(event_id: str, title: str = "", start: str = "",
                 end: str = "", description: str = "") -> str:
    # Try each account's service to find which one owns this event
    for svc in _services:
        if not svc:
            continue
        try:
            event = svc.events().get(calendarId="primary", eventId=event_id).execute()
            if title:       event["summary"]     = title
            if description: event["description"] = description
            if start:       event["start"]       = {"dateTime": start, "timeZone": LOCAL_TZ}
            if end:         event["end"]         = {"dateTime": end,   "timeZone": LOCAL_TZ}
            svc.events().update(calendarId="primary", eventId=event_id, body=event).execute()
            _refresh_cache()
            return "Updated."
        except Exception:
            continue
    return "Event not found across any connected account."


def delete_event(event_id: str) -> str:
    for svc in _services:
        if not svc:
            continue
        try:
            svc.events().delete(calendarId="primary", eventId=event_id).execute()
            _refresh_cache()
            return "Deleted."
        except Exception:
            continue
    return "Event not found across any connected account."


def format_today_for_greeting() -> str:
    events = _today_events
    if not events:
        return ""
    parts = []
    for e in events[:3]:
        t = e["start"].split("T")[1][:5] if "T" in e["start"] else ""
        parts.append(f"{e['title']}{' at ' + t if t else ''}")
    return "Today's calendar: " + "; ".join(parts) + "."


# ── Internal ──────────────────────────────────────────────────────────────────

def _refresh_cache() -> None:
    global _today_events, _upcoming_events
    _today_events    = get_today_events()
    _upcoming_events = get_upcoming_events()


async def _broadcast(msg: dict) -> None:
    raw = json.dumps(msg)
    for ws in list(_clients):
        with contextlib.suppress(Exception):
            await ws.send(raw)


async def _push_update() -> None:
    await _broadcast({
        "type":      "calendar_update",
        "today":     _today_events,
        "upcoming":  _upcoming_events,
        "connected": bool(_services),
    })


# ── WebSocket handler ─────────────────────────────────────────────────────────

async def handle_client(websocket) -> None:
    _clients.add(websocket)
    try:
        if _services:
            await websocket.send(json.dumps({
                "type":      "calendar_update",
                "today":     _today_events,
                "upcoming":  _upcoming_events,
                "connected": True,
            }))
        else:
            await websocket.send(json.dumps({"type": "auth_required"}))
        async for _ in websocket:
            pass
    except Exception:
        pass
    finally:
        _clients.discard(websocket)


async def run() -> None:
    _build_services()
    if _services:
        _refresh_cache()
        print(f"[calendar] {len(_services)} account(s) connected — {len(_today_events)} event(s) today", flush=True)

    async def _poll():
        while True:
            await asyncio.sleep(300)
            if _services:
                _refresh_cache()
                await _push_update()

    asyncio.ensure_future(_poll())
    async with websockets.serve(handle_client, "127.0.0.1", CALENDAR_PORT):
        print(f"[calendar] ws://127.0.0.1:{CALENDAR_PORT}", flush=True)
        await asyncio.Future()
