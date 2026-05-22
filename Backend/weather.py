"""
MAYA Weather — real-time conditions and forecasts via Open-Meteo (no API key needed).
Location is auto-detected on startup via ip-api.com, then cached.
"""
import asyncio
import contextlib
import json
import time
from pathlib import Path

import httpx
import websockets

WEATHER_PORT  = 8768
POLL_INTERVAL = 300   # seconds between full refreshes (5 min)

# WMO 4677 weather interpretation codes → (description, icon char)
WMO = {
    0:  ("Clear",            "☀"),
    1:  ("Mostly clear",     "🌤"),
    2:  ("Partly cloudy",    "⛅"),
    3:  ("Overcast",         "☁"),
    45: ("Foggy",            "🌫"),
    48: ("Freezing fog",     "🌫"),
    51: ("Light drizzle",    "🌦"),
    53: ("Drizzle",          "🌦"),
    55: ("Heavy drizzle",    "🌧"),
    56: ("Freezing drizzle", "🌧"),
    57: ("Freezing drizzle", "🌧"),
    61: ("Light rain",       "🌧"),
    63: ("Rain",             "🌧"),
    65: ("Heavy rain",       "🌧"),
    66: ("Freezing rain",    "🌨"),
    67: ("Freezing rain",    "🌨"),
    71: ("Light snow",       "🌨"),
    73: ("Snow",             "❄"),
    75: ("Heavy snow",       "❄"),
    77: ("Snow grains",      "❄"),
    80: ("Rain showers",     "🌦"),
    81: ("Rain showers",     "🌧"),
    82: ("Heavy showers",    "🌧"),
    85: ("Snow showers",     "🌨"),
    86: ("Heavy snow showers","❄"),
    95: ("Thunderstorm",     "⛈"),
    96: ("Thunderstorm",     "⛈"),
    99: ("Thunderstorm",     "⛈"),
}

_clients: set       = set()
_current_weather    = None
_location: dict     = {}

LOCATION_FILE = Path(__file__).parent / "models/weather_location.json"


def _safe_get(lst: list, i: int, default=None):
    return lst[i] if lst and i < len(lst) else default


def _wind_dir(deg: float) -> str:
    dirs = ["N","NNE","NE","ENE","E","ESE","SE","SSE","S","SSW","SW","WSW","W","WNW","NW","NNW"]
    return dirs[round(deg / 22.5) % 16]


# ── Location ──────────────────────────────────────────────────────────────────

async def detect_location() -> dict:
    """
    Return the current location. Priority:
      1. Pinned cache (pinned=True) — never overwritten by IP detection
      2. Unpinned cache — used if present, may be overwritten by IP on next startup
      3. IP geolocation fallback — only runs if no valid cache exists
    """
    if LOCATION_FILE.exists():
        try:
            cached = json.loads(LOCATION_FILE.read_text(encoding="utf-8"))
            lat, lon = cached.get("lat"), cached.get("lon")
            if lat and lon:
                pinned = cached.get("pinned", False)
                pin_tag = " [PINNED]" if pinned else ""
                print(f"[weather] cached location: {cached.get('city')}, "
                      f"{cached.get('region', '')} {cached.get('country', '')}{pin_tag}", flush=True)
                # Pinned locations are never overwritten — return immediately
                if pinned:
                    return cached
                # Unpinned: use the cache but fall through to refresh below
                return cached
            print("[weather] cache missing coordinates — falling back to IP", flush=True)
        except Exception as e:
            print(f"[weather] cache read error ({e}) — falling back to IP", flush=True)
    else:
        print(f"[weather] no location cache — detecting via IP", flush=True)

    try:
        async with httpx.AsyncClient(timeout=8) as c:
            r = await c.get("http://ip-api.com/json")
        if r.is_success:
            d = r.json()
            if d.get("status") == "success":
                loc = {
                    "lat":     d["lat"],
                    "lon":     d["lon"],
                    "city":    d.get("city", "Unknown"),
                    "country": d.get("countryCode", ""),
                    "region":  d.get("regionName", ""),
                    "pinned":  False,   # IP-detected locations are never pinned
                }
                LOCATION_FILE.parent.mkdir(parents=True, exist_ok=True)
                LOCATION_FILE.write_text(json.dumps(loc), encoding="utf-8")
                return loc
    except Exception as e:
        print(f"[weather] geolocation failed ({e}) — location unavailable, using London as placeholder", flush=True)

    # Fallback — marked so the system prompt can note the location is unknown
    return {"lat": 51.5074, "lon": -0.1278, "city": "Unknown", "country": "??",
            "region": "", "pinned": False, "fallback": True}


# ── Fetch ─────────────────────────────────────────────────────────────────────

async def fetch_weather(lat: float, lon: float) -> "dict | None":
    try:
        params = {
            "latitude":  lat,
            "longitude": lon,
            "current": ",".join([
                "temperature_2m", "relative_humidity_2m", "apparent_temperature",
                "is_day", "precipitation", "weather_code", "cloud_cover",
                "wind_speed_10m", "wind_direction_10m", "uv_index", "visibility",
            ]),
            "hourly": ",".join([
                "temperature_2m", "precipitation_probability",
                "weather_code", "wind_speed_10m", "is_day",
            ]),
            "daily": ",".join([
                "weather_code", "temperature_2m_max", "temperature_2m_min",
                "precipitation_probability_max", "wind_speed_10m_max",
                "sunrise", "sunset",
            ]),
            "wind_speed_unit":  "mph",
            "temperature_unit": "celsius",
            "timezone":         "auto",
            "forecast_days":    7,
        }
        async with httpx.AsyncClient(timeout=15) as c:
            r = await c.get("https://api.open-meteo.com/v1/forecast", params=params)
        if not r.is_success:
            return None

        d    = r.json()
        cur  = d.get("current", {})
        hrly = d.get("hourly", {})
        dly  = d.get("daily", {})

        code        = cur.get("weather_code", 0)
        desc, icon  = WMO.get(code, ("Unknown", "?"))
        wind_speed  = cur.get("wind_speed_10m", 0)
        wind_deg    = cur.get("wind_direction_10m", 0)
        visibility  = cur.get("visibility", 0)

        # Hourly: next 24 entries (data starts at midnight local time)
        h_times  = hrly.get("time", [])
        h_temps  = hrly.get("temperature_2m", [])
        h_precip = hrly.get("precipitation_probability", [])
        h_code   = hrly.get("weather_code", [])
        h_wind   = hrly.get("wind_speed_10m", [])
        h_isday  = hrly.get("is_day", [])

        # Find current hour index using current.time
        cur_time = cur.get("time", "")
        start_i  = 0
        for i, t in enumerate(h_times):
            if t >= cur_time:
                start_i = i
                break

        hourly = []
        for i in range(start_i, min(start_i + 24, len(h_times))):
            hc = _safe_get(h_code, i, 0)
            _, hicon = WMO.get(hc, ("", "?"))
            hourly.append({
                "time":    h_times[i],
                "temp":    round(_safe_get(h_temps, i, 0), 1),
                "precip":  _safe_get(h_precip, i, 0),
                "code":    hc,
                "icon":    hicon,
                "wind":    round(_safe_get(h_wind, i, 0), 1),
                "is_day":  bool(_safe_get(h_isday, i, 1)),
            })

        # Daily: 7 days
        d_times   = dly.get("time", [])
        d_code    = dly.get("weather_code", [])
        d_max     = dly.get("temperature_2m_max", [])
        d_min     = dly.get("temperature_2m_min", [])
        d_precip  = dly.get("precipitation_probability_max", [])
        d_wind    = dly.get("wind_speed_10m_max", [])
        d_sunrise = dly.get("sunrise", [])
        d_sunset  = dly.get("sunset", [])

        daily = []
        for i in range(len(d_times)):
            dc = _safe_get(d_code, i, 0)
            ddesc, dicon = WMO.get(dc, ("Unknown", "?"))
            daily.append({
                "date":    d_times[i],
                "desc":    ddesc,
                "icon":    dicon,
                "max":     round(_safe_get(d_max, i, 0), 1),
                "min":     round(_safe_get(d_min, i, 0), 1),
                "precip":  _safe_get(d_precip, i, 0),
                "wind":    round(_safe_get(d_wind, i, 0), 1),
                "sunrise": _safe_get(d_sunrise, i, ""),
                "sunset":  _safe_get(d_sunset, i, ""),
            })

        return {
            "type":         "weather_update",
            "temp":         round(cur.get("temperature_2m", 0), 1),
            "feels_like":   round(cur.get("apparent_temperature", 0), 1),
            "humidity":     cur.get("relative_humidity_2m", 0),
            "wind_speed":   round(wind_speed, 1),
            "wind_dir":     _wind_dir(wind_deg),
            "wind_deg":     wind_deg,
            "cloud_cover":  cur.get("cloud_cover", 0),
            "uv_index":     round(cur.get("uv_index", 0), 1),
            "visibility_km": round(visibility / 1000, 1),
            "precipitation": cur.get("precipitation", 0),
            "code":         code,
            "description":  desc,
            "icon":         icon,
            "is_day":       bool(cur.get("is_day", 1)),
            "city":         _location.get("city", ""),
            "country":      _location.get("country", ""),
            "hourly":       hourly,
            "daily":        daily,
            "timestamp":    int(time.time() * 1000),
        }

    except Exception as e:
        print(f"[weather] fetch error: {e}", flush=True)
        return None


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


# ── Poll loop ─────────────────────────────────────────────────────────────────

async def _poll() -> None:
    global _current_weather
    while True:
        try:
            w = await fetch_weather(_location["lat"], _location["lon"])
            if w:
                _current_weather = w
                await _broadcast(w)
                print(f"[weather] {w['icon']} {w['description']} {w['temp']}°C  "
                      f"({w['city']})", flush=True)
        except Exception as e:
            print(f"[weather] poll error: {e}", flush=True)
        await asyncio.sleep(POLL_INTERVAL)


async def set_location(query: str) -> "dict | None":
    """Geocode a place name and update the location. Returns new location dict or None.
    Pass 'auto' or 'detect' to clear the pin and re-detect via IP."""
    global _location, _current_weather
    # Unpin: let IP detection take over again
    if query.strip().lower() in ("auto", "detect", "automatic"):
        if LOCATION_FILE.exists():
            LOCATION_FILE.unlink(missing_ok=True)
        _location = await detect_location()
        _current_weather = await fetch_weather(_location["lat"], _location["lon"])
        return _location
    try:
        async with httpx.AsyncClient(timeout=8) as c:
            r = await c.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={"name": query, "count": 1, "language": "en", "format": "json"},
            )
        results = r.json().get("results", [])
        if not results:
            return None
        res = results[0]
        loc = {
            "lat":     res["latitude"],
            "lon":     res["longitude"],
            "city":    res["name"],
            "country": res.get("country_code", ""),
            "region":  res.get("admin2", res.get("admin1", "")),
            "pinned":  True,   # explicitly set by user — never overwritten by IP detection
        }
        LOCATION_FILE.parent.mkdir(parents=True, exist_ok=True)
        LOCATION_FILE.write_text(json.dumps(loc), encoding="utf-8")
        _location = loc
        print(f"[weather] location updated → {loc['city']}, {loc.get('region','')}, {loc['country']}", flush=True)

        # Immediately fetch and broadcast fresh weather for the new location
        w = await fetch_weather(loc["lat"], loc["lon"])
        if w:
            _current_weather = w
            await _broadcast(w)
        return loc
    except Exception as e:
        print(f"[weather] set_location error: {e}", flush=True)
        return None


# ── WebSocket handler ─────────────────────────────────────────────────────────

async def handle_client(websocket) -> None:
    _clients.add(websocket)
    try:
        if _current_weather:
            await websocket.send(json.dumps(_current_weather))
        else:
            await websocket.send(json.dumps({"type": "weather_loading"}))
        async for raw in websocket:
            pass   # no client→server messages needed yet
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        _clients.discard(websocket)


# ── Entry point ───────────────────────────────────────────────────────────────

async def run() -> None:
    global _location, _current_weather

    print("[weather] detecting location...", flush=True)
    _location = await detect_location()
    print(f"[weather] {_location['city']}, {_location['country']}", flush=True)

    _current_weather = await fetch_weather(_location["lat"], _location["lon"])
    if _current_weather:
        print(f"[weather] initial: {_current_weather['icon']} "
              f"{_current_weather['description']} {_current_weather['temp']}°C", flush=True)

    poll_task = None
    try:
        async with websockets.serve(handle_client, "127.0.0.1", WEATHER_PORT):
            print(f"[weather] ws://127.0.0.1:{WEATHER_PORT}", flush=True)
            poll_task = asyncio.create_task(_poll())
            await asyncio.Future()
    finally:
        if poll_task and not poll_task.done():
            poll_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await poll_task
