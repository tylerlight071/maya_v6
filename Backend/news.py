"""MAYA News — RSS aggregator WebSocket server (port 8775).

Fetches headlines from multiple RSS feeds, caches them, and pushes updates
to connected clients every 15 minutes or on demand.
"""
import asyncio
import contextlib
import json
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

import httpx
import websockets

NEWS_PORT    = 8775
REFRESH_SECS = 900   # 15 minutes
FEEDS_FILE   = Path(__file__).parent / "models/news_feeds.json"

# RSS feeds — loaded from models/news_feeds.json; fallback to hardcoded list
def _load_feeds() -> list:
    if FEEDS_FILE.exists():
        try:
            data = json.loads(FEEDS_FILE.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return [tuple(f) for f in data if len(f) == 3]
        except Exception as e:
            print(f"[news] news_feeds.json load error: {e}", flush=True)
    return [
        ("BBC News",     "http://feeds.bbci.co.uk/news/rss.xml",           "top"),
        ("Sky News",     "https://feeds.skynews.com/feeds/rss/home.xml",   "top"),
        ("The Guardian", "https://www.theguardian.com/world/rss",          "world"),
        ("Reuters",      "https://feeds.reuters.com/reuters/topNews",      "top"),
        ("TechCrunch",   "https://techcrunch.com/feed/",                   "tech"),
        ("Ars Technica", "https://feeds.arstechnica.com/arstechnica/index","tech"),
    ]

FEEDS = _load_feeds()

_clients:  set  = set()
_articles: list = []   # [{title, url, summary, source, category, published_ts}]
_last_fetch: float = 0


# ── HTML / text helpers ───────────────────────────────────────────────────────

def _strip_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&amp;",  "&", text)
    text = re.sub(r"&lt;",   "<", text)
    text = re.sub(r"&gt;",   ">", text)
    text = re.sub(r"&quot;", '"', text)
    text = re.sub(r"&#39;",  "'", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"\s+",    " ", text)
    return text.strip()


def _parse_date(date_str: str) -> float:
    """Parse RSS pubDate (RFC 2822) → unix timestamp. Falls back to now."""
    try:
        return parsedate_to_datetime(date_str).timestamp()
    except Exception:
        pass
    try:
        # ISO 8601 fallback
        return datetime.fromisoformat(date_str.rstrip("Z")).replace(
            tzinfo=timezone.utc).timestamp()
    except Exception:
        return time.time()


def _parse_rss(xml_text: str, source: str, category: str) -> list[dict]:
    articles = []
    try:
        root = ET.fromstring(xml_text)
        items = root.findall(".//item")
        if not items:
            # Atom feed fallback
            ns = {"a": "http://www.w3.org/2005/Atom"}
            items = root.findall(".//a:entry", ns)
        for item in items[:20]:
            def t(tag: str) -> str:
                el = item.find(tag)
                return (el.text or "").strip() if el is not None else ""

            title   = _strip_html(t("title"))
            url     = t("link") or t("guid")
            # Atom <link href="...">
            if not url:
                el = item.find("{http://www.w3.org/2005/Atom}link")
                if el is not None:
                    url = el.get("href", "")
            summary = _strip_html(t("description") or t("{http://www.w3.org/2005/Atom}summary"))
            pub     = t("pubDate") or t("updated") or t("{http://www.w3.org/2005/Atom}updated")

            if not title or not url:
                continue
            articles.append({
                "title":        title,
                "url":          url.strip(),
                "summary":      summary[:220],
                "source":       source,
                "category":     category,
                "published_ts": _parse_date(pub),
            })
    except Exception as e:
        print(f"[news] parse error ({source}): {e}", flush=True)
    return articles


# ── Fetch ─────────────────────────────────────────────────────────────────────

async def _fetch_feed(client: httpx.AsyncClient, name: str, url: str, category: str) -> list[dict]:
    try:
        r = await client.get(url, timeout=10, follow_redirects=True)
        if r.is_success:
            return _parse_rss(r.text, name, category)
    except Exception as e:
        print(f"[news] fetch error ({name}): {e}", flush=True)
    return []


async def refresh() -> None:
    global _articles, _last_fetch
    print("[news] fetching feeds...", flush=True)
    all_articles = []
    async with httpx.AsyncClient(headers={"User-Agent": "Mozilla/5.0"}) as client:
        tasks = [_fetch_feed(client, name, url, cat) for name, url, cat in FEEDS]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for (name, _, _cat), r in zip(FEEDS, results):
            if isinstance(r, Exception):
                print(f"[news] fetch error ({name}): {r}", flush=True)
            elif isinstance(r, list):
                all_articles.extend(r)

    # Deduplicate by URL, sort newest first
    seen: set = set()
    unique = []
    for a in sorted(all_articles, key=lambda x: x["published_ts"], reverse=True):
        if a["url"] not in seen:
            seen.add(a["url"])
            unique.append(a)

    _articles   = unique[:60]   # keep top 60
    _last_fetch = time.time()
    print(f"[news] {len(_articles)} articles loaded", flush=True)
    await _broadcast({"type": "news_data", "articles": _articles})


async def _refresh_loop() -> None:
    while True:
        with contextlib.suppress(Exception):
            await refresh()
        await asyncio.sleep(REFRESH_SECS)


# ── Broadcast ─────────────────────────────────────────────────────────────────

async def _broadcast(msg: dict) -> None:
    if not _clients:
        return
    raw  = json.dumps(msg)
    dead = set()
    for ws in list(_clients):
        try:
            await ws.send(raw)
        except Exception as e:
            print(f"[news] broadcast error (client dropped): {e}", flush=True)
            dead.add(ws)
    _clients.difference_update(dead)


# ── WebSocket handler ─────────────────────────────────────────────────────────

async def handle_client(websocket) -> None:
    import auth
    if not await auth.check(websocket):
        return
    _clients.add(websocket)
    try:
        # Send current cache immediately
        await websocket.send(json.dumps({"type": "news_data", "articles": _articles}))
        async for raw in websocket:
            with contextlib.suppress(Exception):
                msg = json.loads(raw)
                if msg.get("type") == "refresh":
                    await refresh()
    except Exception:
        pass
    finally:
        _clients.discard(websocket)


async def run() -> None:
    refresh_task = asyncio.create_task(_refresh_loop())
    try:
        async with websockets.serve(handle_client, "127.0.0.1", NEWS_PORT):
            print(f"[news] ws://127.0.0.1:{NEWS_PORT}", flush=True)
            await asyncio.Future()
    finally:
        refresh_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await refresh_task
