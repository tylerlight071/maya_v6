"""MAYA Notify — phone notifications via ntfy.sh."""
import contextlib
import json
from pathlib import Path

import httpx

CONFIG_FILE = Path(__file__).parent / "models/config.json"
NTFY_BASE   = "https://ntfy.sh"


def _load_topic() -> str | None:
    try:
        if CONFIG_FILE.exists():
            return json.loads(CONFIG_FILE.read_text(encoding="utf-8")).get("ntfy_topic")
    except Exception:
        pass
    return None


def save_topic(topic: str) -> None:
    try:
        CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        existing: dict = {}
        if CONFIG_FILE.exists():
            with contextlib.suppress(Exception):
                existing = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        existing["ntfy_topic"] = topic.strip()
        CONFIG_FILE.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"[notify] save_topic error: {e}", flush=True)


async def send(title: str, body: str, priority: str = "default") -> None:
    """POST a notification to ntfy.sh. Silently no-ops if topic not configured."""
    topic = _load_topic()
    if not topic:
        return
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            await client.post(
                f"{NTFY_BASE}/{topic}",
                json={"title": title, "message": body, "priority": priority},
            )
    except Exception as e:
        print(f"[notify] send error: {e}", flush=True)
