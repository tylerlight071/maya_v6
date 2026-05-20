"""
MAYA Memory System

core.json   — curated facts about Tyler (seeded + auto-updated)
episodes.json — timestamped session summaries extracted by the LLM itself

Extraction runs asynchronously after a session ends, so it never
adds latency to live responses.
"""
import asyncio
import json
from datetime import datetime
from pathlib import Path

import httpx

MEMORY_DIR               = Path(__file__).parent / "memory"
CORE_FILE                = MEMORY_DIR / "core.json"
EPISODES_FILE            = MEMORY_DIR / "episodes.json"
COMPANION_EPISODES_FILE  = MEMORY_DIR / "companion_episodes.json"
LOGS_DIR                 = MEMORY_DIR / "logs"
COMPANION_LOGS_DIR       = MEMORY_DIR / "companion_logs"
MAX_EPISODES  = 60
MAX_LOGS      = 30   # keep ~1 month of daily sessions
EXCERPT_MSGS  = 16   # messages to pull from last session for context

OLLAMA_URL   = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen2.5:14b"


# ── Load / save ───────────────────────────────────────────────────────────────

def load_core() -> dict:
    MEMORY_DIR.mkdir(exist_ok=True)
    if not CORE_FILE.exists():
        return {}
    try:
        return json.loads(CORE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_core(data: dict) -> None:
    MEMORY_DIR.mkdir(exist_ok=True)
    CORE_FILE.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def load_episodes(companion: bool = False) -> list:
    MEMORY_DIR.mkdir(exist_ok=True)
    path = COMPANION_EPISODES_FILE if companion else EPISODES_FILE
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _save_episodes(episodes: list, companion: bool = False) -> None:
    MEMORY_DIR.mkdir(exist_ok=True)
    path = COMPANION_EPISODES_FILE if companion else EPISODES_FILE
    path.write_text(json.dumps(episodes, indent=2, ensure_ascii=False), encoding="utf-8")


# ── Session log persistence ───────────────────────────────────────────────────

def save_log(history: list[dict], companion: bool = False) -> None:
    """Persist the raw conversation from a session to disk."""
    if len(history) < 2:
        return
    logs_dir = COMPANION_LOGS_DIR if companion else LOGS_DIR
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"{ts}.json"
    log_path.write_text(
        json.dumps({"timestamp": datetime.now().isoformat(), "messages": history},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    all_logs = sorted(logs_dir.glob("*.json"))
    for old in all_logs[:-MAX_LOGS]:
        old.unlink(missing_ok=True)


def _read_last_excerpt(logs_dir: Path) -> str:
    logs_dir.mkdir(parents=True, exist_ok=True)
    logs = sorted(logs_dir.glob("*.json"))
    if not logs:
        return ""
    try:
        data     = json.loads(logs[-1].read_text(encoding="utf-8"))
        messages = data.get("messages", [])
        recent   = messages[-EXCERPT_MSGS:]
        lines    = []
        for msg in recent:
            role    = msg.get("role", "")
            content = msg.get("content", "").strip()
            if not content:
                continue
            if role == "user":
                lines.append(f"Tyler: {content}")
            elif role == "assistant":
                lines.append(f"MAYA: {content}")
        return "\n".join(lines)
    except Exception:
        return ""


def load_last_session_excerpt(companion: bool = False) -> str:
    """Return the tail of the most recent session log as formatted dialogue."""
    return _read_last_excerpt(COMPANION_LOGS_DIR if companion else LOGS_DIR)


# ── Context formatting ────────────────────────────────────────────────────────

def format_context(companion: bool = False) -> str:
    """
    Returns the full memory block for injection into the system prompt.
    companion=True: includes companion episodes + last companion session (one-way enrichment).
    """
    core            = load_core()
    default_eps     = load_episodes(companion=False)
    companion_eps   = load_episodes(companion=True) if companion else []
    excerpt         = load_last_session_excerpt(companion=companion)

    sections: list[str] = []

    if core:
        all_facts: list[str] = []
        for items in core.values():
            if isinstance(items, list):
                all_facts.extend(items)
            elif isinstance(items, str):
                all_facts.append(items)
        if all_facts:
            sections.append("WHAT I KNOW ABOUT TYLER:\n" +
                            "\n".join(f"- {f}" for f in all_facts))

    if default_eps:
        recent = [ep["summary"] for ep in default_eps[-5:]]
        sections.append("RECENT SESSIONS:\n" +
                        "\n".join(f"- {s}" for s in reversed(recent)))

    if companion_eps:
        recent_c = [ep["summary"] for ep in companion_eps[-5:]]
        sections.append("RECENT COMPANION SESSIONS:\n" +
                        "\n".join(f"- {s}" for s in reversed(recent_c)))

    if excerpt:
        sections.append(
            "LAST CONVERSATION (use naturally — reference it if relevant, "
            "ask to continue unfinished things):\n" + excerpt
        )

    return "\n\n".join(sections)


# ── Post-session extraction ───────────────────────────────────────────────────

_EXTRACT_PROMPT = """\
A conversation just ended. Review it and extract ONLY genuinely useful long-term memory.

Extraction criteria — only keep facts that:
  - Reveal a preference, habit, or constraint that affects future interactions
  - Update the status of an ongoing project or goal
  - Correct or add precision to something already believed
  - Are still relevant weeks from now

Do NOT extract:
  - The content of what was discussed (only meta-learning about Tyler)
  - Things already obvious or in the current memory
  - One-off requests with no lasting relevance

Respond with valid JSON only:
{{
  "new_facts": [
    {{"category": "personal|technical|preference|project", "fact": "one concise line"}}
  ],
  "session_summary": "one sentence: what this session was about"
}}

If nothing new is worth remembering, return:
{{"new_facts": [], "session_summary": "..."}}

Conversation:
{conversation}

Current memory (don't re-extract what's already here):
{current_memory}
"""


async def extract_and_save(session_history: list[dict], companion: bool = False) -> None:
    """
    Called asynchronously after a session ends.
    Uses the LLM to decide what's worth keeping — runs in the background,
    never blocks live responses.
    """
    if len(session_history) < 4:   # need at least 2 real exchanges
        return

    conv = "\n".join(
        f"{m['role'].upper()}: {m['content']}"
        for m in session_history
        if m.get("role") in ("user", "assistant")
    )
    current = json.dumps(load_core(), indent=2, ensure_ascii=False)

    prompt = _EXTRACT_PROMPT.format(conversation=conv, current_memory=current)

    try:
        async with httpx.AsyncClient(timeout=45) as client:
            r = await client.post(OLLAMA_URL, json={
                "model":   OLLAMA_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream":  False,
                "format":  "json",
            })
        if not r.is_success:
            return

        data    = json.loads(r.json().get("message", {}).get("content", "{}"))
        facts   = data.get("new_facts", [])
        summary = data.get("session_summary", "").strip()

        # Save session summary as an episode (companion or default)
        if summary:
            eps = load_episodes(companion=companion)
            eps.append({"timestamp": datetime.now().isoformat(), "summary": summary})
            if len(eps) > MAX_EPISODES:
                eps = eps[-MAX_EPISODES:]
            _save_episodes(eps, companion=companion)

        # Merge new facts into core.json — companion facts go to core too
        # (they're about Tyler, not the session content)
        if facts and not companion:
            core = load_core()
            for item in facts:
                cat  = item.get("category", "general")
                fact = item.get("fact", "").strip()
                if not fact:
                    continue
                if cat not in core:
                    core[cat] = []
                if isinstance(core[cat], list) and fact not in core[cat]:
                    core[cat].append(fact)
            save_core(core)
            print(f"[memory] +{len(facts)} fact(s), session noted", flush=True)

    except Exception as e:
        print(f"[memory] extraction error: {e}", flush=True)
