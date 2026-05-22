"""
MAYA Memory System — v2

Tiers:
  core.json               — curated facts (permanent + temporary with TTL)
  episodes.json           — session summaries with bullets, tags, in_progress
  episode_embeddings.json — cached nomic-embed-text vectors
  narrative_profile.txt   — periodic narrative about Tyler (every 20 sessions)
  projects.json           — named project tracker
  logs/                   — raw session logs
  core.backup.{1-3}.json  — rolling pre-consolidation backups
"""
import json
import math
import shutil
from collections import defaultdict
from datetime import datetime, date, timedelta
from pathlib import Path

import httpx

# ── Constants ─────────────────────────────────────────────────────────────────

MEMORY_DIR              = Path(__file__).parent / "memory"
CORE_FILE               = MEMORY_DIR / "core.json"
EPISODES_FILE           = MEMORY_DIR / "episodes.json"
COMPANION_EPISODES_FILE = MEMORY_DIR / "companion_episodes.json"
LOGS_DIR                = MEMORY_DIR / "logs"
COMPANION_LOGS_DIR      = MEMORY_DIR / "companion_logs"
EPISODE_EMBEDDINGS_FILE = MEMORY_DIR / "episode_embeddings.json"
NARRATIVE_FILE          = MEMORY_DIR / "narrative_profile.txt"
PROJECTS_FILE           = MEMORY_DIR / "projects.json"

MAX_EPISODES            = 60
MAX_LOGS                = 30
EXCERPT_MSGS            = 16
COMPRESS_THRESHOLD      = 50
COMPRESS_AGE_DAYS       = 30
NARRATIVE_REFRESH_EVERY = 20
TEMP_FACT_TTL           = 5

OLLAMA_URL         = "http://localhost:11434/api/chat"
OLLAMA_MODEL       = "qwen2.5:14b"
OLLAMA_EMBED_URL   = "http://localhost:11434/api/embeddings"
OLLAMA_EMBED_MODEL = "nomic-embed-text"


# ── Fact schema helpers ───────────────────────────────────────────────────────

def _fact_text(item) -> str:
    """Extract plain text from a fact entry (new dict or legacy string)."""
    if isinstance(item, dict):
        return item.get("fact", "")
    return str(item)


def _fact_age_label(item) -> str:
    """Returns '(learned ~N months ago)' for facts older than 30 days."""
    if not isinstance(item, dict):
        return ""
    learned = item.get("learned")
    if not learned:
        return ""
    try:
        days = (date.today() - date.fromisoformat(learned)).days
        if days < 30:
            return ""
        if days < 60:
            return " (learned ~1 month ago)"
        return f" (learned ~{days // 30} months ago)"
    except Exception:
        return ""


def _make_fact(text: str, temporary: bool = False) -> dict:
    """Create a new fact dict."""
    entry: dict = {"fact": text, "learned": date.today().isoformat()}
    if temporary:
        entry["temporary"] = True
        entry["sessions_remaining"] = TEMP_FACT_TTL
    return entry


# ── Core load / save ──────────────────────────────────────────────────────────

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


def _backup_core() -> None:
    """Rolling backup of core.json — keeps last 3 versions."""
    if not CORE_FILE.exists():
        return
    for i in range(2, 0, -1):
        src = MEMORY_DIR / f"core.backup.{i}.json"
        dst = MEMORY_DIR / f"core.backup.{i + 1}.json"
        if src.exists():
            shutil.copy2(src, dst)
    shutil.copy2(CORE_FILE, MEMORY_DIR / "core.backup.1.json")


# ── Episode load / save ───────────────────────────────────────────────────────

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


# ── Embedding helpers ─────────────────────────────────────────────────────────

def _load_episode_embeddings() -> dict:
    """Returns {timestamp_iso: [float, ...]}."""
    MEMORY_DIR.mkdir(exist_ok=True)
    if not EPISODE_EMBEDDINGS_FILE.exists():
        return {}
    try:
        return json.loads(EPISODE_EMBEDDINGS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_episode_embeddings(data: dict) -> None:
    MEMORY_DIR.mkdir(exist_ok=True)
    EPISODE_EMBEDDINGS_FILE.write_text(
        json.dumps(data, ensure_ascii=False), encoding="utf-8"
    )


async def _embed(text: str) -> list | None:
    """Get a nomic-embed-text embedding. Returns None on any failure."""
    if not text or not text.strip():
        return None
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.post(OLLAMA_EMBED_URL, json={
                "model":  OLLAMA_EMBED_MODEL,
                "prompt": text[:2000],
            })
        if r.is_success:
            return r.json().get("embedding")
    except Exception as e:
        print(f"[memory] embed error: {e}", flush=True)
    return None


def _cosine(a: list, b: list) -> float:
    """Pure-Python cosine similarity."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na  = sum(x * x for x in a) ** 0.5
    nb  = sum(x * x for x in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _recency_score(timestamp_iso: str) -> float:
    """1.0 for today; decays via 1/(1+log(1+days_ago)). 1 day→0.59, 7 days→0.32, 30 days→0.23."""
    try:
        days_ago = max(0.0, (datetime.now() - datetime.fromisoformat(timestamp_iso)).total_seconds() / 86400)
        return 1.0 / (1.0 + math.log(1.0 + days_ago))
    except Exception:
        return 0.5


def _relative_time(timestamp_iso: str) -> str:
    """Human-readable relative time: 'today', 'yesterday', '3 days ago', etc."""
    try:
        days_ago = (datetime.now() - datetime.fromisoformat(timestamp_iso)).days
        if days_ago == 0:  return "today"
        if days_ago == 1:  return "yesterday"
        if days_ago < 7:   return f"{days_ago} days ago"
        if days_ago < 14:  return "1 week ago"
        if days_ago < 30:  return f"{days_ago // 7} weeks ago"
        if days_ago < 60:  return "1 month ago"
        return f"{days_ago // 30} months ago"
    except Exception:
        return ""


# ── Episode relevance ranking ─────────────────────────────────────────────────

async def rank_episodes_by_relevance(
    episodes: list[dict],
    query_text: str,
    top_n: int = 10,
) -> list[dict]:
    """
    Blended score = 0.7 × semantic + 0.3 × recency.
    Falls back to recency if embedding fails.
    Pre-filters by tag overlap when possible.
    """
    if not episodes:
        return []

    # Tag pre-filter: if query words match episode tags, rank only that subset
    query_words = set(query_text.lower().split()) if query_text else set()
    if query_words:
        tag_filtered = [
            ep for ep in episodes
            if not ep.get("tags") or query_words & {t.lower() for t in ep.get("tags", [])}
        ]
        pool = tag_filtered if len(tag_filtered) >= 5 else episodes
    else:
        pool = episodes

    # Lazily embed any uncached episodes
    cache   = _load_episode_embeddings()
    changed = False
    for ep in pool:
        ts = ep.get("timestamp", "")
        if ts and ts not in cache:
            text = ep.get("summary", "") + " " + " ".join(ep.get("bullets", []))
            vec  = await _embed(text)
            if vec:
                cache[ts] = vec
                changed = True
    if changed:
        _save_episode_embeddings(cache)
        # Prune stale keys periodically
        if len(cache) % 10 == 0:
            all_ts = {ep.get("timestamp", "") for ep in load_episodes(False)} | \
                     {ep.get("timestamp", "") for ep in load_episodes(True)}
            stale = [k for k in list(cache) if k not in all_ts]
            if stale:
                for k in stale:
                    del cache[k]
                _save_episode_embeddings(cache)

    # Embed the query
    query_vec = await _embed(query_text)
    if query_vec is None:
        return pool[-top_n:]   # fallback: recency

    scored = []
    for ep in pool:
        ts  = ep.get("timestamp", "")
        vec = cache.get(ts)
        sem = _cosine(query_vec, vec) if vec else 0.0
        rec = _recency_score(ts)
        scored.append((0.7 * sem + 0.3 * rec, ep))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [ep for _, ep in scored[:top_n]]


# ── Project tracking ──────────────────────────────────────────────────────────

def load_projects() -> list:
    MEMORY_DIR.mkdir(exist_ok=True)
    if not PROJECTS_FILE.exists():
        return []
    try:
        data = json.loads(PROJECTS_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def save_projects(data: list) -> None:
    MEMORY_DIR.mkdir(exist_ok=True)
    PROJECTS_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def update_project(name: str, status: str | None = None, note: str | None = None) -> str:
    """Create or update a named project."""
    name = name.strip()
    if not name:
        return "Project name required."
    projects = load_projects()
    existing = next((p for p in projects if p["name"].lower() == name.lower()), None)
    today    = date.today().isoformat()
    if existing:
        if status:
            existing["status"] = status
        if note:
            existing.setdefault("notes", []).append(note)
        existing["last_updated"] = today
        save_projects(projects)
        return f"Project updated: {existing['name']}"
    else:
        projects.append({
            "name":             name,
            "status":           status or "active",
            "started":          today,
            "last_updated":     today,
            "notes":            [note] if note else [],
            "linked_episodes":  [],
        })
        save_projects(projects)
        return f"Project created: {name}"


def close_project(name: str) -> str:
    """Mark a project as done."""
    name_lower = name.strip().lower()
    projects   = load_projects()
    for p in projects:
        if p["name"].lower() == name_lower:
            p["status"]       = "done"
            p["last_updated"] = date.today().isoformat()
            save_projects(projects)
            return f"Project closed: {p['name']}"
    return f"Project not found: {name}"


def list_projects() -> str:
    """Return a formatted list of all projects."""
    projects = load_projects()
    if not projects:
        return "No projects tracked yet."
    lines = []
    for status_label in ("active", "paused", "done"):
        group = [p for p in projects if p.get("status") == status_label]
        if group:
            lines.append(f"── {status_label.capitalize()} ──")
            for p in group:
                lines.append(f"  {p['name']} (since {p.get('started', '?')})")
                if p.get("notes"):
                    lines.append(f"    Latest: {p['notes'][-1]}")
    return "\n".join(lines) if lines else "No projects."


def _update_project_record(name: str, note: str) -> None:
    """Internal: auto-update a project from in_progress extraction."""
    projects   = load_projects()
    name_lower = name.lower()
    today      = date.today().isoformat()
    for p in projects:
        if p["name"].lower() == name_lower:
            p.setdefault("notes", []).append(note)
            p["last_updated"] = today
            save_projects(projects)
            return


# ── Context formatting ────────────────────────────────────────────────────────

async def format_context(companion: bool = False, messages: list | None = None) -> str:
    """
    Returns the full memory block for injection into the system prompt.
    messages: recent conversation — used to build the embedding query for relevance ranking.
    """
    core          = load_core()
    default_eps   = load_episodes(companion=False)
    companion_eps = load_episodes(companion=True) if companion else []
    excerpt       = load_last_session_excerpt(companion=companion)

    sections: list[str] = []

    # ── Narrative profile ──────────────────────────────────────────────────────
    if NARRATIVE_FILE.exists():
        try:
            narrative = NARRATIVE_FILE.read_text(encoding="utf-8").strip()
            if narrative:
                sections.append(f"WHO TYLER IS:\n{narrative}")
        except Exception:
            pass

    # ── Currently in progress ──────────────────────────────────────────────────
    in_progress_items: list[str] = []
    for ep in reversed(default_eps[-2:]):
        for item in ep.get("in_progress", []):
            if item and item not in in_progress_items:
                in_progress_items.append(item)
    for proj in load_projects():
        if proj.get("status") == "active":
            entry = f"{proj['name']} (project)"
            if entry not in in_progress_items:
                in_progress_items.append(entry)
    if in_progress_items:
        sections.append("CURRENTLY IN PROGRESS:\n" +
                        "\n".join(f"• {item}" for item in in_progress_items))

    # ── Core facts ────────────────────────────────────────────────────────────
    if core:
        all_facts: list[str] = []
        for items in core.values():
            if isinstance(items, list):
                for item in items:
                    text  = _fact_text(item)
                    label = _fact_age_label(item)
                    if text:
                        all_facts.append(f"{text}{label}")
            elif isinstance(items, str) and items:
                all_facts.append(items)
        if all_facts:
            sections.append("WHAT I KNOW ABOUT TYLER:\n" +
                            "\n".join(f"- {f}" for f in all_facts))

    # ── Recent sessions ───────────────────────────────────────────────────────
    if default_eps:
        query_text = ""
        if messages:
            user_msgs  = [m.get("content", "") for m in messages if m.get("role") == "user"]
            query_text = " ".join(user_msgs[-3:])

        ranked = await rank_episodes_by_relevance(default_eps, query_text, top_n=10) if query_text \
                 else default_eps[-10:]

        lines = []
        for ep in reversed(ranked):
            rel    = _relative_time(ep.get("timestamp", ""))
            prefix = f"[{rel}] " if rel else ""
            title  = ep.get("summary", "Session")
            if ep.get("span"):
                title = f"{ep['span']}: {title}"
            header  = f"- {prefix}{title}"
            bullets = ep.get("bullets", [])
            lines.append(header + ("\n" + "\n".join(f"  {b}" for b in bullets) if bullets else ""))
        sections.append("RECENT SESSIONS:\n" + "\n".join(lines))

    if companion_eps:
        query_text = ""
        if messages:
            user_msgs  = [m.get("content", "") for m in messages if m.get("role") == "user"]
            query_text = " ".join(user_msgs[-3:])
        ranked_c = await rank_episodes_by_relevance(companion_eps, query_text, top_n=10) if query_text \
                   else companion_eps[-10:]
        lines_c  = []
        for ep in reversed(ranked_c):
            rel    = _relative_time(ep.get("timestamp", ""))
            prefix = f"[{rel}] " if rel else ""
            header = f"- {prefix}{ep.get('summary', 'Session')}"
            bullets = ep.get("bullets", [])
            lines_c.append(header + ("\n" + "\n".join(f"  {b}" for b in bullets) if bullets else ""))
        sections.append("RECENT COMPANION SESSIONS:\n" + "\n".join(lines_c))

    # ── Last conversation excerpt ─────────────────────────────────────────────
    if excerpt:
        sections.append(
            "LAST CONVERSATION (use naturally — reference it if relevant, "
            "ask to continue unfinished things):\n" + excerpt
        )

    return "\n\n".join(sections)


# ── Live memory tools ─────────────────────────────────────────────────────────

async def save_memory_fact(category: str, fact: str) -> str:
    """Immediately write a fact to core.json. Rejects semantic near-duplicates (>0.85 cosine)."""
    fact = fact.strip()
    if not fact:
        return "Memory save failed: empty fact."

    core    = load_core()
    new_vec = await _embed(fact)

    if new_vec:
        # Build embedding cache for existing facts once (avoids N HTTP calls)
        all_texts = [
            _fact_text(item)
            for items in core.values() if isinstance(items, list)
            for item in items
        ]
        for ex_text in all_texts:
            if not ex_text:
                continue
            ex_vec = await _embed(ex_text)
            if ex_vec and _cosine(new_vec, ex_vec) > 0.85:
                return f"Similar fact already exists: \"{ex_text}\""

    if category not in core:
        core[category] = []
    if not isinstance(core[category], list):
        return "Memory save failed: category is not a list."

    if fact in [_fact_text(f) for f in core[category]]:
        return f"Memory already contains: {fact}"

    core[category].append(_make_fact(fact))
    save_core(core)
    return f"Memory saved: {fact}"


def forget_memory_fact(fact_text: str) -> str:
    """Remove fact(s) from core.json by case-insensitive substring match."""
    query = fact_text.strip().lower()
    if not query:
        return "Nothing to forget."
    core    = load_core()
    removed = 0
    for cat, items in core.items():
        if isinstance(items, list):
            before     = len(items)
            core[cat]  = [item for item in items if query not in _fact_text(item).lower()]
            removed   += before - len(core[cat])
    if removed:
        save_core(core)
        return f"Removed {removed} fact(s) from memory."
    return "Couldn't find a matching fact to remove."


def get_memory_status() -> str:
    """Return a formatted overview of Maya's current memory state."""
    core     = load_core()
    episodes = load_episodes(False)
    projects = load_projects()

    lines = ["── Memory Status ──", ""]

    # Fact counts per category
    total_perm = total_temp = 0
    for cat, items in core.items():
        if isinstance(items, list):
            perm        = sum(1 for i in items if not (isinstance(i, dict) and i.get("temporary")))
            temp        = len(items) - perm
            total_perm += perm
            total_temp += temp
            lines.append(f"  {cat}: {perm} permanent, {temp} temporary")
    lines.append(f"  Total: {total_perm} permanent + {total_temp} temporary")
    lines.append("")

    # Recent episodes
    lines.append(f"Episodes: {len(episodes)} stored")
    for ep in reversed(episodes[-5:]):
        rel = _relative_time(ep.get("timestamp", ""))
        lines.append(f"  [{rel}] {ep.get('summary', '?')}")
    lines.append("")

    # In-progress
    in_prog: list[str] = []
    for ep in reversed(episodes[-2:]):
        in_prog.extend(ep.get("in_progress", []))
    seen: set[str] = set()
    unique_prog = [x for x in in_prog if not (x in seen or seen.add(x))]  # type: ignore[func-returns-value]
    if unique_prog:
        lines.append("Currently in progress:")
        for item in unique_prog:
            lines.append(f"  • {item}")
        lines.append("")

    # Active projects
    active = [p for p in projects if p.get("status") == "active"]
    if active:
        lines.append(f"Active projects ({len(active)}):")
        for p in active:
            lines.append(f"  • {p['name']} (started {p.get('started', '?')})")
        lines.append("")

    # File sizes
    for label, path in [
        ("core.json",               CORE_FILE),
        ("episodes.json",           EPISODES_FILE),
        ("episode_embeddings.json", EPISODE_EMBEDDINGS_FILE),
        ("narrative_profile.txt",   NARRATIVE_FILE),
        ("projects.json",           PROJECTS_FILE),
    ]:
        if path.exists():
            size = path.stat().st_size
            lines.append(f"  {label}: {size // 1024} KB" if size >= 1024 else f"  {label}: {size} B")
        else:
            lines.append(f"  {label}: not found")

    return "\n".join(lines)


# ── Prompts ───────────────────────────────────────────────────────────────────

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

Mark temporary=true for project/task-specific facts (will become irrelevant once done).
Mark temporary=false for stable facts (preferences, hardware, personal info).

Respond with valid JSON only:
{{
  "new_facts": [
    {{"category": "personal|technical|preferences|project", "fact": "one concise line", "temporary": false}}
  ],
  "session_title": "one sentence describing what this session was about",
  "session_bullets": [
    "• first key point from this session",
    "• second key point",
    "• third key point"
  ],
  "in_progress": ["working on X", "investigating Y"],
  "tags": ["ui", "memory", "spotify"]
}}

Rules:
  - session_bullets: 3 to 5 items, each starting with "• "
  - in_progress: 0 to 3 items for genuinely unfinished work; empty list if nothing pending
  - tags: 2 to 5 lowercase topic keywords
  - new_facts: empty list if nothing new is worth remembering

Conversation:
{conversation}

Current memory (don't re-extract what's already here):
{current_memory}
"""

_INCREMENTAL_EXTRACT_PROMPT = """\
Review this recent conversation excerpt. Extract ONLY new long-term facts about Tyler.

Same criteria: persistent preferences, constraints, technical facts, or project updates.
Mark temporary=true for project/task-specific facts; false for stable ones.

Return valid JSON only:
{{"new_facts": [{{"category": "personal|technical|preferences|project", "fact": "...", "temporary": false}}]}}

If nothing new: {{"new_facts": []}}

Recent conversation:
{conversation}

Current memory (don't re-extract what's already here):
{current_memory}
"""

_CONSOLIDATE_PROMPT = """\
Review the following facts stored in memory. Identify entries that are:
  - Redundant (same information stated twice or more in different words)
  - Outdated (superseded by a newer fact in the same list)
  - Contradicted by one of the newly added facts

IMPORTANT: Do NOT remove or modify facts that have "temporary": true — they expire on their own.
IMPORTANT: Do NOT rewrite the whole file — only return what to change.

Return valid JSON only:
{{
  "remove": ["exact text of fact to remove"],
  "update": [{{"old": "exact text to replace", "new": "corrected text"}}]
}}

If everything looks clean: {{"remove": [], "update": []}}

Current memory:
{current_memory}

Newly added facts (use these to detect what is now outdated):
{new_facts}
"""

_NARRATIVE_PROMPT = """\
You are writing a personal profile to help an AI assistant understand someone well.

Based on the facts and recent sessions below, write 2–3 paragraphs about Tyler in third person.
Cover: who he is, how he works, his technical setup, his preferences and communication style, \
and what he's currently focused on.
Be warm, specific, and concrete. Do not include temporary or task-specific facts.
Do not use headers or bullet points — just natural prose.

Facts:
{facts}

Recent sessions:
{sessions}
"""

_COMPRESS_PROMPT = """\
Summarise the following session notes into a single compact paragraph.
Capture the main topics, key decisions, and outcomes. Be specific but concise.
Return only the paragraph — no headers, no bullets, no preamble.

Sessions:
{sessions}
"""


# ── Background extraction helpers ─────────────────────────────────────────────

def _decrement_temp_facts() -> None:
    """Decrement TTL on temporary facts; remove any that expire this session."""
    core    = load_core()
    changed = False
    for cat, items in list(core.items()):
        if not isinstance(items, list):
            continue
        new_items = []
        for item in items:
            if isinstance(item, dict) and item.get("temporary"):
                remaining = item.get("sessions_remaining", 1) - 1
                changed = True
                if remaining <= 0:
                    print(f"[memory] expired temp fact: {item.get('fact', '')[:60]}", flush=True)
                else:
                    item["sessions_remaining"] = remaining
                    new_items.append(item)
            else:
                new_items.append(item)
        core[cat] = new_items
    if changed:
        save_core(core)


async def _consolidate_core(new_facts: list[str]) -> None:
    """LLM pass to remove redundant/outdated facts. Never touches temporary facts."""
    core    = load_core()
    current = json.dumps(core, indent=2, ensure_ascii=False)
    new_str = "\n".join(f"- {f}" for f in new_facts)
    prompt  = _CONSOLIDATE_PROMPT.format(current_memory=current, new_facts=new_str)

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(OLLAMA_URL, json={
                "model":    OLLAMA_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream":   False,
                "format":   "json",
            })
        if not r.is_success:
            return

        data    = json.loads(r.json().get("message", {}).get("content", "{}"))
        removes = data.get("remove", [])
        updates = data.get("update", [])
        if not removes and not updates:
            return

        core    = load_core()   # re-load fresh
        changed = False
        for to_remove in removes:
            for cat, items in core.items():
                if isinstance(items, list):
                    before    = len(items)
                    core[cat] = [i for i in items
                                 if _fact_text(i) != to_remove
                                 or (isinstance(i, dict) and i.get("temporary"))]
                    if len(core[cat]) < before:
                        changed = True
                        print(f"[memory] consolidated — removed: {to_remove[:60]}", flush=True)
        for upd in updates:
            old_text = upd.get("old", "")
            new_text = upd.get("new", "")
            if not old_text or not new_text:
                continue
            for cat, items in core.items():
                if isinstance(items, list):
                    for i, item in enumerate(items):
                        if _fact_text(item) == old_text and not (isinstance(item, dict) and item.get("temporary")):
                            items[i] = {**item, "fact": new_text} if isinstance(item, dict) else _make_fact(new_text)
                            changed = True
                            print("[memory] consolidated — updated fact", flush=True)
        if changed:
            save_core(core)
    except Exception as e:
        print(f"[memory] consolidation error: {e}", flush=True)


async def extract_facts_incremental(conversation_tail: list[dict], companion: bool = False) -> None:
    """Lightweight mid-session extraction — only updates core.json facts, no episode created."""
    if len(conversation_tail) < 4:
        return
    conv    = "\n".join(
        f"{m['role'].upper()}: {m['content']}"
        for m in conversation_tail
        if m.get("role") in ("user", "assistant")
    )
    current = json.dumps(load_core(), indent=2, ensure_ascii=False)
    prompt  = _INCREMENTAL_EXTRACT_PROMPT.format(conversation=conv, current_memory=current)
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(OLLAMA_URL, json={
                "model":    OLLAMA_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream":   False,
                "format":   "json",
            })
        if not r.is_success:
            return
        data  = json.loads(r.json().get("message", {}).get("content", "{}"))
        facts = data.get("new_facts", [])
        if facts:
            core  = load_core()
            added = 0
            for item in facts:
                cat       = item.get("category", "general")
                fact_text = item.get("fact", "").strip()
                temporary = item.get("temporary", False)
                if not fact_text:
                    continue
                if cat not in core:
                    core[cat] = []
                if isinstance(core[cat], list) and fact_text not in [_fact_text(f) for f in core[cat]]:
                    core[cat].append(_make_fact(fact_text, temporary=temporary))
                    added += 1
            if added:
                save_core(core)
                print(f"[memory] incremental +{added} fact(s)", flush=True)
        _decrement_temp_facts()
    except Exception as e:
        print(f"[memory] incremental extraction error: {e}", flush=True)


async def _regenerate_narrative() -> None:
    """Regenerate narrative_profile.txt from current facts and recent sessions."""
    core      = load_core()
    episodes  = load_episodes(False)

    all_facts = [
        _fact_text(item) for items in core.values() if isinstance(items, list)
        for item in items
        if _fact_text(item) and not (isinstance(item, dict) and item.get("temporary"))
    ]
    facts_str    = "\n".join(f"- {f}" for f in all_facts)
    sessions_str = "\n".join(
        f"- {ep.get('summary', '')} " + "; ".join(ep.get("bullets", []))
        for ep in reversed(episodes[-10:])
    )
    prompt = _NARRATIVE_PROMPT.format(facts=facts_str, sessions=sessions_str)
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(OLLAMA_URL, json={
                "model":    OLLAMA_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream":   False,
                "options":  {"temperature": 0.4},
            })
        if r.is_success:
            narrative = r.json().get("message", {}).get("content", "").strip()
            if narrative:
                MEMORY_DIR.mkdir(exist_ok=True)
                NARRATIVE_FILE.write_text(narrative, encoding="utf-8")
                print("[memory] narrative profile refreshed", flush=True)
    except Exception as e:
        print(f"[memory] narrative error: {e}", flush=True)


async def regenerate_profile_tool() -> str:
    """Tool-callable wrapper for narrative regeneration."""
    await _regenerate_narrative()
    return "Narrative profile refreshed."


async def _merge_new_facts(facts: list[dict], companion: bool) -> list[str]:
    """Merge extracted facts into core.json with semantic dedup. Returns added fact texts."""
    if not facts or companion:
        return []
    core  = load_core()
    added: list[str] = []

    # Build an embedding cache for ALL existing facts once — avoids N² HTTP calls
    all_existing: list[str] = [
        _fact_text(item)
        for items in core.values() if isinstance(items, list)
        for item in items
    ]
    embed_cache: dict[str, list | None] = {}
    for ex_text in all_existing:
        if ex_text and ex_text not in embed_cache:
            embed_cache[ex_text] = await _embed(ex_text)

    for item in facts:
        cat       = item.get("category", "general")
        fact_text = item.get("fact", "").strip()
        temporary = item.get("temporary", False)
        if not fact_text:
            continue
        if cat not in core:
            core[cat] = []
        if not isinstance(core[cat], list):
            continue
        existing_texts = [_fact_text(f) for f in core[cat]]
        if fact_text in existing_texts:
            continue
        # Semantic near-duplicate check using pre-cached embeddings
        skip    = False
        new_vec = await _embed(fact_text)
        if new_vec:
            for ex_text in existing_texts:
                ex_vec = embed_cache.get(ex_text) or await _embed(ex_text)
                if ex_vec and _cosine(new_vec, ex_vec) > 0.85:
                    skip = True
                    break
        if not skip:
            core[cat].append(_make_fact(fact_text, temporary=temporary))
            added.append(fact_text)
            # Add to cache so subsequent new facts can check against this one too
            if new_vec:
                embed_cache[fact_text] = new_vec

    if added:
        save_core(core)
        print(f"[memory] +{len(added)} fact(s)", flush=True)
    return added


# ── Post-session extraction ───────────────────────────────────────────────────

async def extract_and_save(session_history: list[dict], companion: bool = False) -> None:
    """
    Called asynchronously after a session ends.
    Extracts rich episode data, updates facts, consolidates, and refreshes narrative if due.
    """
    if len(session_history) < 4:
        return

    _decrement_temp_facts()
    _backup_core()

    conv    = "\n".join(
        f"{m['role'].upper()}: {m['content']}"
        for m in session_history
        if m.get("role") in ("user", "assistant")
    )
    current = json.dumps(load_core(), indent=2, ensure_ascii=False)
    prompt  = _EXTRACT_PROMPT.format(conversation=conv, current_memory=current)

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(OLLAMA_URL, json={
                "model":    OLLAMA_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream":   False,
                "format":   "json",
            })
        if not r.is_success:
            return

        data    = json.loads(r.json().get("message", {}).get("content", "{}"))
        facts   = data.get("new_facts", [])
        title   = data.get("session_title", "").strip()
        bullets = data.get("session_bullets", [])
        in_prog = data.get("in_progress", [])
        tags    = data.get("tags", [])

        clean_bullets = []
        for b in bullets:
            b = b.strip()
            if b:
                if not b.startswith("•"):
                    b = "• " + b
                clean_bullets.append(b)

        if title:
            eps     = load_episodes(companion=companion)
            episode = {
                "timestamp":   datetime.now().isoformat(),
                "summary":     title,
                "bullets":     clean_bullets,
                "in_progress": [i.strip() for i in in_prog if i.strip()],
                "tags":        [t.lower().strip() for t in tags if t.strip()],
            }
            eps.append(episode)
            if len(eps) > MAX_EPISODES:
                eps = eps[-MAX_EPISODES:]
            _save_episodes(eps, companion=companion)

            vec = await _embed(title + " " + " ".join(clean_bullets))
            if vec:
                cache = _load_episode_embeddings()
                cache[episode["timestamp"]] = vec
                _save_episode_embeddings(cache)

        added_facts = await _merge_new_facts(facts, companion)
        if added_facts:
            await _consolidate_core(added_facts)

        # Auto-update matching projects from in_progress
        if in_prog:
            active_projects = [p for p in load_projects() if p.get("status") == "active"]
            for item in in_prog:
                for proj in active_projects:
                    if proj["name"].lower() in item.lower():
                        _update_project_record(proj["name"], item)
                        break

        # Narrative refresh every N sessions
        all_eps = load_episodes(companion=False)
        if len(all_eps) > 0 and len(all_eps) % NARRATIVE_REFRESH_EVERY == 0:
            await _regenerate_narrative()

        await compress_old_episodes(companion=companion)
        print(f"[memory] session saved: \"{title}\"", flush=True)

    except Exception as e:
        print(f"[memory] extraction error: {e}", flush=True)


# ── Episode compression ───────────────────────────────────────────────────────

async def compress_old_episodes(companion: bool = False) -> None:
    """Compress old low-relevance episodes into monthly summaries when store exceeds threshold."""
    episodes = load_episodes(companion)
    if len(episodes) <= COMPRESS_THRESHOLD:
        return

    cutoff    = datetime.now() - timedelta(days=COMPRESS_AGE_DAYS)
    old_eps   = [ep for ep in episodes
                 if not ep.get("compressed")
                 and datetime.fromisoformat(ep["timestamp"]) < cutoff]
    keep_eps  = [ep for ep in episodes if ep not in old_eps]

    if not old_eps:
        return

    groups: dict[str, list[dict]] = defaultdict(list)
    for ep in old_eps:
        month_key = datetime.fromisoformat(ep["timestamp"]).strftime("%Y-%m")
        groups[month_key].append(ep)

    cache = _load_episode_embeddings()
    for month_key, group in groups.items():
        if len(group) < 5:
            keep_eps.extend(group)
            continue

        sessions_text = "\n\n".join(
            f"[{ep.get('summary', '')}]\n" + "\n".join(ep.get("bullets", []))
            for ep in group
        )
        prompt = _COMPRESS_PROMPT.format(sessions=sessions_text)
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                r = await client.post(OLLAMA_URL, json={
                    "model":    OLLAMA_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream":   False,
                })
            if not r.is_success:
                keep_eps.extend(group)
                continue

            summary = r.json().get("message", {}).get("content", "").strip()
            if not summary:
                keep_eps.extend(group)
                continue

            month_label = datetime.strptime(month_key, "%Y-%m").strftime("%B %Y")
            compressed  = {
                "timestamp":   group[0]["timestamp"],
                "summary":     f"Summary of {month_label}",
                "bullets":     [f"• {summary}"],
                "compressed":  True,
                "span":        month_label,
                "tags":        list({t for ep in group for t in ep.get("tags", [])}),
                "in_progress": [],
            }
            for ep in group:
                cache.pop(ep.get("timestamp", ""), None)
            keep_eps.append(compressed)
            vec = await _embed(summary)
            if vec:
                cache[compressed["timestamp"]] = vec
            print(f"[memory] compressed {len(group)} episodes → {month_label}", flush=True)
        except Exception as e:
            print(f"[memory] compression error for {month_key}: {e}", flush=True)
            keep_eps.extend(group)

    keep_eps.sort(key=lambda ep: ep.get("timestamp", ""))
    _save_episodes(keep_eps, companion)
    _save_episode_embeddings(cache)


# ── Session log search ────────────────────────────────────────────────────────

async def search_session_logs(query: str, top_n: int = 3) -> str:
    """Semantic search over past session logs. Returns formatted excerpts."""
    episodes = load_episodes(False)
    if not episodes:
        return "No session history found."

    # Pre-filter by tag overlap
    query_words  = set(query.lower().split())
    tag_filtered = [
        ep for ep in episodes
        if not ep.get("tags") or query_words & {t.lower() for t in ep.get("tags", [])}
    ]
    pool = tag_filtered if len(tag_filtered) >= 3 else episodes

    # Embed query; rank by pure semantic similarity (no recency — searching intentionally)
    query_vec = await _embed(query)
    if query_vec is None:
        pool = episodes[-top_n:]
    else:
        cache   = _load_episode_embeddings()
        changed = False
        for ep in pool:
            ts = ep.get("timestamp", "")
            if ts and ts not in cache:
                text = ep.get("summary", "") + " " + " ".join(ep.get("bullets", []))
                vec  = await _embed(text)
                if vec:
                    cache[ts] = vec
                    changed = True
        if changed:
            _save_episode_embeddings(cache)

        scored = []
        for ep in pool:
            ts  = ep.get("timestamp", "")
            vec = cache.get(ts)
            scored.append((_cosine(query_vec, vec) if vec else 0.0, ep))
        scored.sort(key=lambda x: x[0], reverse=True)
        pool = [ep for _, ep in scored[:top_n]]

    # Find matching log files and extract excerpts
    all_logs = sorted(LOGS_DIR.glob("*.json")) if LOGS_DIR.exists() else []
    results  = []
    for ep in pool:
        rel    = _relative_time(ep.get("timestamp", ""))
        title  = ep.get("summary", "Session")
        header = f"[Session from {rel} — \"{title}\"]"

        closest = None
        try:
            ep_dt   = datetime.fromisoformat(ep["timestamp"])
            closest = min(
                all_logs,
                key=lambda f: abs(datetime.strptime(f.stem, "%Y%m%d_%H%M%S") - ep_dt),
                default=None,
            )
        except Exception:
            pass

        excerpt = ""
        if closest:
            try:
                messages = json.loads(closest.read_text(encoding="utf-8")).get("messages", [])
                lines    = []
                for msg in messages[-10:]:
                    role    = msg.get("role", "")
                    content = msg.get("content", "").strip()[:200]
                    if not content:
                        continue
                    if role == "user":
                        lines.append(f"Tyler: {content}")
                    elif role == "assistant":
                        lines.append(f"MAYA: {content}")
                excerpt = "\n".join(lines)
            except Exception:
                pass

        if excerpt:
            results.append(f"{header}\n{excerpt}")
        else:
            bullets = ep.get("bullets", [])
            results.append(header + ("\n" + "\n".join(f"  {b}" for b in bullets) if bullets else ""))

    return "\n\n".join(results) if results else "No relevant sessions found."
