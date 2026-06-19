"""
MAYA ComfyUI integration — image generation with a dedicated widget WebSocket.

Port 8770: streams generation status to the ImageGen widget.
ComfyUI must be running locally (start via Stability Matrix).
"""
import asyncio
import contextlib
import copy
import json
import random
import shutil
import time
import uuid
from pathlib import Path

import httpx
import websockets

COMFYUI_URL    = "http://127.0.0.1:8188"
COMFYUI_WS     = "ws://127.0.0.1:8188"
WIDGET_PORT    = 8770
OLLAMA_URL     = "http://localhost:11434/api/chat"
OLLAMA_MODEL   = "huihui_ai/dolphin3-abliterated:8b-llama3.1-q4_K_M"  # keep in sync with server.py

# ComfyUI path — read from models/config.json, fallback to default
def _comfyui_dir_from_config() -> Path:
    config_path = Path(__file__).parent / "models/config.json"
    try:
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        if cfg.get("comfyui_dir"):
            return Path(cfg["comfyui_dir"])
    except Exception:
        pass
    return Path(r"C:\Users\tyler\Downloads\Data\Packages\ComfyUI")

COMFYUI_DIR    = _comfyui_dir_from_config()

def _find_comfyui_python() -> Path:
    """Find the Python executable for ComfyUI, checking common Stability Matrix paths."""
    candidates = [
        COMFYUI_DIR / ".venv" / "Scripts" / "python.exe",
        COMFYUI_DIR / "venv"  / "Scripts" / "python.exe",
        Path(r"C:\Users\tyler\Downloads\Data\Python310\python.exe"),
        Path(r"C:\Users\tyler\Downloads\Data\Python311\python.exe"),
        Path(r"C:\Users\tyler\Downloads\Data\Python\python.exe"),
    ]
    for p in candidates:
        if p.exists():
            print(f"[comfyui] Python found: {p}", flush=True)
            return p
    return COMFYUI_DIR / "venv" / "Scripts" / "python.exe"   # expected default

COMFYUI_PYTHON = _find_comfyui_python()
COMFYUI_OUTPUT = COMFYUI_DIR / "output"
GALLERY_DIR    = Path(r"C:\Users\tyler\Pictures\Images")

# Auto-started ComfyUI process handle (None if not started by us)
_comfyui_process = None

# Workflow node IDs
POSITIVE_NODE      = "6"
KSAMPLER_NODE      = "9"
FACE_DETAILER_NODE = "15"

_clients:        set  = set()
_current_status: dict = {"status": "idle", "step": 0, "total": 0, "filename": None, "error": None}
_current_prompt_id: "str | None" = None      # last queued prompt, used for status polls
_job_done_event:  "asyncio.Event | None" = None  # set by poll_status to unblock wait_for_completion

# Workflow loaded from disk at startup — used when the widget doesn't send one
_WORKFLOW_FILE = Path(__file__).parent / "models/comfyui_workflow.json"
def _load_workflow() -> "dict | None":
    try:
        if _WORKFLOW_FILE.exists():
            return json.loads(_WORKFLOW_FILE.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[comfyui] workflow load error: {e}", flush=True)
    return None
_loaded_workflow: "dict | None" = _load_workflow()


# ── Pose / action LoRA library — loaded from models/pose_loras.json ──────────

def _load_pose_loras() -> dict:
    p = Path(__file__).parent / "models/pose_loras.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[comfyui] pose_loras.json load error: {e}", flush=True)
    return {}

POSE_LORAS: dict = _load_pose_loras()


_POSE_SLOTS = ["18", "20", "21"]   # workflow nodes available for pose LoRAs


def select_pose_loras(description: str) -> "tuple[list[dict], str]":
    """
    Score LoRAs against description, then pick at most one per category to avoid
    conflicts. Within each category, winning LoRA is chosen randomly weighted by score.
    Up to 3 LoRAs total (one per workflow slot).
    """
    dl = description.lower()

    # Score every LoRA; group matches by category
    by_category: dict[str, list] = {}
    for filename, data in POSE_LORAS.items():
        score = sum(1 for kw in data["keywords"] if kw in dl)
        if score == 0:
            continue
        cat = data.get("category", "misc")
        by_category.setdefault(cat, []).append((score, filename, data))

    # From each category pick ONE winner — weighted-random among candidates with score > 0
    winners: list[tuple[int, str, dict]] = []
    for cat, candidates in by_category.items():
        total = sum(s for s, _, _ in candidates)
        r = random.uniform(0, total)
        cumulative = 0.0
        for score, filename, data in sorted(candidates, key=lambda x: x[0], reverse=True):
            cumulative += score
            if r <= cumulative:
                winners.append((score, filename, data))
                break

    # Sort winners by score descending; take up to 3
    winners.sort(key=lambda x: x[0], reverse=True)
    selected = winners[:len(_POSE_SLOTS)]

    if not selected:
        return [], "full body, medium shot, wide angle"

    for score, filename, _ in selected:
        print(f"[comfyui] pose LoRA: {filename} (score {score})", flush=True)

    composition = selected[0][2]["composition"]
    result = [{"filename": fn, "strength": data["strength"], "triggers": data["triggers"]}
              for _, fn, data in selected]
    return result, composition


# ── Prompt enhancement via Ollama ─────────────────────────────────────────────

_ENHANCE_SYSTEM = """\
You are an expert Stable Diffusion prompt engineer specialising in explicit anime/illustration.

Convert the user's scene description into rich, comma-separated SD prompt tags. Be generous — expand every detail the scene implies. A good output has 20-35 tags covering: action/pose, setting/environment, lighting, mood, clothing/state, body language, and expression.

Do NOT add character appearance tags (hair, eye colour, body type) — LoRA handles those.
Do NOT repeat quality tags (best quality, masterpiece, absurdres) — already present.
Do NOT include flowers, floral, botanical imagery.
Output ONLY the comma-separated tags. No explanation, no preamble, no quotes.

SETTING — always expand the environment with 3-5 tags:
- Dark forest / outdoors → moonlit clearing, ancient stone altar, gnarled trees, mist, ethereal glow, dramatic shadows
- Dungeon / underground → stone dungeon, iron chains, torchlight, damp walls, dramatic shadows
- Penthouse / indoor luxury → luxury penthouse, city skyline view, soft ambient lighting, silk sheets
- Throne room / fantasy → demonic throne room, dramatic lighting, ornate pillars, hellfire background
- Always include at least: a location tag, a lighting tag, an atmosphere tag

LIGHTING — always be specific:
- Night/outdoor → moonlight, silvery light, rim lighting
- Indoor fantasy → torchlight, candlelight, dramatic shadows
- Cinematic → cinematic lighting, dramatic chiaroscuro
- Glowing/magical → ethereal glow, magic particle effects, bioluminescent

COMPOSITION RULES — always include a shot type:
- Full-body sex scenes → full body, wide shot
- Facial / cumshot → bust shot, upper body, face and chest in frame, from above
- Titjob / paizuri → upper body, chest focus, medium shot
- Gangbang / group → wide shot, group framing, full body
- Default → full body, medium shot
Never default to close-up face unless the scene is specifically about the face.

EXPRESSION RULES — always include 2-3 expression tags matching the intensity:
- Rough sex / hard fucking → open mouth moaning, flushed, sweat, heavy breathing, desperate
- Orgasm / creampie → ahegao, eyes rolled back, tongue out, drooling, blissful, trembling
- Gangbang / many men → ahegao, mind broken, dazed, eyes glazed, drooling, overwhelmed
- Deepthroat / face fuck → gagging, tears streaming, eyes watering, drool, mascara running
- Tentacle / monster / extreme → wide eyes, tears, crying, open mouth, terrified, shocked, struggling
- Restrained / BDSM → gritting teeth, desperate, biting lip, fearful, straining against bonds
- Seductive / teasing → bedroom eyes, half-lidded, smirk, seductive smile, biting lip, flushed
- Defeated / mind-break → ahegao, vacant eyes, tongue out, drooling, limp, used
- Creampie aftermath → satisfied, blissful, flushed, panting, cum dripping
- Gentle / romantic → soft smile, happy, blissful, loving gaze, tender

Examples:
Input:  "succubus, dark altar, bound and helpless, sex from behind"
Output: full body, wide shot, demonic altar, hellfire background, chains, iron shackles, stone floor, torchlight, dramatic shadows, smoke, sex from behind, legs spread, arms restrained above head, ahegao, eyes rolled back, tongue out, drooling, flushed, tears, overwhelmed, sweat glistening

Input:  "getting railed from behind standing up, forest clearing at night"
Output: full body, from behind, standing, sex from behind, wide shot, moonlit forest clearing, ancient trees, mist, silver moonlight, rim lighting, fog, bent over slightly, open mouth moaning, flushed, heavy breathing, desperate, sweat, hair dishevelled

Input:  "tentacle pit, restrained, multiple tentacles"
Output: full body, wide shot, tentacle environment, underground cavern, bioluminescent glow, damp stone, tentacles everywhere, suspended mid-air, arms and legs wrapped, multiple penetration, wide eyes, tears streaming, crying, open mouth, terrified, shocked, struggling, drool

Input:  "seductive lingerie tease, luxury penthouse"
Output: upper body, medium shot, luxury penthouse, floor-to-ceiling windows, city lights bokeh, soft ambient lighting, silk sheets, bedroom eyes, half-lidded, seductive smile, biting lip, flushed, leaning forward, cleavage focus, inviting, warm glow
"""


async def enhance_prompt(description: str) -> str:
    """Convert natural language to optimised SD prompt tags using Ollama."""
    try:
        async with httpx.AsyncClient(timeout=20) as c:
            r = await c.post(OLLAMA_URL, json={
                "model":   OLLAMA_MODEL,
                "messages": [
                    {"role": "system",  "content": _ENHANCE_SYSTEM},
                    {"role": "user",    "content": description},
                ],
                "stream":     False,
                "keep_alive": "10m",
                "options":    {"temperature": 0.4, "num_predict": 350},
            })
        if r.is_success:
            return r.json().get("message", {}).get("content", description).strip()
    except Exception as e:
        print(f"[comfyui] enhance_prompt error: {e}", flush=True)
    return description   # fallback: use as-is


# ── Connectivity ──────────────────────────────────────────────────────────────

async def is_running() -> bool:
    try:
        async with httpx.AsyncClient(timeout=3) as c:
            r = await c.get(f"{COMFYUI_URL}/system_stats")
            return r.is_success
    except Exception:
        return False


# ── Broadcast to widget clients ───────────────────────────────────────────────

async def _broadcast(msg: dict) -> None:
    global _current_status
    _current_status = msg
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


# ── ComfyUI job submission ────────────────────────────────────────────────────

async def queue_prompt(workflow: dict, client_id: str) -> str:
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.post(
            f"{COMFYUI_URL}/prompt",
            json={"prompt": workflow, "client_id": client_id},
        )
    r.raise_for_status()
    return r.json()["prompt_id"]


async def _history_complete(prompt_id: str) -> bool:
    """Check ComfyUI history API to see if a prompt has finished."""
    try:
        async with httpx.AsyncClient(timeout=5) as c:
            r = await c.get(f"{COMFYUI_URL}/history/{prompt_id}")
        if r.is_success:
            data = r.json()
            entry = data.get(prompt_id, {})
            return bool(entry.get("outputs"))
    except Exception:
        pass
    return False


async def wait_for_completion(prompt_id: str, client_id: str, timeout: float = 120) -> bool:
    """
    Wait for a ComfyUI prompt to finish.

    Primary method: watch the WebSocket for execution/progress events.
    Fallback: poll the history API every 5 s so we never get permanently stuck
    even if the completion event is missed.
    """
    uri = f"{COMFYUI_WS}/ws?client_id={client_id}"
    deadline    = time.time() + timeout
    last_check  = time.time()
    seen_progress = False   # True once we've received at least one progress event

    try:
        async with websockets.connect(uri, ping_interval=20) as ws:
            while time.time() < deadline:
                # Check if poll_status already confirmed completion
                if _job_done_event and _job_done_event.is_set():
                    print("[comfyui] wait_for_completion: job_done_event set, returning True", flush=True)
                    return True

                # Periodic history fallback — checks every 5 s once sampling has started
                if seen_progress and time.time() - last_check > 5:
                    last_check = time.time()
                    if await _history_complete(prompt_id):
                        print("[comfyui] completion confirmed via history API (in wait loop)", flush=True)
                        return True

                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=3)
                except asyncio.TimeoutError:
                    continue

                try:
                    msg = json.loads(raw)
                except Exception:
                    continue

                mtype = msg.get("type")
                data  = msg.get("data", {})
                # Some ComfyUI versions omit prompt_id in certain events;
                # accept the event if prompt_id matches OR if it's absent and
                # we've already seen progress for our prompt.
                pid_matches = (
                    data.get("prompt_id") == prompt_id
                    or (seen_progress and not data.get("prompt_id"))
                )

                if mtype == "progress" and data.get("prompt_id") == prompt_id:
                    seen_progress = True
                    step  = int(data.get("value", 0))
                    total = int(data.get("max", 1))
                    asyncio.create_task(_broadcast({
                        "status": "generating", "step": step, "total": total,
                        "filename": None, "error": None,
                    }))

                elif mtype in ("executing", "executed") and pid_matches:
                    if data.get("node") is None:
                        print(f"[comfyui] execution complete (executing node=null)", flush=True)
                        return True

                elif mtype == "execution_success" and pid_matches:
                    print(f"[comfyui] execution complete (execution_success)", flush=True)
                    return True

                elif mtype == "execution_error" and pid_matches:
                    print(f"[comfyui] execution error: {data}", flush=True)
                    return False

    except Exception as e:
        print(f"[comfyui] wait_for_completion exception: {e}", flush=True)

    # Final fallback after timeout or exception
    if await _history_complete(prompt_id):
        print(f"[comfyui] completion confirmed via history (post-loop)", flush=True)
        return True

    return False


async def get_output_filenames(prompt_id: str) -> list[dict]:
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(f"{COMFYUI_URL}/history/{prompt_id}")
        if not r.is_success:
            return []
        history = r.json()
        outputs = history.get(prompt_id, {}).get("outputs", {})
        images  = []
        for node_output in outputs.values():
            images.extend(node_output.get("images", []))
        return images
    except Exception:
        return []


async def copy_to_gallery(image_infos: list[dict]) -> list[str]:
    GALLERY_DIR.mkdir(parents=True, exist_ok=True)
    copied = []

    for info in image_infos:
        filename  = info.get("filename", "")
        subfolder = info.get("subfolder", "")
        img_type  = info.get("type", "output")
        if img_type == "temp" or not filename:
            continue
        src = COMFYUI_OUTPUT / subfolder / filename if subfolder else COMFYUI_OUTPUT / filename
        print(f"[comfyui] copy attempt: {src} → {GALLERY_DIR / filename}", flush=True)
        if src.exists():
            dst = GALLERY_DIR / filename
            shutil.copy2(str(src), str(dst))
            copied.append(filename)
        else:
            print(f"[comfyui] source not found: {src}", flush=True)

    # Fallback: if history API returned nothing, grab the most recent PNG from output
    if not copied:
        print("[comfyui] history copy found nothing — scanning output folder for recent files", flush=True)
        cutoff = time.time() - 120   # files modified in last 2 minutes
        try:
            recent = sorted(
                [f for f in COMFYUI_OUTPUT.glob("*.png") if f.stat().st_mtime > cutoff],
                key=lambda f: f.stat().st_mtime,
                reverse=True,
            )
            for src in recent[:3]:   # at most 3 most recent
                dst = GALLERY_DIR / src.name
                shutil.copy2(str(src), str(dst))
                copied.append(src.name)
                print(f"[comfyui] fallback copy: {src.name}", flush=True)
        except Exception as e:
            print(f"[comfyui] fallback scan error: {e}", flush=True)

    return copied


# ── Full generation pipeline ──────────────────────────────────────────────────

async def start_comfyui() -> bool:
    """Launch ComfyUI if it isn't already running. Returns True when ready."""
    global _comfyui_process

    if await is_running():
        return True

    if not COMFYUI_PYTHON.exists():
        print(f"[comfyui] Python not found at {COMFYUI_PYTHON}", flush=True)
        return False

    import subprocess
    print("[comfyui] starting ComfyUI...", flush=True)
    try:
        _comfyui_process = subprocess.Popen(
            [str(COMFYUI_PYTHON), "main.py", "--listen", "127.0.0.1", "--port", "8188"],
            cwd=str(COMFYUI_DIR),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as e:
        print(f"[comfyui] failed to start: {e}", flush=True)
        return False

    # Poll until ready — up to 60 seconds
    for _ in range(60):
        await asyncio.sleep(1)
        if await is_running():
            print("[comfyui] ComfyUI ready", flush=True)
            return True

    print("[comfyui] timed out waiting for ComfyUI to start", flush=True)
    return False


def stop_comfyui() -> None:
    """Terminate the ComfyUI process if we started it."""
    global _comfyui_process
    if _comfyui_process and _comfyui_process.poll() is None:
        _comfyui_process.terminate()
        print("[comfyui] ComfyUI stopped", flush=True)
    _comfyui_process = None


async def _unload_llm_models() -> None:
    """Unload all currently-loaded Ollama models to free VRAM before image generation."""
    try:
        async with httpx.AsyncClient(timeout=5) as c:
            r = await c.get("http://localhost:11434/api/ps")
            if not r.is_success:
                print(f"[comfyui] could not list Ollama models: {r.status_code}", flush=True)
                return
            for m in r.json().get("models", []):
                name = m.get("name", "")
                if not name:
                    continue
                ur = await c.post(OLLAMA_URL, json={"model": name, "messages": [], "keep_alive": 0})
                if ur.is_success:
                    print(f"[comfyui] unloaded {name} from VRAM", flush=True)
                else:
                    print(f"[comfyui] failed to unload {name}: {ur.status_code}", flush=True)
    except Exception as e:
        print(f"[comfyui] _unload_llm_models error: {e}", flush=True)


async def generate(description: str, workflow_template: dict) -> "tuple[bool, str, list[str]]":
    """
    Enhance prompt, queue to ComfyUI, wait for completion, copy to gallery.
    Broadcasts status events to the widget throughout.
    """
    # 1. Ensure ComfyUI is running — auto-start if needed
    print("[comfyui] checking if ComfyUI is running...", flush=True)
    if not await is_running():
        await _broadcast({"status": "enhancing", "step": 0, "total": 0, "filename": None, "error": None})
        print("[comfyui] not running — attempting auto-start...", flush=True)
        if not await start_comfyui():
            msg = "ComfyUI could not be started automatically. Check the install path."
            await _broadcast({"status": "error", "step": 0, "total": 0, "filename": None, "error": msg})
            return False, msg, []
    print("[comfyui] ComfyUI is running.", flush=True)

    # 2. Unload LLM models from VRAM to make room for generation
    await _unload_llm_models()

    # 3. Enhance prompt
    await _broadcast({"status": "enhancing", "step": 0, "total": 0, "filename": None, "error": None})
    enhanced = await enhance_prompt(description)
    print(f"[comfyui] enhanced prompt: {enhanced}", flush=True)

    # 3. Build workflow
    workflow = copy.deepcopy(workflow_template)
    missing = [n for n in (POSITIVE_NODE, KSAMPLER_NODE) if n not in workflow]
    if missing:
        return False, f"Workflow is missing required node(s): {', '.join(missing)}. Check ComfyUI workflow JSON.", []
    for node_id in (KSAMPLER_NODE, FACE_DETAILER_NODE):
        if node_id in workflow:
            workflow[node_id]["inputs"]["seed"] = random.randint(1, 2**31 - 1)

    # Select and inject pose LoRAs (up to 3 can be mixed)
    pose_loras, composition = select_pose_loras(description + " " + enhanced)
    all_triggers = []
    for i, slot in enumerate(_POSE_SLOTS):
        if slot not in workflow:
            continue
        if i < len(pose_loras):
            lora = pose_loras[i]
            workflow[slot]["inputs"]["lora_name"]      = lora["filename"]
            workflow[slot]["inputs"]["strength_model"] = lora["strength"]
            workflow[slot]["inputs"]["strength_clip"]  = lora["strength"]
            all_triggers.append(lora["triggers"])
        else:
            # Zero out unused slots
            workflow[slot]["inputs"]["strength_model"] = 0
            workflow[slot]["inputs"]["strength_clip"]  = 0

    # Build positive prompt: base + all pose triggers + composition + enhanced
    base = workflow[POSITIVE_NODE]["inputs"].get("text", "").rstrip()
    parts = [base]
    if all_triggers:
        parts.append(", ".join(all_triggers))
    parts.append(composition)
    parts.append(enhanced)
    workflow[POSITIVE_NODE]["inputs"]["text"] = "\n\n".join(parts)

    # 4. Queue
    print("[comfyui] queuing prompt...", flush=True)
    await _broadcast({"status": "generating", "step": 0, "total": 0, "filename": None, "error": None})
    client_id = str(uuid.uuid4())
    global _current_prompt_id, _job_done_event
    _job_done_event = asyncio.Event()  # fresh event for this job
    try:
        prompt_id = await queue_prompt(workflow, client_id)
        _current_prompt_id = prompt_id
        print(f"[comfyui] queued — prompt_id: {prompt_id}", flush=True)
    except Exception as e:
        print(f"[comfyui] queue failed: {e}", flush=True)
        await _broadcast({"status": "error", "step": 0, "total": 0, "filename": None, "error": str(e)})
        return False, f"Queue failed: {e}", []

    # 5. Wait (progress events broadcast inside wait_for_completion)
    print(f"[comfyui] waiting for completion (client_id={client_id[:8]}...)", flush=True)
    success = await wait_for_completion(prompt_id, client_id)
    if not success:
        msg = "Generation timed out or ComfyUI reported an error."
        await _broadcast({"status": "error", "step": 0, "total": 0, "filename": None, "error": msg})
        return False, msg, []
    print("[comfyui] generation complete, fetching output files...", flush=True)

    # 6. Copy to gallery
    image_infos = await get_output_filenames(prompt_id)
    print(f"[comfyui] found {len(image_infos)} output image(s)", flush=True)
    filenames   = await copy_to_gallery(image_infos)
    print(f"[comfyui] copied to gallery: {filenames}", flush=True)

    if not filenames:
        msg = f"No images copied. Check output path: {COMFYUI_OUTPUT}"
        await _broadcast({"status": "error", "step": 0, "total": 0, "filename": None, "error": msg})
        return False, msg, []

    # 7. Done
    print(f"[comfyui] broadcasting done — {filenames[0]}", flush=True)
    await _broadcast({
        "status": "done", "step": 0, "total": 0,
        "filename": filenames[0], "error": None,
        "enhanced_prompt": enhanced,
    })
    return True, f"Generated {len(filenames)} image(s).", filenames


# ── Widget WebSocket handler ──────────────────────────────────────────────────

_generation_lock: asyncio.Lock | None = None


async def handle_client(websocket) -> None:
    _clients.add(websocket)
    try:
        # Push current status on connect
        await websocket.send(json.dumps(_current_status))

        async for raw in websocket:
            try:
                msg = json.loads(raw)
            except Exception:
                continue

            if msg.get("type") == "poll_status":
                if _current_prompt_id and _current_status.get("status") == "generating":
                    done = await _history_complete(_current_prompt_id)
                    if done:
                        print("[comfyui] poll_status: history confirms done — signalling wait loop", flush=True)
                        # Signal wait_for_completion to exit immediately and release the lock
                        if _job_done_event and not _job_done_event.is_set():
                            _job_done_event.set()
                        # Also handle the file copy + broadcast here in case wait_for_completion
                        # already exited (race condition safety)
                        if _current_status.get("status") != "done":
                            image_infos = await get_output_filenames(_current_prompt_id)
                            filenames   = await copy_to_gallery(image_infos)
                            if filenames:
                                await _broadcast({
                                    "status": "done", "step": 0, "total": 0,
                                    "filename": filenames[0], "error": None,
                                })
                    else:
                        await websocket.send(json.dumps(_current_status))
                else:
                    await websocket.send(json.dumps(_current_status))

            elif msg.get("type") == "generate":
                prompt   = msg.get("prompt", "").strip()
                workflow = msg.get("workflow") or _loaded_workflow
                if not prompt:
                    continue
                if not workflow:
                    await websocket.send(json.dumps({
                        "status": "error", "step": 0, "total": 0,
                        "filename": None, "error": "No workflow loaded — check Backend/models/comfyui_workflow.json.",
                    }))
                    continue
                if _generation_lock and _generation_lock.locked():
                    await websocket.send(json.dumps({
                        "status": "error", "step": 0, "total": 0,
                        "filename": None, "error": "Already generating — please wait.",
                    }))
                    continue
                asyncio.create_task(_guarded_generate(prompt, workflow))

            elif msg.get("type") == "enhance_only":
                prompt = msg.get("prompt", "").strip()
                if prompt:
                    enhanced = await enhance_prompt(prompt)
                    await websocket.send(json.dumps({"type": "enhanced", "result": enhanced}))

    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        _clients.discard(websocket)


async def _guarded_generate(prompt: str, workflow: dict) -> None:
    global _generation_lock
    if _generation_lock is None:
        _generation_lock = asyncio.Lock()
    async with _generation_lock:
        await generate(prompt, workflow)


# ── Entry point ───────────────────────────────────────────────────────────────

async def run() -> None:
    global _generation_lock
    _generation_lock = asyncio.Lock()   # created here so it's always bound to running loop
    try:
        async with websockets.serve(handle_client, "127.0.0.1", WIDGET_PORT):
            print(f"[comfyui] widget ws://127.0.0.1:{WIDGET_PORT}", flush=True)
            await asyncio.Future()
    finally:
        pass   # no background poll_task to cancel
