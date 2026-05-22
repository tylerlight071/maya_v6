"""
MAYA Vision — presence detection, face recognition, and real-time emotion analysis.

Reference images live in Backend/models/tyler_refs/
  - Drop images there manually (any filename, .jpg or .png)
  - MAYA will also auto-capture high-confidence frames to grow the database
  - Fallback: Backend/models/tyler_ref.jpg (single image)

Emotion detection uses DeepFace (already installed) on every 3rd frame to keep
GPU load manageable. Results are stored in _last_emotion for the system prompt.
"""
import asyncio
import base64
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import websockets

VISION_PORT          = 8766
REFS_DIR             = Path(__file__).parent / "models/tyler_refs"
SINGLE_REF           = Path(__file__).parent / "models/tyler_ref.jpg"
MAX_AUTO_REFS        = 40
AUTO_CAPTURE_CONF    = 0.80
AUTO_CAPTURE_DET     = 0.92
MIN_CAPTURE_INTERVAL = 20 * 60

EMOTION_INTERVAL     = 3    # run emotion detection every N frames
EMOTION_CONFIDENCE   = 0.35 # minimum confidence to report a non-neutral emotion
FACE_PADDING         = 25   # pixels of padding around face crop for emotion

# Face recognition thresholds (cosine similarity against InsightFace embeddings).
# FACE_RECOG_FLOOR: minimum similarity to count as a match at all.
# FACE_RECOG_RANGE: span above the floor that maps to 0–100% confidence.
# Calibrated for the buffalo_l ArcFace model; adjust if recognition is unreliable.
FACE_RECOG_FLOOR = 0.28
FACE_RECOG_RANGE = 0.72

_app              = None
_tyler_embeddings: list = []
_last_capture_at  = 0.0

# Emotion state — read by server.py's build_system_prompt()
_last_emotion: dict = {"dominant": "neutral", "confidence": 0.5, "scores": {}}
_emotion_frame_count: int = 0

# Human-readable labels for the more clinical DeepFace names
_EMOTION_DISPLAY = {
    "angry":   "frustrated",
    "disgust": "uncomfortable",
    "fear":    "anxious",
    "happy":   "happy",
    "neutral": "neutral",
    "sad":     "low",
    "surprise":"surprised",
}


# ── Emotion detection ─────────────────────────────────────────────────────────

def _detect_emotion(face_bgr: np.ndarray) -> dict:
    """
    Run DeepFace emotion analysis on a pre-cropped face region.
    Returns dominant label, confidence (0-1), and all scores.
    Gracefully returns neutral on any failure.
    """
    try:
        from deepface import DeepFace  # lazy import — model loads on first call
        result = DeepFace.analyze(
            face_bgr,
            actions=["emotion"],
            enforce_detection=False,   # we already have the face crop
            silent=True,
        )
        if isinstance(result, list):
            result = result[0]

        dominant = result.get("dominant_emotion", "neutral")
        scores   = result.get("emotion", {})
        # DeepFace returns percentages 0-100; normalise to 0-1
        conf     = round(float(scores.get(dominant, 50.0)) / 100.0, 3)
        norm     = {k: round(float(v) / 100.0, 3) for k, v in scores.items()}

        return {"dominant": dominant, "confidence": conf, "scores": norm}
    except Exception as e:
        return {"dominant": "neutral", "confidence": 0.5, "scores": {}}


# ── Model loading ─────────────────────────────────────────────────────────────

def _embed_path(img_path: Path):
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    faces = _app.get(img)
    if not faces:
        return None
    largest = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
    return largest.normed_embedding


def _reload_embeddings() -> None:
    global _tyler_embeddings
    REFS_DIR.mkdir(parents=True, exist_ok=True)
    sources: list[Path] = sorted(REFS_DIR.glob("*.jpg")) + sorted(REFS_DIR.glob("*.png"))
    if not sources and SINGLE_REF.exists():
        sources = [SINGLE_REF]
    _tyler_embeddings = [e for e in (_embed_path(p) for p in sources) if e is not None]


def _maybe_auto_capture(img: np.ndarray, face, confidence: float) -> None:
    global _last_capture_at
    if confidence < AUTO_CAPTURE_CONF:
        return
    if getattr(face, "det_score", 1.0) < AUTO_CAPTURE_DET:
        return
    if time.monotonic() - _last_capture_at < MIN_CAPTURE_INTERVAL:
        return
    REFS_DIR.mkdir(parents=True, exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = REFS_DIR / f"auto_{ts}.jpg"
    cv2.imwrite(str(out_path), img)
    _last_capture_at = time.monotonic()
    print(f"[vision] auto-captured reference: {out_path.name} (conf={confidence:.2f})", flush=True)
    auto_files = sorted(REFS_DIR.glob("auto_*.jpg"))
    while len(auto_files) > MAX_AUTO_REFS:
        auto_files[0].unlink()
        auto_files = auto_files[1:]
    _reload_embeddings()
    print(f"[vision] database now has {len(_tyler_embeddings)} references", flush=True)


def load_vision_models() -> None:
    global _app
    from insightface.app import FaceAnalysis
    print("[vision] loading InsightFace...", flush=True)
    _app = FaceAnalysis(
        name="buffalo_l",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    _app.prepare(ctx_id=0, det_size=(640, 480))
    print("[vision] InsightFace ready", flush=True)
    _reload_embeddings()
    if _tyler_embeddings:
        print(f"[vision] {len(_tyler_embeddings)} reference(s) loaded for Tyler", flush=True)
    else:
        print(f"[vision] no references found — drop images in {REFS_DIR}", flush=True)

    # Warm up DeepFace emotion model on first load to avoid first-call delay
    try:
        dummy = np.zeros((112, 112, 3), dtype=np.uint8)
        _detect_emotion(dummy)
        print("[vision] emotion model warmed up", flush=True)
    except Exception:
        pass


# ── Frame analysis ────────────────────────────────────────────────────────────

def analyze_frame(b64_jpeg: str) -> dict:
    global _emotion_frame_count, _last_emotion

    base_result = {
        "present":            False,
        "recognized":         None,
        "confidence":         0.0,
        "emotion":            "neutral",
        "emotion_display":    "neutral",
        "emotion_confidence": 0.5,
        "emotion_scores":     {},
    }

    if _app is None:
        return base_result

    try:
        img_bytes = base64.b64decode(b64_jpeg)
        img_array = np.frombuffer(img_bytes, np.uint8)
        img       = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            return base_result

        faces = _app.get(img)
        if not faces:
            return base_result

        largest = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))

        # ── Emotion detection every EMOTION_INTERVAL frames ───────────────────
        _emotion_frame_count += 1
        if _emotion_frame_count % EMOTION_INTERVAL == 0:
            h, w  = img.shape[:2]
            x1, y1, x2, y2 = largest.bbox
            x1 = max(0,  int(x1) - FACE_PADDING)
            y1 = max(0,  int(y1) - FACE_PADDING)
            x2 = min(w,  int(x2) + FACE_PADDING)
            y2 = min(h,  int(y2) + FACE_PADDING)
            face_crop = img[y1:y2, x1:x2]
            if face_crop.size > 0:
                _last_emotion = _detect_emotion(face_crop)

        emotion_data = _last_emotion
        dominant     = emotion_data.get("dominant", "neutral")
        emo_conf     = emotion_data.get("confidence", 0.5)
        # Report neutral when confidence is too low — avoids misread spam
        reported_emotion = dominant if emo_conf >= EMOTION_CONFIDENCE else "neutral"
        display_emotion  = _EMOTION_DISPLAY.get(reported_emotion, reported_emotion)

        if not _tyler_embeddings:
            return {
                **base_result,
                "present":            True,
                "recognized":         "unknown",
                "emotion":            reported_emotion,
                "emotion_display":    display_emotion,
                "emotion_confidence": emo_conf,
                "emotion_scores":     emotion_data.get("scores", {}),
            }

        emb  = largest.normed_embedding
        best = max(float(np.dot(emb, ref)) for ref in _tyler_embeddings)

        if best > FACE_RECOG_FLOOR:
            confidence = round(min((best - FACE_RECOG_FLOOR) / FACE_RECOG_RANGE, 1.0), 2)
            _maybe_auto_capture(img, largest, confidence)
            return {
                "present":            True,
                "recognized":         "Tyler",
                "confidence":         confidence,
                "emotion":            reported_emotion,
                "emotion_display":    display_emotion,
                "emotion_confidence": emo_conf,
                "emotion_scores":     emotion_data.get("scores", {}),
            }

        return {
            "present":            True,
            "recognized":         "unknown",
            "confidence":         0.0,
            "emotion":            reported_emotion,
            "emotion_display":    display_emotion,
            "emotion_confidence": emo_conf,
            "emotion_scores":     emotion_data.get("scores", {}),
        }

    except Exception as e:
        return {**base_result, "error": str(e)}


# ── WebSocket server ──────────────────────────────────────────────────────────

VISION_MIN_INTERVAL = 1.0   # seconds between analyses (server-side rate limit ≈ 1 fps)

async def handle_vision_client(websocket) -> None:
    print("[vision] client connected", flush=True)
    last_analysis = 0.0
    try:
        async for raw in websocket:
            try:
                msg = json.loads(raw)
            except Exception:
                continue

            if msg.get("type") != "vision_frame":
                continue

            # Server-side rate limit — drop frames that arrive too quickly
            now = time.time()
            if now - last_analysis < VISION_MIN_INTERVAL:
                continue
            last_analysis = now

            try:
                loop   = asyncio.get_running_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, analyze_frame, msg.get("image", "")),
                    timeout=5.0,
                )
                await websocket.send(json.dumps({"type": "vision_update", **result}))
            except asyncio.TimeoutError:
                print("[vision] frame analysis timed out — skipping frame", flush=True)
            except websockets.exceptions.ConnectionClosed:
                raise
            except Exception as e:
                print(f"[vision] frame error: {e}", flush=True)

    except websockets.exceptions.ConnectionClosed:
        print("[vision] client disconnected", flush=True)


async def run() -> None:
    async with websockets.serve(handle_vision_client, "127.0.0.1", VISION_PORT):
        print(f"[vision] listening on ws://127.0.0.1:{VISION_PORT}", flush=True)
        await asyncio.Future()
