"""
Tests for the security/correctness hardening added to MAYA:

  * auth.check          — per-session WebSocket token gate (fails closed)
  * server._render_prompt — brace-safe prompt substitution
  * server.switch_mode  — protected modes gated behind face verification
  * server._validate_config — loud warnings on a malformed config.json
  * vision.tyler_recently_verified — fail-closed identity gate

The auth tests run everywhere (stdlib only). The server/vision tests import
heavy optional deps (torch, cv2, insightface); where those are unavailable
(CI, sandbox) the tests skip instead of erroring, and run fully in the app's
own virtualenv.
"""
import asyncio
import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

import auth  # stdlib-only, always importable

# Optional heavy modules — import defensively so the file collects cleanly.
try:
    import server
    HAVE_SERVER = True
except Exception:                       # pragma: no cover - env dependent
    HAVE_SERVER = False

try:
    import vision
    HAVE_VISION = True
except Exception:                       # pragma: no cover - env dependent
    HAVE_VISION = False


# --------------------------------------------------------------------------- #
# auth.check — session-token gate                                             #
# --------------------------------------------------------------------------- #
class _FakeWS:
    """Minimal stand-in for a websockets connection."""
    def __init__(self, path: str):
        self.request = type("Req", (), {"path": path})()
        self.close_code = None

    async def close(self, code=None, reason=None):
        self.close_code = code


def test_provided_token_parsing():
    assert auth._provided_token(_FakeWS("/?token=abc123")) == "abc123"
    assert auth._provided_token(_FakeWS("/")) == ""
    assert auth._provided_token(_FakeWS("/?foo=bar")) == ""


def _write_token(tmp_path, monkeypatch, value: str):
    f = tmp_path / ".session_token"
    f.write_text(value, encoding="utf-8")
    monkeypatch.setattr(auth, "_TOKEN_FILE", f)


def test_check_accepts_matching_token(tmp_path, monkeypatch):
    _write_token(tmp_path, monkeypatch, "s3cr3t")
    ws = _FakeWS("/?token=s3cr3t")
    assert asyncio.run(auth.check(ws)) is True
    assert ws.close_code is None


def test_check_rejects_wrong_token(tmp_path, monkeypatch):
    _write_token(tmp_path, monkeypatch, "s3cr3t")
    ws = _FakeWS("/?token=nope")
    assert asyncio.run(auth.check(ws)) is False
    assert ws.close_code == 4401


def test_check_rejects_missing_token(tmp_path, monkeypatch):
    _write_token(tmp_path, monkeypatch, "s3cr3t")
    ws = _FakeWS("/")
    assert asyncio.run(auth.check(ws)) is False
    assert ws.close_code == 4401


def test_check_fails_closed_without_token_file(tmp_path, monkeypatch):
    # No token file on disk -> expected token empty -> everything rejected.
    monkeypatch.setattr(auth, "_TOKEN_FILE", tmp_path / "does_not_exist")
    ws = _FakeWS("/?token=anything")
    assert asyncio.run(auth.check(ws)) is False
    assert ws.close_code == 4401


# --------------------------------------------------------------------------- #
# server._render_prompt — brace-safe substitution                            #
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not HAVE_SERVER, reason="server module deps unavailable")
def test_render_prompt_resolves_placeholders():
    out = server._render_prompt("Time: {datetime}\n{memory_block}",
                                datetime="X", memory_block="M")
    assert out == "Time: X\nM"


@pytest.mark.skipif(not HAVE_SERVER, reason="server module deps unavailable")
def test_render_prompt_survives_stray_braces():
    # A mode file using SD-style {weight:1.2} must NOT crash (str.format would).
    tpl = "SD: {weight:1.2} then {datetime}"
    out = server._render_prompt(tpl, datetime="X", memory_block="")
    assert out == "SD: {weight:1.2} then X"


@pytest.mark.skipif(not HAVE_SERVER, reason="server module deps unavailable")
def test_render_prompt_does_not_reparse_value_braces():
    out = server._render_prompt("{memory_block}", datetime="", memory_block="note {curly}")
    assert out == "note {curly}"


# --------------------------------------------------------------------------- #
# server.switch_mode — protected-mode identity gate                          #
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not HAVE_SERVER, reason="server module deps unavailable")
def test_switch_mode_default_resets(monkeypatch):
    msg = server.switch_mode("default")
    assert "default" in msg.lower()
    assert server._active_mode_name is None


@pytest.mark.skipif(not (HAVE_SERVER and HAVE_VISION),
                    reason="server/vision deps unavailable")
def test_switch_mode_blocks_protected_when_unverified(monkeypatch):
    monkeypatch.setattr(server, "_active_mode_name", None, raising=False)
    monkeypatch.setattr(server, "_active_mode_prompt", None, raising=False)
    monkeypatch.setattr(vision, "tyler_recently_verified", lambda *a, **k: False)
    # Should refuse and NOT activate the protected mode.
    msg = server.switch_mode("companion")
    assert "recognise" in msg.lower() or "recognize" in msg.lower() or "see" in msg.lower()
    assert server._active_mode_name is None


@pytest.mark.skipif(not (HAVE_SERVER and HAVE_VISION),
                    reason="server/vision deps unavailable")
def test_switch_mode_allows_protected_when_verified(monkeypatch):
    monkeypatch.setattr(server, "_active_mode_name", None, raising=False)
    monkeypatch.setattr(server, "_active_mode_prompt", None, raising=False)
    monkeypatch.setattr(vision, "tyler_recently_verified", lambda *a, **k: True)
    # Stub the mode loader so the test doesn't depend on the mode file's content.
    monkeypatch.setattr(server, "load_mode", lambda name: "STUB PROMPT")
    msg = server.switch_mode("companion")
    assert "companion" in msg.lower()
    assert server._active_mode_name == "companion"
    server.switch_mode("default")  # cleanup


@pytest.mark.skipif(not HAVE_SERVER, reason="server module deps unavailable")
def test_switch_mode_unknown_nonprotected_reports_not_found(monkeypatch):
    msg = server.switch_mode("banana")
    assert "not found" in msg.lower()
    assert server._active_mode_name is None


# --------------------------------------------------------------------------- #
# server._validate_config — loud warnings                                    #
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not HAVE_SERVER, reason="server module deps unavailable")
def test_validate_config_flags_problems(monkeypatch):
    logs = []
    monkeypatch.setattr(server, "log", lambda m: logs.append(m))
    server._validate_config({
        "stt_gpu": "zero",          # wrong type (should be int)
        "unknown_key": 1,           # unknown -> typo warning
        "file_search_roots": [],    # empty critical -> note
    })
    blob = " ".join(logs).lower()
    assert "stt_gpu" in blob
    assert "unknown_key" in blob
    assert "file_search_roots" in blob


@pytest.mark.skipif(not HAVE_SERVER, reason="server module deps unavailable")
def test_validate_config_empty_warns(monkeypatch):
    logs = []
    monkeypatch.setattr(server, "log", lambda m: logs.append(m))
    server._validate_config({})
    assert any("config.json" in m.lower() for m in logs)


# --------------------------------------------------------------------------- #
# vision.tyler_recently_verified — fail-closed identity gate                 #
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not HAVE_VISION, reason="vision module deps unavailable")
def test_verify_fails_closed_without_model(monkeypatch):
    monkeypatch.setattr(vision, "_app", None)
    assert vision.tyler_recently_verified() is False


@pytest.mark.skipif(not HAVE_VISION, reason="vision module deps unavailable")
def test_verify_fails_closed_without_embeddings(monkeypatch):
    monkeypatch.setattr(vision, "_app", object())
    monkeypatch.setattr(vision, "_tyler_embeddings", [])
    assert vision.tyler_recently_verified() is False


@pytest.mark.skipif(not HAVE_VISION, reason="vision module deps unavailable")
def test_verify_respects_recency_window(monkeypatch):
    import time as _t
    monkeypatch.setattr(vision, "_app", object())
    monkeypatch.setattr(vision, "_tyler_embeddings", ["ref"])
    # Recognized just now -> verified within a 30s window.
    monkeypatch.setattr(vision, "_last_recognized_at", _t.monotonic())
    assert vision.tyler_recently_verified(30.0) is True
    # Recognized long ago -> stale -> not verified.
    monkeypatch.setattr(vision, "_last_recognized_at", _t.monotonic() - 999)
    assert vision.tyler_recently_verified(30.0) is False
