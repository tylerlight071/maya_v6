"""
Integration tests for the file tools in server.py.

Unlike test_paths.py (which tests the is_within helper in isolation), these call
the REAL _tool_files handler so the actual production code path — open_file's
roots enforcement and run_app's allowlist — is exercised. This is what catches
the "helper fixed but never wired in" class of regression.
"""
import asyncio
import pytest

# server.py pulls in heavy deps (sounddevice, numpy, ...). Skip cleanly if they
# aren't importable in the current environment rather than erroring the suite.
server = pytest.importorskip("server")


def _run(coro):
    return asyncio.run(coro)


def test_open_file_denies_path_outside_roots(tmp_path, monkeypatch):
    root = tmp_path / "allowed"
    root.mkdir()
    outside = tmp_path / "secret.txt"
    outside.write_text("x", encoding="utf-8")
    monkeypatch.setattr(server, "_load_config",
                        lambda: {"file_search_roots": [str(root)]})
    result = _run(server._tool_files("open_file", {"path": str(outside)}))
    assert "Access denied" in result


def test_open_file_denies_sibling_prefix(tmp_path, monkeypatch):
    """Regression guard for the prefix bug: '/x/DocumentsEvil' must NOT pass a
    root of '/x/Documents'. Fails if open_file reverts to startswith()."""
    root = tmp_path / "Documents"
    root.mkdir()
    sibling = tmp_path / "DocumentsEvil"
    sibling.mkdir()
    evil = sibling / "secret.txt"
    evil.write_text("x", encoding="utf-8")
    monkeypatch.setattr(server, "_load_config",
                        lambda: {"file_search_roots": [str(root)]})
    result = _run(server._tool_files("open_file", {"path": str(evil)}))
    assert "Access denied" in result


def test_run_app_rejects_unlisted_app(monkeypatch):
    monkeypatch.setattr(server, "_load_config",
                        lambda: {"app_allowlist": {"notepad": "C:/Windows/notepad.exe"}})
    result = _run(server._tool_files("run_app", {"name": "calc"}))
    assert "not in the app allowlist" in result
