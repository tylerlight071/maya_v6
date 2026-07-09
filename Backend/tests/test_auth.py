"""Tests for Backend/auth.py — the per-session token gate."""
import asyncio
import auth


class _Req:
    def __init__(self, path: str):
        self.path = path


class FakeWS:
    def __init__(self, path: str):
        self.request = _Req(path)
        self.closed = False
        self.close_code: int | None = None

    async def close(self, code: int = 1000, reason: str = "") -> None:
        self.closed = True
        self.close_code = code


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def test_missing_token_rejected(monkeypatch):
    monkeypatch.setattr(auth, "read_token", lambda: "secret")
    ws = FakeWS("/")
    assert _run(auth.check(ws)) is False
    assert ws.closed
    assert ws.close_code == 4401


def test_empty_token_rejected(monkeypatch):
    monkeypatch.setattr(auth, "read_token", lambda: "secret")
    ws = FakeWS("/?token=")
    assert _run(auth.check(ws)) is False


def test_wrong_token_rejected(monkeypatch):
    monkeypatch.setattr(auth, "read_token", lambda: "secret")
    ws = FakeWS("/?token=nope")
    assert _run(auth.check(ws)) is False


def test_valid_token_accepted(monkeypatch):
    monkeypatch.setattr(auth, "read_token", lambda: "secret")
    ws = FakeWS("/?token=secret")
    assert _run(auth.check(ws)) is True
    assert not ws.closed


def test_no_server_token_rejects_all(monkeypatch):
    """Fails closed when the token file is empty/missing."""
    monkeypatch.setattr(auth, "read_token", lambda: "")
    ws = FakeWS("/?token=anything")
    assert _run(auth.check(ws)) is False
