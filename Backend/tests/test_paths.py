"""Tests for path_utils.is_within — guards open_file against traversal."""
from pathlib import Path
from path_utils import is_within


def test_direct_child_allowed(tmp_path):
    child = tmp_path / "file.txt"
    child.touch()
    assert is_within(child, tmp_path) is True


def test_nested_child_allowed(tmp_path):
    nested = tmp_path / "a" / "b" / "file.txt"
    nested.parent.mkdir(parents=True)
    nested.touch()
    assert is_within(nested, tmp_path) is True


def test_sibling_prefix_rejected(tmp_path):
    """'/a/Doc' must NOT match root '/a/Documents' — the classic prefix bug."""
    root = tmp_path / "Documents"
    root.mkdir()
    sibling = tmp_path / "DocumentsEvil"
    sibling.mkdir()
    evil_file = sibling / "secret.txt"
    evil_file.touch()
    assert is_within(evil_file, root) is False


def test_parent_escape_rejected(tmp_path):
    root = tmp_path / "safe"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.touch()
    assert is_within(outside, root) is False


def test_dotdot_traversal_rejected(tmp_path):
    """Ensure '..' sequences cannot escape the root after resolve()."""
    root = tmp_path / "safe"
    root.mkdir()
    inner = root / "sub"
    inner.mkdir()
    # Construct a Path that appears inside root but tries to escape
    traversal = inner / ".." / ".." / "outside.txt"
    outside = tmp_path / "outside.txt"
    outside.touch()
    assert is_within(traversal, root) is False
