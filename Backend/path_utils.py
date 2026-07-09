"""Shared path-safety utilities."""
from pathlib import Path


def is_within(child: Path, parent: Path) -> bool:
    """Return True iff child resolves to a path inside parent."""
    try:
        child.resolve().relative_to(parent)
        return True
    except ValueError:
        return False
