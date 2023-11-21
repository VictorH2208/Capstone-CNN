"""Utility functions."""

from pathlib import Path


def get_project_root_dir() -> Path:
    """Returns the project root directory."""
    code_dir = Path(__file__).parent
    project_root_dir = code_dir.parent
    # Assuming no symlinks
    return project_root_dir.resolve()
