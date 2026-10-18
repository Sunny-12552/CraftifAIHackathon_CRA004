# SPDX-License-Identifier: Apache-2.0
"""Filesystem locations used across VisionSentinel AI.

The project folder lives inside OneDrive on the development laptop. Virtual
environments, downloaded models, the SQLite database and runtime exports are
therefore kept *outside* OneDrive by default (``%LOCALAPPDATA%\\VisionSentinelAI``)
so that OneDrive never syncs or locks them while inference is running.

Deliverable artefacts that must be committed (screenshots, reports, PipeGen
exports) stay inside the repository under ``evidence/``.
"""

from __future__ import annotations

import os
from pathlib import Path

ENV_APP_DIR = "VS_APP_DIR"

#: Repository root (the folder containing ``app.py``).
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _default_local_app_dir() -> Path:
    """Return a per-user writable directory that is not cloud-synced."""
    override = os.environ.get(ENV_APP_DIR)
    if override:
        return Path(override).expanduser().resolve()

    base = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_DATA_HOME")
    if base:
        return Path(base) / "VisionSentinelAI"
    return Path.home() / ".local" / "share" / "VisionSentinelAI"


#: Writable runtime state: models, database, exports, snapshots.
LOCAL_APP_DIR = _default_local_app_dir()

#: Default location for the pretrained ONNX model.
DEFAULT_MODELS_DIR = LOCAL_APP_DIR / "models"

#: Default location for the incident database and JSON exports.
DEFAULT_DATA_DIR = LOCAL_APP_DIR / "data"

#: Committed evidence: screenshots, speed reports, PipeGen exports, run logs.
EVIDENCE_DIR = PROJECT_ROOT / "evidence"

#: Default configuration file shipped with the project.
DEFAULT_SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.yaml"


def ensure_dir(path: Path) -> Path:
    """Create ``path`` (and parents) if needed and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_path(value: str | os.PathLike[str] | None, default: Path) -> Path:
    """Resolve a possibly-empty configured path against a default.

    Relative paths are interpreted relative to the project root so that a
    checked-in configuration stays portable between machines.
    """
    if value is None or str(value).strip() == "":
        return default
    candidate = Path(str(value)).expanduser()
    return candidate if candidate.is_absolute() else (PROJECT_ROOT / candidate)
