# SPDX-License-Identifier: Apache-2.0
"""Model acquisition with an atomic, size-checked download.

Ported from the verified Intel baseline. A partial or truncated download must
never be left behind to be loaded as a model, so the file is written to a
temporary name and moved into place only after its size has been validated.
"""

from __future__ import annotations

import os
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

from .base import DetectorError

#: A pretrained YOLOX-S export is ~36 MB; anything smaller is not a model.
MINIMUM_MODEL_BYTES = 1_000_000


def ensure_model(
    model_path: Path,
    model_url: str,
    allow_download: bool = True,
    timeout_s: float = 60.0,
) -> Path:
    """Return a usable model path, downloading it once if necessary."""
    if model_path.is_file() and model_path.stat().st_size >= MINIMUM_MODEL_BYTES:
        return model_path

    if model_path.is_file() and model_path.stat().st_size < MINIMUM_MODEL_BYTES:
        raise DetectorError(
            f"The model file at {model_path} is only "
            f"{model_path.stat().st_size} bytes, which is too small to be a real "
            "model. Delete it and retry the download."
        )

    if not allow_download:
        raise DetectorError(
            f"Detection model not found at {model_path}.\n"
            "Provide one with --model-path, or allow the download by leaving "
            "detection.model_url set."
        )

    if not model_url.strip():
        raise DetectorError(
            f"Detection model not found at {model_path} and no model URL is "
            "configured (detection.model_url)."
        )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        model_url, headers={"User-Agent": "VisionSentinelAI/1.0"}
    )
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=model_path.parent, suffix=".download", delete=False
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                while chunk := response.read(1024 * 1024):
                    temporary_file.write(chunk)

        written = temporary_path.stat().st_size
        if written < MINIMUM_MODEL_BYTES:
            raise DetectorError(
                f"Downloaded model is only {written} bytes, which is too small "
                "to be a real model. Check the network or the model URL."
            )
        os.replace(temporary_path, model_path)
        temporary_path = None
    except urllib.error.HTTPError as error:
        raise DetectorError(
            f"Could not download the detection model (HTTP {error.code}) from "
            f"{model_url}. Download it manually and pass --model-path."
        ) from error
    except urllib.error.URLError as error:
        raise DetectorError(
            f"Could not reach {model_url} to download the detection model "
            f"({error.reason}). Download it manually and pass --model-path."
        ) from error
    except OSError as error:
        raise DetectorError(
            f"Could not write the detection model to {model_path}: {error}"
        ) from error
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

    return model_path
