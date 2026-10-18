# SPDX-License-Identifier: Apache-2.0
"""VisionSentinel AI -- real-time safety and restricted-zone monitoring.

The package is deliberately layered so that one detection/zone/alert/storage
pipeline serves both execution paths:

    video source -> detection backend -> shared detection events -> tracking
    -> zone monitoring -> alert manager -> SQLite incident store -> JSON/MQTT
    -> dashboard

Nothing below the detection layer knows which engine is running, and the
NVIDIA path is the genuine CraftifAI PipeGen pipeline rather than a substitute
detector.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
