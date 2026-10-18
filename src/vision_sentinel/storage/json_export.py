# SPDX-License-Identifier: Apache-2.0
"""JSON export of the incident history.

The export is an envelope rather than a bare array, so a consumer can tell which
filters produced the file and whether it is complete. Writes are atomic: the
dashboard can download the file while the engine keeps writing incidents.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..events import SCHEMA_VERSION, Incident, utc_now, utc_now_iso

#: Version of the export envelope itself (distinct from the record schema).
EXPORT_FORMAT_VERSION = 1

APPLICATION_NAME = "VisionSentinel AI"

REQUIRED_INCIDENT_KEYS = (
    "incident_id",
    "created_utc",
    "event_type",
    "camera_id",
    "zone_id",
    "zone_name",
    "track_id",
    "label",
    "confidence",
    "bbox_xywh",
    "foot_point",
    "source",
    "frame_id",
    "snapshot_path",
    "schema_version",
)


@dataclass(frozen=True, slots=True)
class ExportResult:
    """Summary of one export operation."""

    path: Path
    count: int
    exported_utc: str
    filters: Mapping[str, Any]

    def describe(self) -> str:
        return (
            f"Exported {self.count} incident(s) to {self.path} at {self.exported_utc}"
        )


def build_export(
    incidents: Sequence[Incident],
    filters: Mapping[str, Any] | None = None,
    exported_utc: str | None = None,
) -> dict[str, Any]:
    """Build the export document for a set of incidents."""
    clean_filters = {
        key: value for key, value in (filters or {}).items() if value not in (None, "")
    }
    return {
        "application": APPLICATION_NAME,
        "export_format_version": EXPORT_FORMAT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "exported_utc": exported_utc or utc_now_iso(),
        "count": len(incidents),
        "filters": clean_filters,
        "incidents": [incident.to_dict() for incident in incidents],
    }


def export_incidents(
    store: Any,
    path: str | Path,
    *,
    start_utc: str | None = None,
    end_utc: str | None = None,
    camera_id: str | None = None,
    zone_id: str | None = None,
    event_type: str | None = None,
    source: str | None = None,
    newest_first: bool = False,
) -> ExportResult:
    """Query the store and write a validated JSON export.

    ``store`` is anything exposing ``query(**filters) -> list[Incident]``, which
    keeps this function unit-testable with a plain in-memory list wrapper.
    """
    incidents = store.query(
        start_utc=start_utc,
        end_utc=end_utc,
        camera_id=camera_id,
        zone_id=zone_id,
        event_type=event_type,
        source=source,
        newest_first=newest_first,
    )
    filters = {
        "start_utc": start_utc,
        "end_utc": end_utc,
        "camera_id": camera_id,
        "zone_id": zone_id,
        "event_type": event_type,
        "source": source,
    }
    payload = build_export(incidents, filters)
    validate_export(payload)

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    _write_atomic(target, payload)

    return ExportResult(
        path=target,
        count=len(incidents),
        exported_utc=str(payload["exported_utc"]),
        filters=payload["filters"],
    )


def default_export_path(export_dir: str | Path, moment: datetime | None = None) -> Path:
    """Build a timestamped export filename."""
    stamp = (moment or utc_now()).strftime("%Y%m%dT%H%M%SZ")
    return Path(export_dir) / f"incidents_{stamp}.json"


def validate_export(payload: Any) -> None:
    """Raise :class:`ValueError` if ``payload`` is not a valid export document.

    Used by the test suite to prove AT-08 rather than trusting ``json.dump``.
    """
    if not isinstance(payload, dict):
        raise ValueError("An export document must be a JSON object")

    for key in ("application", "export_format_version", "schema_version", "exported_utc", "count", "incidents"):
        if key not in payload:
            raise ValueError(f"Export document is missing the {key!r} field")

    if not isinstance(payload["incidents"], list):
        raise ValueError("Export 'incidents' must be a list")
    if payload["count"] != len(payload["incidents"]):
        raise ValueError(
            f"Export 'count' ({payload['count']}) does not match the number of "
            f"incidents ({len(payload['incidents'])})"
        )
    if not payload["exported_utc"].endswith("Z") and "+00:00" not in payload["exported_utc"]:
        raise ValueError("Export 'exported_utc' must be a UTC timestamp")

    seen: set[str] = set()
    for index, incident in enumerate(payload["incidents"]):
        if not isinstance(incident, dict):
            raise ValueError(f"incidents[{index}] must be an object")
        for key in REQUIRED_INCIDENT_KEYS:
            if key not in incident:
                raise ValueError(f"incidents[{index}] is missing the {key!r} field")
        incident_id = incident["incident_id"]
        if not incident_id:
            raise ValueError(f"incidents[{index}] has an empty incident_id")
        if incident_id in seen:
            raise ValueError(f"Duplicate incident_id {incident_id!r} in the export")
        seen.add(incident_id)
        created = incident["created_utc"]
        if not isinstance(created, str) or not (
            created.endswith("Z") or "+00:00" in created
        ):
            raise ValueError(
                f"incidents[{index}].created_utc must be a UTC timestamp, got {created!r}"
            )
        if not isinstance(incident["bbox_xywh"], list) or len(incident["bbox_xywh"]) != 4:
            raise ValueError(f"incidents[{index}].bbox_xywh must be a four-item list")
        confidence = incident["confidence"]
        if not isinstance(confidence, (int, float)) or not 0.0 <= float(confidence) <= 1.0:
            raise ValueError(
                f"incidents[{index}].confidence must be between 0.0 and 1.0"
            )


def load_export(path: str | Path) -> dict[str, Any]:
    """Read and validate an export file."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_export(payload)
    return payload


def _write_atomic(target: Path, payload: Mapping[str, Any]) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target.parent,
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        os.replace(temporary_path, target)
        temporary_path = None
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
