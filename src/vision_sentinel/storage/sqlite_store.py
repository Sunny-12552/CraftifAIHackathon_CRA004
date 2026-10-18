# SPDX-License-Identifier: Apache-2.0
"""SQLite incident storage.

Design notes
------------
* **WAL journal mode.** The engine thread writes incidents while the dashboard
  thread reads them. Write-ahead logging lets both proceed without the reader
  blocking the writer.
* **UTC only.** ``created_utc`` is an ISO-8601 UTC string, which sorts
  lexicographically, so date-range filtering is a simple string comparison and
  needs no timezone conversion in SQL.
* **Schema versioning.** ``schema_info`` records the applied version so a future
  migration has somewhere to start, and an unexpected version fails loudly
  rather than silently reading the wrong layout.
* **Thread safety.** A single connection guarded by a re-entrant lock; the write
  volume (a handful of incidents per hour) makes pooling unnecessary.

Replaces the baseline's ``events.json``, which rewrote its entire array on every
event and had no identity, camera, zone or track fields.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..events import SCHEMA_VERSION, Incident, utc_now

SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_info (
    version INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS incidents (
    incident_id     TEXT PRIMARY KEY,
    created_utc     TEXT NOT NULL,
    event_type      TEXT NOT NULL,
    camera_id       TEXT NOT NULL,
    zone_id         TEXT NOT NULL,
    zone_name       TEXT,
    track_id        TEXT,
    class_name      TEXT NOT NULL,
    confidence      REAL NOT NULL,
    bbox_json       TEXT NOT NULL,
    foot_point_json TEXT,
    source          TEXT NOT NULL,
    frame_id        INTEGER NOT NULL,
    snapshot_path   TEXT,
    schema_version  INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_incidents_created ON incidents (created_utc);
CREATE INDEX IF NOT EXISTS idx_incidents_zone    ON incidents (zone_id);
CREATE INDEX IF NOT EXISTS idx_incidents_camera  ON incidents (camera_id);
CREATE INDEX IF NOT EXISTS idx_incidents_type    ON incidents (event_type);
"""

_COLUMNS = (
    "incident_id, created_utc, event_type, camera_id, zone_id, zone_name, "
    "track_id, class_name, confidence, bbox_json, foot_point_json, source, "
    "frame_id, snapshot_path, schema_version"
)


class StorageError(RuntimeError):
    """Raised when the incident store cannot be opened or written."""


class IncidentStore:
    """Persistent incident history backed by SQLite."""

    def __init__(self, db_path: str | Path, retention_days: int = 0) -> None:
        self.db_path = Path(db_path)
        self.retention_days = retention_days
        self._lock = threading.RLock()

        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise StorageError(
                f"Could not create the incident database directory "
                f"{self.db_path.parent}: {error}"
            ) from error

        try:
            self._connection = sqlite3.connect(
                str(self.db_path), check_same_thread=False, timeout=10.0
            )
        except sqlite3.Error as error:
            raise StorageError(
                f"Could not open the incident database at {self.db_path}: {error}\n"
                "If the file is locked, close any other instance of the "
                "application or set storage.db_path to a different location."
            ) from error

        self._connection.row_factory = sqlite3.Row
        self._initialise()

    # -- setup -----------------------------------------------------------
    def _initialise(self) -> None:
        with self._lock:
            try:
                self._connection.execute("PRAGMA journal_mode=WAL")
                self._connection.execute("PRAGMA synchronous=NORMAL")
                self._connection.executescript(SCHEMA)
                row = self._connection.execute(
                    "SELECT version FROM schema_info LIMIT 1"
                ).fetchone()
                if row is None:
                    self._connection.execute(
                        "INSERT INTO schema_info (version) VALUES (?)",
                        (SCHEMA_VERSION,),
                    )
                elif int(row["version"]) != SCHEMA_VERSION:
                    raise StorageError(
                        f"The incident database at {self.db_path} uses schema "
                        f"version {int(row['version'])}, but this build expects "
                        f"{SCHEMA_VERSION}. Move the existing file aside to start "
                        "a fresh database."
                    )
                self._connection.commit()
            except sqlite3.Error as error:
                raise StorageError(
                    f"Could not initialise the incident database at {self.db_path}: {error}"
                ) from error

    # -- writes ----------------------------------------------------------
    def append(self, incident: Incident) -> Incident:
        """Insert one incident."""
        return self.append_many([incident])[0]

    def append_many(self, incidents: Sequence[Incident]) -> list[Incident]:
        """Insert several incidents in one transaction."""
        if not incidents:
            return []
        rows = [
            (
                incident.incident_id,
                incident.created_utc,
                incident.event_type,
                incident.camera_id,
                incident.zone_id,
                incident.zone_name,
                incident.track_id,
                incident.label,
                float(incident.confidence),
                json.dumps(list(incident.bbox_xywh)),
                json.dumps([float(incident.foot_point[0]), float(incident.foot_point[1])]),
                incident.source,
                int(incident.frame_id),
                incident.snapshot_path,
                int(incident.schema_version),
            )
            for incident in incidents
        ]
        with self._lock:
            try:
                self._connection.executemany(
                    f"INSERT INTO incidents ({_COLUMNS}) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    rows,
                )
                self._connection.commit()
            except sqlite3.IntegrityError as error:
                raise StorageError(
                    f"An incident with one of these identifiers already exists "
                    f"({error}). Incident ids must be unique."
                ) from error
            except sqlite3.Error as error:
                raise StorageError(
                    f"Could not write {len(rows)} incident(s) to {self.db_path}: {error}"
                ) from error
        return list(incidents)

    def prune(self, retention_days: int | None = None) -> int:
        """Delete incidents older than the retention window. Returns rows removed."""
        days = self.retention_days if retention_days is None else retention_days
        if days <= 0:
            return 0
        cutoff = (utc_now() - timedelta(days=days)).isoformat(timespec="milliseconds").replace(
            "+00:00", "Z"
        )
        with self._lock:
            try:
                cursor = self._connection.execute(
                    "DELETE FROM incidents WHERE created_utc < ?", (cutoff,)
                )
                self._connection.commit()
                return int(cursor.rowcount or 0)
            except sqlite3.Error as error:
                raise StorageError(
                    f"Could not apply retention to {self.db_path}: {error}"
                ) from error

    # -- reads -----------------------------------------------------------
    def query(
        self,
        start_utc: str | None = None,
        end_utc: str | None = None,
        camera_id: str | None = None,
        zone_id: str | None = None,
        event_type: str | None = None,
        source: str | None = None,
        limit: int | None = None,
        offset: int = 0,
        newest_first: bool = True,
    ) -> list[Incident]:
        """Query incidents with optional filters."""
        where, params = self._build_filter(
            start_utc=start_utc,
            end_utc=end_utc,
            camera_id=camera_id,
            zone_id=zone_id,
            event_type=event_type,
            source=source,
        )
        direction = "DESC" if newest_first else "ASC"
        sql = f"SELECT {_COLUMNS} FROM incidents {where} ORDER BY created_utc {direction}, rowid {direction}"
        if limit is not None:
            sql += " LIMIT ? OFFSET ?"
            params = [*params, int(limit), int(offset)]
        elif offset:
            sql += " LIMIT -1 OFFSET ?"
            params = [*params, int(offset)]

        with self._lock:
            try:
                rows = self._connection.execute(sql, params).fetchall()
            except sqlite3.Error as error:
                raise StorageError(f"Could not query {self.db_path}: {error}") from error
        return [Incident.from_row(row) for row in rows]

    def count(
        self,
        start_utc: str | None = None,
        end_utc: str | None = None,
        camera_id: str | None = None,
        zone_id: str | None = None,
        event_type: str | None = None,
        source: str | None = None,
    ) -> int:
        where, params = self._build_filter(
            start_utc=start_utc,
            end_utc=end_utc,
            camera_id=camera_id,
            zone_id=zone_id,
            event_type=event_type,
            source=source,
        )
        with self._lock:
            try:
                row = self._connection.execute(
                    f"SELECT COUNT(*) AS total FROM incidents {where}", params
                ).fetchone()
            except sqlite3.Error as error:
                raise StorageError(f"Could not count {self.db_path}: {error}") from error
        return int(row["total"]) if row else 0

    def stats(self) -> dict[str, Any]:
        """Aggregate counts used by the dashboard."""
        with self._lock:
            try:
                total = self._connection.execute(
                    "SELECT COUNT(*) AS total FROM incidents"
                ).fetchone()
                by_type = self._connection.execute(
                    "SELECT event_type, COUNT(*) AS total FROM incidents "
                    "GROUP BY event_type ORDER BY total DESC"
                ).fetchall()
                by_zone = self._connection.execute(
                    "SELECT zone_id, zone_name, COUNT(*) AS total FROM incidents "
                    "GROUP BY zone_id, zone_name ORDER BY total DESC"
                ).fetchall()
                by_camera = self._connection.execute(
                    "SELECT camera_id, COUNT(*) AS total FROM incidents "
                    "GROUP BY camera_id ORDER BY total DESC"
                ).fetchall()
                window = self._connection.execute(
                    "SELECT MIN(created_utc) AS first_utc, MAX(created_utc) AS last_utc "
                    "FROM incidents"
                ).fetchone()
            except sqlite3.Error as error:
                raise StorageError(f"Could not read statistics from {self.db_path}: {error}") from error

        return {
            "total": int(total["total"]) if total else 0,
            "by_event_type": {row["event_type"]: int(row["total"]) for row in by_type},
            "by_zone": {
                (row["zone_name"] or row["zone_id"]): int(row["total"]) for row in by_zone
            },
            "by_camera": {row["camera_id"]: int(row["total"]) for row in by_camera},
            "first_utc": window["first_utc"] if window else None,
            "last_utc": window["last_utc"] if window else None,
            "db_path": str(self.db_path),
        }

    def distinct(self, column: str) -> list[str]:
        """Distinct non-null values of an allow-listed column, for filter menus."""
        allowed = {"camera_id", "zone_id", "event_type", "source"}
        if column not in allowed:
            raise ValueError(f"Unsupported column {column!r}; expected one of {sorted(allowed)}")
        with self._lock:
            try:
                rows = self._connection.execute(
                    f"SELECT DISTINCT {column} AS value FROM incidents "
                    f"WHERE {column} IS NOT NULL AND {column} != '' ORDER BY value"
                ).fetchall()
            except sqlite3.Error as error:
                raise StorageError(f"Could not list {column} values: {error}") from error
        return [str(row["value"]) for row in rows]

    # -- helpers ---------------------------------------------------------
    @staticmethod
    def _build_filter(
        start_utc: str | None = None,
        end_utc: str | None = None,
        camera_id: str | None = None,
        zone_id: str | None = None,
        event_type: str | None = None,
        source: str | None = None,
    ) -> tuple[str, list[Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if start_utc:
            clauses.append("created_utc >= ?")
            params.append(start_utc)
        if end_utc:
            clauses.append("created_utc <= ?")
            params.append(end_utc)
        if camera_id:
            clauses.append("camera_id = ?")
            params.append(camera_id)
        if zone_id:
            clauses.append("zone_id = ?")
            params.append(zone_id)
        if event_type:
            clauses.append("event_type = ?")
            params.append(event_type)
        if source:
            clauses.append("source = ?")
            params.append(source)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        return where, params

    def close(self) -> None:
        with self._lock:
            try:
                self._connection.close()
            except sqlite3.Error:
                pass

    def __enter__(self) -> "IncidentStore":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def utc_iso_at(moment: Any) -> str:
    """Render an aware datetime as a stored UTC string (used by callers/tests)."""
    return moment.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )
