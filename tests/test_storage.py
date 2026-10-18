# SPDX-License-Identifier: Apache-2.0
"""SQLite incident storage and JSON export (FR-06, FR-07, AT-07, AT-08)."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from vision_sentinel.events import Incident, ZoneEvent, ZoneEventType
from vision_sentinel.storage import (
    IncidentStore,
    StorageError,
    build_export,
    default_export_path,
    export_incidents,
    load_export,
    validate_export,
)


def make_incident(
    suffix: str = "1",
    *,
    event_type: str = "person_entered_zone",
    camera_id: str = "cam-test",
    zone_id: str = "zone-1",
    created: datetime | None = None,
    source: str = "fake",
) -> Incident:
    moment = created or datetime(2026, 10, 17, 9, 30, 0, tzinfo=timezone.utc)
    stamp = moment.isoformat(timespec="milliseconds").replace("+00:00", "Z")
    event = ZoneEvent(
        event_type=ZoneEventType.ENTERED
        if event_type == "person_entered_zone"
        else ZoneEventType.EXITED,
        timestamp_utc=stamp,
        camera_id=camera_id,
        zone_id=zone_id,
        zone_name="Zone A",
        track_id=suffix,
        label="person",
        confidence=0.91,
        bbox_xywh=(140, 60, 40, 100),
        foot_point=(160.0, 160.0),
        people_count=1,
        source=source,
        frame_id=10,
    )
    return Incident.from_zone_event(event, id_factory=lambda: f"INC-TEST-{suffix}")


@pytest.fixture
def store(tmp_path: Path):
    incident_store = IncidentStore(tmp_path / "incidents.db")
    yield incident_store
    incident_store.close()


def test_an_incident_round_trips_through_the_database(store: IncidentStore) -> None:
    incident = make_incident("1")
    store.append(incident)

    rows = store.query()
    assert len(rows) == 1
    assert rows[0] == incident


def test_multiple_incidents_are_stored_in_order(store: IncidentStore) -> None:
    base = datetime(2026, 10, 17, 9, 0, 0, tzinfo=timezone.utc)
    for index in range(5):
        store.append(make_incident(str(index), created=base + timedelta(minutes=index)))

    newest = store.query()
    oldest = store.query(newest_first=False)

    assert [item.incident_id for item in newest] == [
        "INC-TEST-4", "INC-TEST-3", "INC-TEST-2", "INC-TEST-1", "INC-TEST-0"
    ]
    assert [item.incident_id for item in oldest] == [
        "INC-TEST-0", "INC-TEST-1", "INC-TEST-2", "INC-TEST-3", "INC-TEST-4"
    ]


def test_duplicate_identifiers_are_rejected(store: IncidentStore) -> None:
    store.append(make_incident("1"))
    with pytest.raises(StorageError, match="already exists"):
        store.append(make_incident("1"))


def test_filters_narrow_the_result(store: IncidentStore) -> None:
    base = datetime(2026, 10, 17, 9, 0, 0, tzinfo=timezone.utc)
    store.append(make_incident("a", zone_id="zone-1", created=base))
    store.append(make_incident("b", zone_id="zone-2", created=base + timedelta(hours=1)))
    store.append(
        make_incident(
            "c",
            zone_id="zone-9",
            camera_id="cam-other",
            event_type="person_exited_zone",
            created=base + timedelta(hours=2),
        )
    )

    assert {item.incident_id for item in store.query(zone_id="zone-1")} == {"INC-TEST-a"}
    assert {item.incident_id for item in store.query(camera_id="cam-other")} == {"INC-TEST-c"}
    assert {
        item.incident_id
        for item in store.query(event_type="person_exited_zone")
    } == {"INC-TEST-c"}

    window_start = (base + timedelta(minutes=30)).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    assert {item.incident_id for item in store.query(start_utc=window_start)} == {
        "INC-TEST-b",
        "INC-TEST-c",
    }


def test_count_and_stats_agree_with_the_rows(store: IncidentStore) -> None:
    base = datetime(2026, 10, 17, 9, 0, 0, tzinfo=timezone.utc)
    store.append(make_incident("a", created=base))
    store.append(make_incident("b", created=base + timedelta(minutes=1)))
    store.append(
        make_incident(
            "c", zone_id="zone-9", event_type="person_exited_zone",
            created=base + timedelta(minutes=2),
        )
    )

    assert store.count() == 3
    assert store.count(zone_id="zone-9") == 1

    stats = store.stats()
    assert stats["total"] == 3
    assert stats["by_event_type"]["person_entered_zone"] == 2
    assert stats["by_event_type"]["person_exited_zone"] == 1
    assert stats["by_camera"]["cam-test"] == 3
    assert stats["first_utc"] is not None
    assert stats["last_utc"] is not None


def test_distinct_values_are_available_for_filter_menus(store: IncidentStore) -> None:
    store.append(make_incident("a", zone_id="zone-1"))
    store.append(make_incident("b", zone_id="zone-2"))

    assert store.distinct("zone_id") == ["zone-1", "zone-2"]
    with pytest.raises(ValueError, match="Unsupported column"):
        store.distinct("incident_id")


def test_retention_pruning_removes_only_old_rows(store: IncidentStore) -> None:
    now = datetime.now(timezone.utc)
    store.append(make_incident("old", created=now - timedelta(days=30)))
    store.append(make_incident("new", created=now))

    removed = store.prune(retention_days=7)

    assert removed == 1
    assert [item.incident_id for item in store.query()] == ["INC-TEST-new"]


def test_prune_is_a_no_op_when_retention_is_disabled(store: IncidentStore) -> None:
    store.append(make_incident("a"))
    assert store.prune(retention_days=0) == 0
    assert store.count() == 1


def test_wal_journal_mode_is_enabled(tmp_path: Path) -> None:
    """The dashboard reads while the engine writes, so WAL is required."""
    with IncidentStore(tmp_path / "incidents.db") as incident_store:
        mode = incident_store._connection.execute("PRAGMA journal_mode").fetchone()[0]
        assert str(mode).lower() == "wal"


def test_schema_version_is_recorded(tmp_path: Path) -> None:
    with IncidentStore(tmp_path / "incidents.db") as incident_store:
        row = incident_store._connection.execute(
            "SELECT version FROM schema_info"
        ).fetchone()
        assert int(row["version"]) == 1


def test_an_unusable_path_produces_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(StorageError):
        IncidentStore(tmp_path)  # a directory is not a database file


def test_an_incompatible_schema_version_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "incidents.db"
    with IncidentStore(path):
        pass
    connection = sqlite3.connect(str(path))
    connection.execute("UPDATE schema_info SET version = 99")
    connection.commit()
    connection.close()

    with pytest.raises(StorageError, match="schema version"):
        IncidentStore(path)


# --- JSON export ------------------------------------------------------------


def test_export_produces_a_valid_document(store: IncidentStore, tmp_path: Path) -> None:
    base = datetime(2026, 10, 17, 9, 0, 0, tzinfo=timezone.utc)
    for index in range(3):
        store.append(make_incident(str(index), created=base + timedelta(minutes=index)))

    target = tmp_path / "exports" / "incidents.json"
    result = export_incidents(store, target)

    assert result.count == 3
    assert result.path == target
    assert target.exists()

    payload = load_export(target)  # validates as it reads
    assert payload["application"] == "VisionSentinel AI"
    assert payload["count"] == 3
    assert len(payload["incidents"]) == 3
    assert payload["exported_utc"].endswith("Z")


def test_export_honours_filters(store: IncidentStore, tmp_path: Path) -> None:
    store.append(make_incident("a", zone_id="zone-1"))
    store.append(make_incident("b", zone_id="zone-2"))

    result = export_incidents(store, tmp_path / "filtered.json", zone_id="zone-2")
    payload = load_export(result.path)

    assert result.count == 1
    assert payload["filters"]["zone_id"] == "zone-2"
    assert payload["incidents"][0]["incident_id"] == "INC-TEST-b"


def test_export_of_an_empty_history_is_still_valid(store: IncidentStore, tmp_path: Path) -> None:
    result = export_incidents(store, tmp_path / "empty.json")
    payload = load_export(result.path)

    assert result.count == 0
    assert payload["incidents"] == []


def test_default_export_path_is_timestamped(tmp_path: Path) -> None:
    moment = datetime(2026, 10, 17, 9, 30, 5, tzinfo=timezone.utc)
    path = default_export_path(tmp_path, moment)
    assert path.name == "incidents_20261017T093005Z.json"


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"application": "x", "export_format_version": 1, "schema_version": 1,
         "exported_utc": "2026-10-17T09:30:00.000Z", "count": 1, "incidents": []},
        {"application": "x", "export_format_version": 1, "schema_version": 1,
         "exported_utc": "2026-10-17T09:30:00.000Z", "count": 1,
         "incidents": [{"incident_id": "INC-1"}]},
        {"application": "x", "export_format_version": 1, "schema_version": 1,
         "exported_utc": "not-a-time", "count": 0, "incidents": []},
    ],
)
def test_invalid_export_documents_are_rejected(payload: dict) -> None:
    with pytest.raises(ValueError):
        validate_export(payload)


def test_export_rejects_duplicate_incident_ids() -> None:
    incident = make_incident("dup")
    payload = build_export([incident, incident])
    with pytest.raises(ValueError, match="Duplicate incident_id"):
        validate_export(payload)


def test_export_rejects_out_of_range_confidence() -> None:
    incident = make_incident("a")
    payload = build_export([incident])
    payload["incidents"][0]["confidence"] = 4.2
    with pytest.raises(ValueError, match="confidence"):
        validate_export(payload)


def test_export_writes_are_atomic_and_leave_no_temporary_files(
    store: IncidentStore, tmp_path: Path
) -> None:
    store.append(make_incident("a"))
    export_dir = tmp_path / "exports"
    export_incidents(store, export_dir / "out.json")

    assert [path.name for path in export_dir.iterdir()] == ["out.json"]


def test_exported_json_is_plain_serialisable_data(store: IncidentStore, tmp_path: Path) -> None:
    store.append(make_incident("a"))
    result = export_incidents(store, tmp_path / "out.json")

    # Must be re-readable by any JSON consumer, with no Python-specific types.
    with result.path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    assert isinstance(raw["incidents"][0]["bbox_xywh"], list)
    assert all(isinstance(value, (int, float)) for value in raw["incidents"][0]["bbox_xywh"])
    assert raw["incidents"][0]["snapshot_path"] is None
    assert raw["incidents"][0]["schema_version"] == 1
