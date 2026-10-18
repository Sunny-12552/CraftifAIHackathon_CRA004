# SPDX-License-Identifier: Apache-2.0
"""Configuration loading and validation (FR-10, AT-09)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from vision_sentinel.config import (
    ConfigError,
    ZoneConfig,
    load_settings,
    parse_settings,
    save_zones,
)


def test_the_shipped_default_settings_load() -> None:
    settings = load_settings()
    assert settings.video.source_type == "webcam"
    assert settings.detection.backend == "auto"
    assert len(settings.zones) >= 1
    assert settings.alerts.cooldown_seconds > 0


def test_auto_backend_never_selects_a_non_compliant_path() -> None:
    """``auto`` must resolve to Intel then CPU only.

    Selecting ``nvidia_ort`` automatically would risk an ONNX Runtime run being
    mistaken for the CraftifAI PipeGen branch.
    """
    settings = load_settings()
    assert settings.resolved_backend() == ("intel", "cpu")
    assert "nvidia_ort" not in settings.resolved_backend()
    assert "pipegen" not in settings.resolved_backend()


def test_explicit_backend_is_respected() -> None:
    settings = parse_settings({"detection": {"backend": "nvidia_ort"}})
    assert settings.resolved_backend() == ("nvidia_ort",)


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ({"detection": {"confidence_threshold": 0.0}}, "confidence_threshold"),
        ({"detection": {"confidence_threshold": 1.5}}, "confidence_threshold"),
        ({"detection": {"nms_threshold": 0}}, "nms_threshold"),
        ({"detection": {"backend": "tensorrt"}}, "detection.backend"),
        ({"detection": {"model_size": 630}}, "divisible"),
        ({"detection": {"model_size": 0}}, "positive"),
        ({"detection": {"strides": []}}, "strides"),
        ({"video": {"width": 0}}, "positive"),
        ({"video": {"camera_index": -1}}, "zero or greater"),
        ({"video": {"source_type": "carrier-pigeon"}}, "source_type"),
        ({"video": {"source_type": "file"}}, "file_path is required"),
        ({"video": {"source_type": "rtsp"}}, "rtsp_url is required"),
        ({"alerts": {"enter_frames": 0}}, "enter_frames"),
        ({"alerts": {"exit_frames": 0}}, "exit_frames"),
        ({"alerts": {"cooldown_seconds": -1}}, "cooldown_seconds"),
        ({"alerts": {"min_confidence": 2}}, "min_confidence"),
        ({"tracking": {"iou_threshold": 2}}, "iou_threshold"),
        ({"tracking": {"max_age": -1}}, "max_age"),
        ({"tracking": {"min_hits": 0}}, "min_hits"),
        ({"metrics": {"window": 0}}, "window"),
        ({"storage": {"retention_days": -2}}, "retention_days"),
    ],
)
def test_invalid_values_are_rejected_with_a_useful_message(raw: dict, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        parse_settings(raw)


def test_invalid_zone_polygon_is_rejected() -> None:
    raw = {
        "zones": [
            {"id": "z", "name": "Z", "polygon_normalized": [[0.1, 0.1], [0.2, 0.2]]}
        ]
    }
    with pytest.raises(ConfigError, match="at least three points"):
        parse_settings(raw)


def test_duplicate_zone_ids_are_rejected() -> None:
    polygon = [[0.1, 0.1], [0.5, 0.1], [0.5, 0.5]]
    raw = {
        "zones": [
            {"id": "same", "polygon_normalized": polygon},
            {"id": "same", "polygon_normalized": polygon},
        ]
    }
    with pytest.raises(ConfigError, match="Duplicate zone id"):
        parse_settings(raw)


def test_non_mapping_sections_are_rejected() -> None:
    with pytest.raises(ConfigError, match="video must be a mapping"):
        parse_settings({"video": ["not", "a", "mapping"]})


def test_malformed_yaml_produces_a_clear_error(tmp_path: Path) -> None:
    bad = tmp_path / "settings.yaml"
    bad.write_text("video: [unclosed\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="not valid YAML"):
        load_settings(bad)


def test_missing_explicit_config_file_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_settings(tmp_path / "absent.yaml")


def test_environment_overrides_are_applied(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VS_DEVICE", "cpu")
    monkeypatch.setenv("VS_CAMERA_INDEX", "2")
    monkeypatch.setenv("VS_SOURCE_TYPE", "file")
    monkeypatch.setenv("VS_VIDEO_FILE", "clip.mp4")
    monkeypatch.setenv("VS_DB_PATH", "custom.db")
    monkeypatch.setenv("VS_PIPEGEN_ENDPOINT", "127.0.0.1:9999")

    settings = load_settings()

    assert settings.detection.backend == "cpu"
    assert settings.video.camera_index == 2
    assert settings.video.source_type == "file"
    assert settings.video.file_path == "clip.mp4"
    assert settings.storage.db_path == "custom.db"
    assert settings.detection.pipegen_endpoint == "127.0.0.1:9999"


def test_invalid_environment_override_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VS_DEVICE", "quantum")
    with pytest.raises(ConfigError, match="VS_DEVICE"):
        load_settings()


def test_disabled_zones_are_filtered_from_the_active_set() -> None:
    polygon = [[0.1, 0.1], [0.5, 0.1], [0.5, 0.5]]
    settings = parse_settings(
        {
            "zones": [
                {"id": "on", "polygon_normalized": polygon, "enabled": True},
                {"id": "off", "polygon_normalized": polygon, "enabled": False},
            ]
        }
    )
    assert [zone.zone_id for zone in settings.enabled_zones()] == ["on"]


def test_resolved_paths_default_outside_the_synced_project(tmp_path: Path) -> None:
    settings = parse_settings({"storage": {"db_path": str(tmp_path / "x.db")}})
    assert settings.storage.resolved_db_path() == tmp_path / "x.db"

    default_settings = parse_settings({})
    assert default_settings.storage.resolved_db_path().name == "incidents.db"


def test_save_zones_round_trips_without_losing_other_settings(tmp_path: Path) -> None:
    path = tmp_path / "settings.yaml"
    path.write_text(
        yaml.safe_dump(
            {"video": {"source_type": "webcam", "camera_index": 1}},
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    zones = [
        ZoneConfig(
            zone_id="zone-9",
            name="Edited Zone",
            polygon_normalized=((0.1, 0.1), (0.9, 0.1), (0.5, 0.9)),
        )
    ]
    save_zones(path, zones)

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert payload["video"]["camera_index"] == 1
    assert payload["zones"][0]["id"] == "zone-9"

    reloaded = load_settings(path)
    assert reloaded.zones[0].polygon_normalized == ((0.1, 0.1), (0.9, 0.1), (0.5, 0.9))


def test_invalid_polygon_is_not_written_to_disk(tmp_path: Path) -> None:
    path = tmp_path / "settings.yaml"
    with pytest.raises(ValueError):
        save_zones(
            path,
            [ZoneConfig("bad", "Bad", ((0.0, 0.0), (0.1, 0.1)))],
        )
    assert not path.exists()
