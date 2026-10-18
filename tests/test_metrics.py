# SPDX-License-Identifier: Apache-2.0
"""Runtime performance measurement (FR-11, AT-12).

These tests are about the measurement machinery, never about a performance
figure. The report is a container for what was observed on the machine that ran
it, so the assertions check that nothing is invented and that every number
traces back to a recorded sample.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vision_sentinel.metrics import (
    RUNTIME_MEASURED_NOTE,
    FpsMeter,
    LatencySummary,
    LatencyTracker,
    PerformanceReport,
    environment_info,
    snapshot_to_json,
)


# --- latency ----------------------------------------------------------------


def test_an_empty_series_reports_zeros() -> None:
    summary = LatencyTracker(window=10).summary()
    assert summary.count == 0
    assert summary.mean_ms == 0.0
    assert summary.p95_ms == 0.0


def test_percentiles_are_computed_from_real_samples() -> None:
    tracker = LatencyTracker(window=100)
    for value in (10.0, 20.0, 30.0, 40.0, 50.0):
        tracker.record(value)

    summary = tracker.summary()

    assert summary.count == 5
    assert summary.min_ms == pytest.approx(10.0)
    assert summary.max_ms == pytest.approx(50.0)
    assert summary.mean_ms == pytest.approx(30.0)
    assert summary.p50_ms == pytest.approx(30.0)
    assert 10.0 <= summary.p95_ms <= 50.0
    assert summary.to_dict()["mean_ms"] == pytest.approx(30.0, abs=0.001)


def test_the_window_keeps_only_the_most_recent_samples() -> None:
    tracker = LatencyTracker(window=3)
    for value in (1.0, 2.0, 3.0, 100.0):
        tracker.record(value)

    assert len(tracker) == 3
    assert tracker.total_recorded == 4
    # The oldest sample (1.0) has been dropped, so the minimum must have moved.
    assert tracker.summary().min_ms == pytest.approx(2.0)


def test_a_negative_latency_is_rejected() -> None:
    with pytest.raises(ValueError, match="negative"):
        LatencyTracker().record(-1.0)


def test_a_window_of_zero_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        LatencyTracker(window=0)


def test_reset_discards_every_sample() -> None:
    tracker = LatencyTracker(window=5)
    tracker.record(12.0)
    tracker.reset()

    assert len(tracker) == 0
    assert tracker.total_recorded == 0
    assert tracker.summary() == LatencySummary(0, 0.0, 0.0, 0.0, 0.0, 0.0)


# --- fps --------------------------------------------------------------------


def test_fps_requires_at_least_two_samples() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        FpsMeter(window=1)


def test_fps_needs_two_ticks_before_reporting_a_rate() -> None:
    meter = FpsMeter()
    meter.tick(0.0)
    assert meter.value == 0.0


def test_fps_is_computed_from_the_injected_clock() -> None:
    ticks = iter([0.0, 0.1, 0.2, 0.3])
    meter = FpsMeter(window=10, clock=lambda: next(ticks))

    for _ in range(4):
        meter.tick()

    # Four frames spanning 0.3 s is three intervals.
    assert meter.value == pytest.approx(10.0)
    assert meter.instantaneous == pytest.approx(10.0)


def test_a_stalled_clock_does_not_produce_a_rate() -> None:
    meter = FpsMeter(window=10)
    meter.tick(5.0)
    meter.tick(5.0)
    assert meter.value == 0.0


def test_fps_reset_clears_the_window() -> None:
    meter = FpsMeter(window=4)
    meter.tick(0.0)
    meter.tick(1.0)
    meter.reset()

    assert meter.value == 0.0
    assert meter.instantaneous == 0.0


# --- the report -------------------------------------------------------------


def test_recorded_frames_accumulate_stage_timings() -> None:
    report = PerformanceReport(window=5)
    report.record_frame(inference_ms=10.0, total_ms=15.0)
    report.record_frame(inference_ms=20.0, total_ms=25.0)

    assert report.frames == 2
    assert list(report.stages) == ["inference_ms", "total_ms"]
    assert report.stages["inference_ms"].latency.summary().mean_ms == pytest.approx(15.0)
    assert report.stages["total_ms"].total_ms == pytest.approx(40.0)


def test_a_dropped_frame_is_counted_but_not_timed() -> None:
    report = PerformanceReport()
    report.record_drop()
    report.record_drop()

    assert report.frames == 0
    assert report.frames_dropped == 2
    assert report.snapshot()["totals"]["frames_dropped"] == 2


def test_the_document_says_the_numbers_were_measured_at_runtime() -> None:
    report = PerformanceReport(window=5)
    report.record_frame(inference_ms=8.0, total_ms=9.0)

    document = report.snapshot()

    assert document["application"] == "VisionSentinel AI"
    assert document["measurement"] == "runtime-measured"
    assert document["note"] == RUNTIME_MEASURED_NOTE
    assert document["generated_utc"].endswith("Z")
    assert document["totals"]["frames_processed"] == 1
    assert document["stages"]["inference_ms"]["mean_ms"] == pytest.approx(8.0)
    assert document["stage_share_ms"]["inference_ms"] == pytest.approx(8.0)
    assert document["environment"]["python_version"]
    assert "openvino_devices" in document["environment"]


def test_extra_context_is_attached_and_absent_by_default() -> None:
    report = PerformanceReport()
    assert "context" not in report.snapshot()

    document = report.snapshot({"backend": "intel", "compliance_eligible": False})
    assert document["context"] == {"backend": "intel", "compliance_eligible": False}


def test_throughput_reflects_the_elapsed_window() -> None:
    report = PerformanceReport()
    # elapsed_seconds() is wall-clock, so pin the start a couple of seconds into
    # the past. Without that, the window between the two live readings below is
    # only microseconds wide and the comparison is a coin toss.
    report.started_monotonic -= 2.0
    report.record_frame(total_ms=1.0)
    rate = report.throughput_fps()

    assert rate > 0.0
    assert rate == pytest.approx(report.frames / report.elapsed_seconds(), rel=0.01)


def test_writing_the_report_is_atomic_and_leaves_no_temporary_files(
    tmp_path: Path,
) -> None:
    report = PerformanceReport(window=5)
    report.record_frame(inference_ms=4.0, total_ms=6.0)

    target = tmp_path / "evidence" / "performance_report.json"
    written = report.write(target, {"backend": "fake"})

    assert written == target
    assert [path.name for path in target.parent.iterdir()] == [target.name]

    document = json.loads(target.read_text(encoding="utf-8"))
    assert document["totals"]["frames_processed"] == 1
    assert document["context"]["backend"] == "fake"


def test_writing_without_a_configured_path_is_an_error() -> None:
    with pytest.raises(ValueError, match="No report path"):
        PerformanceReport().write()


def test_the_configured_path_is_used_when_none_is_given(tmp_path: Path) -> None:
    target = tmp_path / "configured.json"
    written = PerformanceReport(report_path=target).write()
    assert written == target
    assert target.exists()


def test_reset_clears_stages_and_counters(tmp_path: Path) -> None:
    report = PerformanceReport(report_path=tmp_path / "report.json")
    report.record_frame(inference_ms=1.0)
    report.record_drop()
    report.reset()

    assert report.frames == 0
    assert report.frames_dropped == 0
    assert report.stages == {}


def test_snapshot_to_json_writes_plain_json(tmp_path: Path) -> None:
    target = snapshot_to_json({"application": "VisionSentinel AI"}, tmp_path / "x.json")
    assert json.loads(target.read_text(encoding="utf-8")) == {
        "application": "VisionSentinel AI"
    }


def test_environment_info_describes_the_machine() -> None:
    info = environment_info()
    assert info["python_version"]
    assert info["platform"]
    assert info["cpu_count"] is None or info["cpu_count"] >= 1
