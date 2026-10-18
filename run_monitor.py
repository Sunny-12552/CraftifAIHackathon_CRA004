# SPDX-License-Identifier: Apache-2.0
"""Native desktop viewer for VisionSentinel AI.

This is the OpenCV front-end: one process, one thread, one window. It drives
:class:`~vision_sentinel.engine.SentinelEngine` synchronously so that drawing and
key handling happen in the thread that owns the window, which is what OpenCV's
highgui backend requires on Windows.

    python run_monitor.py
    python run_monitor.py --source-type file --file clip.mp4
    python run_monitor.py --backend cpu --camera-index 1
    python run_monitor.py --headless --max-frames 200

The dashboard in ``app.py`` is the other front-end. Both share the same engine,
the same configuration and the same incident database; they differ only in how
the operator sees it.

Press ``q`` or ``Esc`` in the window to stop. The session ends by writing a
runtime-measured performance report and a JSON incident export under
``evidence/`` and the configured export directory (unless ``--no-evidence``).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vision_sentinel.config import (  # noqa: E402
    DETECTION_BACKENDS,
    VIDEO_SOURCE_TYPES,
    ConfigError,
    Settings,
    load_settings,
)
from vision_sentinel.detection import DetectorError  # noqa: E402
from vision_sentinel.engine import SentinelEngine  # noqa: E402
from vision_sentinel.storage import default_export_path, export_incidents  # noqa: E402
from vision_sentinel.video import VideoSourceError  # noqa: E402

WINDOW_NAME = "VisionSentinel AI"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run VisionSentinel AI in a native OpenCV window.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to settings.yaml (defaults to config/settings.yaml).",
    )
    parser.add_argument(
        "--backend",
        choices=DETECTION_BACKENDS,
        default=None,
        help="Override detection.backend. 'auto' = Intel GPU, else CPU. "
        "'nvidia_ort' and 'pipegen' must be requested explicitly.",
    )
    parser.add_argument(
        "--source-type",
        choices=VIDEO_SOURCE_TYPES,
        default=None,
        help="Override video.source_type.",
    )
    parser.add_argument("--file", default=None, help="Video file for source_type=file.")
    parser.add_argument("--camera-index", type=int, default=None, help="Webcam index.")
    parser.add_argument(
        "--fps-limit",
        type=float,
        default=None,
        help="Cap the processing rate (0 = uncapped).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after this many frames (0 = run until the source ends or you quit).",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Do not open a window; print one line per frame instead.",
    )
    parser.add_argument(
        "--no-overlay",
        action="store_true",
        help="Skip rendering; useful with --headless to measure inference alone.",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Never download the model; fail if it is not already present.",
    )
    parser.add_argument(
        "--no-evidence",
        action="store_true",
        help="Do not write the performance report or the JSON incident export.",
    )
    return parser.parse_args(argv)


def apply_overrides(settings: Settings, args: argparse.Namespace) -> Settings:
    """Fold the CLI flags into a validated Settings object."""
    video = settings.video
    if args.source_type is not None:
        video = replace(video, source_type=args.source_type)
    if args.file is not None:
        video = replace(video, file_path=args.file)
    if args.camera_index is not None:
        video = replace(video, camera_index=args.camera_index, camera_id=f"cam-{args.camera_index}")
    if args.fps_limit is not None:
        video = replace(video, fps_limit=args.fps_limit)

    detection = settings.detection
    if args.backend is not None:
        detection = replace(detection, backend=args.backend)

    return settings.with_changes(video=video, detection=detection)


def _print_startup(engine: SentinelEngine) -> None:
    source = engine.source.describe()
    print(f"VisionSentinel AI  |  backend: {engine.backend_id} on {engine.device_name}")
    print(f"  source         : {engine._source_description()} ({source.get('type')})")
    print(f"  compliance     : {'eligible' if engine.compliance_eligible else 'not eligible'}")
    for zone in engine.zones:
        state = "enabled" if zone.enabled else "disabled"
        print(f"  zone {zone.zone_id:<8} : {zone.name} ({state})")
    print(f"  enter/exit     : {engine.monitor.enter_frames}/{engine.monitor.exit_frames} frames")
    for note in engine.notes:
        print(f"  note           : {note}")
    if not engine.compliance_eligible:
        print("  The detection backend is not the CraftifAI PipeGen pipeline, so this run")
        print("  cannot be used as hackathon (Track A) compliance evidence.")
    print()


def _run_viewer(engine: SentinelEngine, max_frames: int) -> int:
    import cv2

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    counter = {"frames": 0}

    def on_frame(result) -> bool | None:
        counter["frames"] += 1
        if result.annotated is not None:
            cv2.imshow(WINDOW_NAME, result.annotated)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            return False
        if max_frames and counter["frames"] >= max_frames:
            return False
        return None

    try:
        engine.run_forever(on_frame=on_frame)
    finally:
        cv2.destroyAllWindows()
    return 0


def _run_headless(engine: SentinelEngine, max_frames: int) -> int:
    counter = {"frames": 0}

    def on_frame(result) -> bool | None:
        counter["frames"] += 1
        occupancy = ", ".join(f"{k}={v}" for k, v in result.occupancy.items()) or "none"
        line = (
            f"frame {result.frame_id:>6} | detections {len(result.detections):>2} | "
            f"occupancy {occupancy:<16} | {result.total_ms:6.1f} ms"
        )
        if result.error:
            line += f" | ERROR: {result.error}"
        print(line)
        if max_frames and counter["frames"] >= max_frames:
            return False
        return None

    engine.run_forever(on_frame=on_frame)
    return 0


def _write_session_evidence(engine: SentinelEngine) -> None:
    """Persist the runtime-measured report and the incident export."""
    try:
        report = engine.write_performance_report()
        print(f"Performance report : {report}")
    except Exception as error:  # pragma: no cover - evidence is best-effort
        print(f"Could not write the performance report: {error}")

    if engine.store is None:
        return
    try:
        target = default_export_path(engine.settings.storage.resolved_export_dir())
        result = export_incidents(engine.store, target)
        print(f"Incident export    : {result.describe()}")
    except Exception as error:  # pragma: no cover - evidence is best-effort
        print(f"Could not export the incident history: {error}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        settings = load_settings(args.config)
    except ConfigError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 2
    settings = apply_overrides(settings, args)

    try:
        engine = SentinelEngine(
            settings,
            allow_download=not args.no_download,
            render=not args.no_overlay,
        )
    except DetectorError as error:
        print(f"Could not start the detector:\n{error}", file=sys.stderr)
        return 3
    except VideoSourceError as error:
        print(f"Could not configure the video source:\n{error}", file=sys.stderr)
        return 4

    _print_startup(engine)

    try:
        if args.headless:
            code = _run_headless(engine, args.max_frames)
        else:
            code = _run_viewer(engine, args.max_frames)
    except VideoSourceError as error:
        print(f"Video source failed:\n{error}", file=sys.stderr)
        code = 1
    except KeyboardInterrupt:
        print("\nStopped by the operator.")
        code = 0
    finally:
        engine.stop()

    stats = engine.alerts.stats()
    print()
    print(
        f"Processed {engine.metrics.frames} frame(s): "
        f"{stats['entries']} entr(y/ies), {stats['exits']} exit(s), "
        f"{stats['incidents']} incident(s), "
        f"{stats['suppressed_duplicates']} suppressed duplicate(s)."
    )
    occupancy = engine.monitor.occupancy()
    if occupancy:
        print("Final occupancy: " + ", ".join(f"{k}={v}" for k, v in occupancy.items()))

    if not args.no_evidence:
        _write_session_evidence(engine)

    engine.close()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
