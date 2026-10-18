# SPDX-License-Identifier: Apache-2.0
"""Streamlit dashboard for VisionSentinel AI.

This is the operator front-end: a live annotated view, the aggregate KPIs, the
incident log, the zone editor and the runtime performance evidence. It drives the
very same :class:`~vision_sentinel.engine.SentinelEngine` as the native OpenCV
viewer in ``run_monitor.py`` -- only the presentation differs.

    streamlit run app.py

The engine runs on a background thread; the page reads
:meth:`SentinelEngine.snapshot` from a timer-driven fragment so the live view
refreshes without rerunning the whole script (and without fighting the engine
thread for the camera).

Design notes
------------
* The engine lives in ``st.session_state`` and is created only when the operator
  chooses **Start monitoring**, so opening the page never grabs the camera.
* Every frame, incident, latency number and compliance flag shown here is read
  from the engine. The dashboard fabricates nothing.
* A backend that is not the genuine CraftifAI PipeGen pipeline is presented as
  *not* compliance-eligible, prominently, on every view.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from vision_sentinel import __version__ as APP_VERSION  # noqa: E402
from vision_sentinel.config import (  # noqa: E402
    DETECTION_BACKENDS,
    VIDEO_SOURCE_TYPES,
    ConfigError,
    Settings,
    ZoneConfig,
    load_settings,
    save_zones,
)
from vision_sentinel.detection import DetectorError, PIPEGEN_CHECKLIST  # noqa: E402
from vision_sentinel.engine import SentinelEngine  # noqa: E402
from vision_sentinel.events import ZoneEventType  # noqa: E402
from vision_sentinel.metrics import RUNTIME_MEASURED_NOTE, environment_info  # noqa: E402
from vision_sentinel.storage import (  # noqa: E402
    StorageError,
    default_export_path,
    export_incidents,
)
from vision_sentinel.video import VideoSourceError  # noqa: E402
from vision_sentinel.zones import build_zones  # noqa: E402

st.set_page_config(
    page_title="VisionSentinel AI",
    page_icon=":material/security:",
    layout="wide",
)

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "settings.yaml"

VIEWS = ("Live monitor", "Incidents", "Zones", "Performance", "Help & compliance")

_EVENT_LABELS = {
    "Entered": ZoneEventType.ENTERED.value,
    "Exited": ZoneEventType.EXITED.value,
}


# ---------------------------------------------------------------------------
# Session state and engine lifecycle
# ---------------------------------------------------------------------------
def _init_state() -> None:
    defaults = {
        "engine": None,
        "engine_error": None,
        "engine_signature": None,
        "last_message": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _session_engine() -> SentinelEngine | None:
    engine = st.session_state.get("engine")
    return engine if isinstance(engine, SentinelEngine) else None


def _dispose_engine() -> None:
    """Close and forget the engine, releasing the camera and the database."""
    engine = st.session_state.pop("engine", None)
    st.session_state["engine_signature"] = None
    if engine is not None:
        try:
            engine.close()
        except Exception:  # pragma: no cover - release is best-effort
            pass


def _effective_settings(
    config_path: str,
    source_type: str,
    camera_index: int,
    video_file: str,
    rtsp_url: str,
    backend: str,
) -> Settings:
    """Load the configuration file and fold in the sidebar overrides."""
    settings = load_settings(config_path.strip() or None)

    video = replace(settings.video, source_type=source_type)
    if source_type == "webcam":
        video = replace(
            video, camera_index=int(camera_index), camera_id=f"cam-{int(camera_index)}"
        )
    elif source_type == "file":
        if not video_file.strip():
            raise ConfigError("Choose a video file when the source type is 'file'.")
        video = replace(video, file_path=video_file.strip())
    elif source_type == "rtsp":
        if not rtsp_url.strip():
            raise ConfigError("Enter an RTSP URL when the source type is 'rtsp'.")
        video = replace(video, rtsp_url=rtsp_url.strip())

    detection = replace(settings.detection, backend=backend)
    return settings.with_changes(video=video, detection=detection)


def _start_engine(settings: Settings, allow_download: bool) -> None:
    """Build a fresh engine from the current settings and start its worker."""
    _dispose_engine()
    st.session_state["engine_error"] = None
    try:
        engine = SentinelEngine(settings, allow_download=allow_download, render=True)
        engine.start()
    except (DetectorError, VideoSourceError, ConfigError, StorageError) as error:
        st.session_state["engine"] = None
        st.session_state["engine_error"] = str(error)
        return
    st.session_state["engine"] = engine
    st.session_state["engine_signature"] = (
        f"{settings.detection.backend}|{settings.video.source_type}"
    )


def _stop_engine() -> None:
    engine = _session_engine()
    if engine is not None:
        engine.stop()


# ---------------------------------------------------------------------------
# Small formatting helpers
# ---------------------------------------------------------------------------
def _zone_lookup(settings: Settings | None) -> dict[str, str]:
    if settings is None:
        return {}
    return {zone.zone_id: zone.name for zone in settings.zones}


def _incident_rows(incidents) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for incident in incidents:
        rows.append(
            {
                "Incident": incident.incident_id,
                "Time (UTC)": incident.created_utc,
                "Event": incident.event_type,
                "Zone": incident.zone_name or incident.zone_id,
                "Track": incident.track_id or "-",
                "Label": incident.label,
                "Confidence": round(float(incident.confidence), 3),
                "Box (x,y,w,h)": ", ".join(str(int(v)) for v in incident.bbox_xywh),
                "Foot point": f"{incident.foot_point[0]:.1f}, {incident.foot_point[1]:.1f}",
                "Source": incident.source,
                "Frame": int(incident.frame_id),
            }
        )
    return rows


def _compliance_badge(eligible: bool) -> None:
    if eligible:
        st.badge("CraftifAI PipeGen pipeline — compliance-eligible", color="green")
    else:
        st.badge("Not compliance-eligible (not the PipeGen pipeline)", color="orange")


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------
@st.fragment(run_every="1s")
def render_live_view() -> None:
    engine = _session_engine()
    if engine is None:
        error = st.session_state.get("engine_error")
        if error:
            st.error(error, title="Could not start the engine")
        else:
            st.info(
                "No engine is running. Configure the source in the sidebar and "
                "choose **Start monitoring**.",
                icon=":material/info:",
            )
        return

    snapshot = engine.snapshot()

    with st.container(horizontal=True):
        st.metric("Status", snapshot.status, border=True)
        st.metric("FPS", f"{snapshot.fps:.1f}", border=True)
        st.metric("Inference", f"{snapshot.inference_ms:.0f} ms", border=True)
        st.metric("Frames", f"{snapshot.frames_processed}", border=True)
        st.metric("Incidents", f"{snapshot.incident_count}", border=True)
        st.metric("Active alerts", f"{len(snapshot.active_alerts)}", border=True)

    if snapshot.error:
        st.error(
            f"The detection backend reported a problem: {snapshot.error}",
            icon=":material/warning:",
        )
    if not snapshot.compliance_eligible:
        st.caption(
            "This run uses a local backend, not the CraftifAI PipeGen pipeline, so "
            "it cannot be used as Track A compliance evidence."
        )

    left, right = st.columns([2, 1], gap="large")

    with left:
        with st.container(border=True):
            st.markdown(f"**Live view** — {snapshot.source} on {snapshot.device or snapshot.backend}")
            if snapshot.frame is not None:
                st.image(
                    snapshot.frame,
                    channels="BGR",
                    width="stretch",
                    caption=(
                        f"Frame {snapshot.frame_id} — restricted zones, detections "
                        "and foot points"
                    ),
                    alt="Annotated camera frame with restricted zones and detected people",
                )
            else:
                st.info("Waiting for the first frame from the source…", icon=":material/hourglass_empty:")

    with right:
        with st.container(border=True):
            st.markdown("**Occupancy**")
            occupancy = snapshot.occupancy
            if occupancy:
                names = {zone.zone_id: zone.name for zone in engine.zones}
                for zone_id, count in occupancy.items():
                    st.metric(names.get(zone_id, zone_id), count, border=True)
            else:
                st.caption("No zones are configured.")

        with st.container(border=True):
            st.markdown("**Active alerts**")
            if snapshot.active_alerts:
                for incident in snapshot.active_alerts:
                    st.markdown(
                        f":red[**{incident.zone_name or incident.zone_id}**] — "
                        f"{incident.label} #{incident.track_id or 'anon'} "
                        f"({incident.confidence:.0%})"
                    )
                    st.caption(f"{incident.created_utc} · {incident.incident_id}")
            else:
                st.caption("No confirmed intrusions in the restricted zones.")

        detections = snapshot.detections
        with st.container(border=True):
            st.markdown(f"**Detections this frame** ({len(detections)})")
            if detections:
                st.dataframe(
                    [
                        {
                            "Label": d.label,
                            "Confidence": round(float(d.confidence), 3),
                            "Track": d.track_id or "-",
                            "Box": ", ".join(str(int(v)) for v in d.bbox_xywh),
                        }
                        for d in detections
                    ],
                    width="stretch",
                    hide_index=True,
                    alt="Detections in the current frame",
                )
            else:
                st.caption("Nothing detected in this frame.")


@st.fragment(run_every="5s")
def render_incidents_view() -> None:
    engine = _session_engine()
    if engine is None:
        st.info("Start monitoring to record incidents.", icon=":material/info:")
        return

    history = list(engine.alerts.history())
    stats = engine.alerts.stats()

    with st.container(horizontal=True):
        st.metric("Entries", stats["entries"], border=True)
        st.metric("Exits", stats["exits"], border=True)
        st.metric("Incidents", stats["incidents"], border=True)
        st.metric("Suppressed duplicates", stats["suppressed_duplicates"], border=True)
        st.metric("Active", stats["active_alerts"], border=True)

    with st.container(horizontal=True):
        event_labels = st.pills(
            "Event type",
            options=list(_EVENT_LABELS),
            selection_mode="multi",
            default=list(_EVENT_LABELS),
            key="incident_event_filter",
        )
        zone_names = sorted({i.zone_name or i.zone_id for i in history})
        zone_choice = st.selectbox(
            "Zone", ["All zones", *zone_names], key="incident_zone_filter"
        )

    selected_events = {_EVENT_LABELS[label] for label in (event_labels or [])}
    filtered = [
        i
        for i in history
        if i.event_type in selected_events
        and (zone_choice == "All zones" or (i.zone_name or i.zone_id) == zone_choice)
    ]
    filtered.reverse()  # newest first

    if filtered:
        st.dataframe(
            _incident_rows(filtered),
            width="stretch",
            hide_index=True,
            alt="Recorded zone incidents for this session",
        )
    else:
        st.caption("No incidents match the current filters.")

    with st.container(horizontal=True):
        st.download_button(
            "Download session incidents (JSON)",
            data=engine.incidents_json(),
            file_name="vision_sentinel_session_incidents.json",
            mime="application/json",
            icon=":material/download:",
        )
        if st.button("Export session to file", icon=":material/save:"):
            if engine.store is None:
                st.warning("This engine has no incident store attached.")
            else:
                try:
                    target = default_export_path(engine.settings.storage.resolved_export_dir())
                    result = export_incidents(engine.store, target)
                    st.success(result.describe())
                except (StorageError, OSError) as error:
                    st.error(f"Could not export the incidents: {error}")

    with st.expander("Incident database", expanded=False):
        if engine.store is None:
            st.caption("No persistent store is attached to this engine.")
            return
        try:
            store_stats = engine.store.stats()
        except StorageError as error:
            st.error(f"Could not read the incident database: {error}")
            return
        st.markdown(f"**Database:** `{store_stats['db_path']}`")
        st.metric("Total persisted incidents", store_stats["total"], border=True)
        if store_stats["by_zone"]:
            st.dataframe(
                [
                    {"Zone": zone, "Incidents": count}
                    for zone, count in store_stats["by_zone"].items()
                ],
                width="stretch",
                hide_index=True,
                alt="Persisted incidents per zone",
            )
        if store_stats["first_utc"]:
            st.caption(
                f"Window: {store_stats['first_utc']} → {store_stats['last_utc']}"
            )


def render_zones_view() -> None:
    config_path = st.session_state.get("config_path") or str(DEFAULT_CONFIG_PATH)
    try:
        settings = load_settings(config_path.strip() or None)
    except ConfigError as error:
        st.error(f"Could not load the configuration: {error}")
        return

    if not settings.zones:
        st.info(
            "No restricted zones are configured. Add at least one polygon under "
            "`zones:` in config/settings.yaml.",
            icon=":material/select_all:",
        )
        return

    st.caption(
        "Vertices are normalised (0.0 - 1.0) so a zone stays on the same physical "
        "area at any capture resolution. Saving writes them back to the "
        "configuration file and applies them to the running engine."
    )

    labels = {f"{zone.zone_id} — {zone.name}": zone for zone in settings.zones}
    choice = st.selectbox("Zone to edit", list(labels), key="zone_edit_choice")
    zone = labels[choice]

    with st.form("zone_editor"):
        name = st.text_input("Zone name", value=zone.name)
        enabled = st.toggle("Enabled", value=zone.enabled)
        st.markdown("**Vertices** (normalised x, y)")
        frame = pd.DataFrame(zone.polygon_normalized, columns=["x", "y"])
        edited = st.data_editor(
            frame,
            num_rows="dynamic",
            width="stretch",
            hide_index=True,
            key="zone_edit_points",
            column_config={
                "x": st.column_config.NumberColumn("x", min_value=0.0, max_value=1.0, format="%.4f"),
                "y": st.column_config.NumberColumn("y", min_value=0.0, max_value=1.0, format="%.4f"),
            },
        )
        submitted = st.form_submit_button("Save zone", icon=":material/save:")

    if not submitted:
        return

    points: list[tuple[float, float]] = []
    for row in edited.itertuples(index=False):
        x, y = float(row.x), float(row.y)
        if pd.isna(x) or pd.isna(y):
            continue
        points.append((x, y))

    if len(points) < 3:
        st.error("A restricted zone needs at least three vertices.")
        return

    updated = ZoneConfig(
        zone_id=zone.zone_id,
        name=name.strip() or zone.zone_id,
        polygon_normalized=tuple(points),
        enabled=enabled,
    )
    new_zones = [updated if z.zone_id == zone.zone_id else z for z in settings.zones]

    try:
        save_zones(config_path, new_zones)
    except (ConfigError, OSError) as error:
        st.error(f"Could not save the zone: {error}")
        return

    engine = _session_engine()
    if engine is not None:
        try:
            engine.set_zones(build_zones(new_zones))
        except ValueError as error:
            st.error(f"Saved, but the running engine rejected the zone: {error}")
            return
        st.success("Zone saved and applied to the running engine.")
    else:
        st.success("Zone saved. It will apply the next time monitoring starts.")


def render_performance_view() -> None:
    engine = _session_engine()
    st.caption(RUNTIME_MEASURED_NOTE)

    if engine is not None:
        document = engine.metrics.snapshot()
        totals = document["totals"]

        with st.container(horizontal=True):
            st.metric("Frames processed", totals["frames_processed"], border=True)
            st.metric("Throughput", f"{totals['throughput_fps']:.1f} FPS", border=True)
            st.metric("Rolling FPS", f"{totals['rolling_fps']:.1f}", border=True)
            st.metric("Elapsed", f"{totals['elapsed_s']:.0f} s", border=True)

        if document["stages"]:
            st.dataframe(
                [
                    {
                        "Stage": stage,
                        "Samples": summary["count"],
                        "Mean (ms)": summary["mean_ms"],
                        "p50 (ms)": summary["p50_ms"],
                        "p95 (ms)": summary["p95_ms"],
                        "Max (ms)": summary["max_ms"],
                    }
                    for stage, summary in document["stages"].items()
                ],
                width="stretch",
                hide_index=True,
                alt="Per-stage latency summary for this run",
            )
        else:
            st.caption("No frames have been measured yet.")

        with st.container(horizontal=True):
            if st.button("Write performance report", icon=":material/save:"):
                try:
                    path = engine.write_performance_report()
                    st.success(f"Wrote {path}")
                except (OSError, ValueError) as error:
                    st.error(f"Could not write the report: {error}")
            st.download_button(
                "Download report (JSON)",
                data=json.dumps(document, indent=2),
                file_name="performance_report.json",
                mime="application/json",
                icon=":material/download:",
            )
    else:
        st.info(
            "Start monitoring to collect runtime performance measurements.",
            icon=":material/info:",
        )

    st.subheader("Runtime environment")
    st.json(environment_info(), expanded=False)


def render_help_view() -> None:
    engine = _session_engine()
    st.subheader("Compliance status")
    if engine is not None:
        _compliance_badge(engine.compliance_eligible)
        st.markdown(
            f"**Backend in use:** `{engine.backend_id}` on {engine.device_name}  \n"
            f"**Camera:** {engine.settings.video.camera_id}  \n"
            f"**Enter / exit debounce:** {engine.monitor.enter_frames} / "
            f"{engine.monitor.exit_frames} consecutive frames  \n"
            f"**Cooldown:** {engine.settings.alerts.cooldown_seconds:.0f} s per "
            "(zone, track)"
        )
        if engine.notes:
            st.markdown("**Notes from the engine**")
            for note in engine.notes:
                st.markdown(f"- {note}")
    else:
        st.caption("Start monitoring to see which backend and device are in use.")

    st.subheader("CraftifAI PipeGen — outstanding steps")
    st.markdown(
        "The NVIDIA path must be the genuine CraftifAI PipeGen pipeline; a local "
        "YOLO detector is *not* acceptable as proof that it is running. These "
        "steps must be completed on compatible NVIDIA hardware before any "
        "compliance claim can be made:"
    )
    for step in PIPEGEN_CHECKLIST:
        st.markdown(f"- {step}")
    st.markdown(
        "See `evidence/pipegen/README.md` for exactly what evidence must be "
        "captured, and `README.md` for the full acceptance-status table."
    )

    st.subheader("Acceptance status")
    st.caption(
        "AT-01 to AT-12 are covered by the automated test suite and the local "
        "runs on this machine. AT-13 to AT-17 require the PipeGen/TensorRT "
        "pipeline and live NVIDIA hardware, which are not available here, so "
        "they are recorded as BLOCKED rather than claimed."
    )
    st.markdown(
        "- `pytest` (216 tests) covers the pipeline end to end without a camera "
        "or a GPU.\n"
        "- `run_monitor.py` is the native viewer; `app.py` is this dashboard.\n"
        "- Every performance figure shown here is measured at runtime."
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("### VisionSentinel AI")
        st.caption(f"v{APP_VERSION} · restricted-zone monitoring")

        st.text_input(
            "Configuration file",
            value=str(DEFAULT_CONFIG_PATH),
            key="config_path",
            help="settings.yaml is validated before the camera is opened.",
        )
        st.selectbox("Video source", VIDEO_SOURCE_TYPES, key="source_type")
        st.number_input(
            "Camera index", min_value=0, max_value=16, step=1, value=0, key="camera_index"
        )
        st.text_input(
            "Video file (source = file)",
            key="video_file",
            placeholder="clips/front-door.mp4",
        )
        st.text_input(
            "RTSP URL (source = rtsp)",
            key="rtsp_url",
            placeholder="rtsp://user:pass@host:554/stream",
        )
        st.selectbox(
            "Detection backend",
            DETECTION_BACKENDS,
            key="backend",
            help=(
                "auto = Intel GPU, else CPU. nvidia_ort is a developer "
                "convenience and is NOT PipeGen. pipegen reads the real pipeline."
            ),
        )
        allow_download = st.toggle(
            "Allow model download when missing",
            value=True,
            key="allow_download",
            help="Disable to fail instead of fetching the ONNX model.",
        )

        st.divider()
        engine = _session_engine()
        running = engine is not None and engine.is_running

        if st.button(
            "Start monitoring",
            icon=":material/play_arrow:",
            type="primary",
            width="stretch",
            disabled=running,
        ):
            try:
                settings = _effective_settings(
                    st.session_state["config_path"],
                    st.session_state["source_type"],
                    int(st.session_state["camera_index"]),
                    st.session_state["video_file"],
                    st.session_state["rtsp_url"],
                    st.session_state["backend"],
                )
            except ConfigError as error:
                st.session_state["engine_error"] = str(error)
                st.rerun()
            else:
                with st.spinner("Starting the detection engine…"):
                    _start_engine(settings, allow_download)
                st.rerun()

        if st.button(
            "Stop monitoring",
            icon=":material/stop:",
            width="stretch",
            disabled=not running,
        ):
            _stop_engine()
            st.rerun()

        if st.button(
            "Restart",
            icon=":material/restart_alt:",
            width="stretch",
            disabled=engine is None,
        ):
            _dispose_engine()
            st.rerun()

        if engine is not None:
            st.caption(
                f"Backend: `{engine.backend_id}` · device: {engine.device_name}"
            )
            _compliance_badge(engine.compliance_eligible)

        st.divider()
        st.caption("Entry points: `app.py` (this dashboard) · `run_monitor.py` (native window).")


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------
def main() -> None:
    _init_state()
    render_sidebar()

    st.title(":material/security: VisionSentinel AI")
    st.caption(
        "Real-time restricted-zone monitoring · one detection pipeline shared by "
        "the Intel OpenVINO path and the NVIDIA CraftifAI PipeGen path"
    )

    view = st.segmented_control(
        "View",
        options=list(VIEWS),
        default=VIEWS[0],
        required=True,
        key="active_view",
        label_visibility="collapsed",
    )

    if view == "Live monitor":
        render_live_view()
    elif view == "Incidents":
        render_incidents_view()
    elif view == "Zones":
        render_zones_view()
    elif view == "Performance":
        render_performance_view()
    else:
        render_help_view()


main()
