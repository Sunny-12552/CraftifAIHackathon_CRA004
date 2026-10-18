# VisionSentinel AI

**Real-Time Intelligent Safety and Restricted-Zone Monitoring.**

VisionSentinel AI watches a live camera feed, detects people with a real
pretrained model, decides whether a person's foot point has entered a configured
polygonal restricted zone, raises a de-duplicated alert, persists the incident to
SQLite, exports it as JSON, and shows the whole thing on a dashboard. It runs on
an ordinary Intel Windows laptop today; the same detection → zone → alert →
storage pipeline is designed to be fed by the genuine CraftifAI PipeGen /
TensorRT pipeline on compatible NVIDIA hardware.

> **Honest status.** AT-01 to AT-12 (the application criteria) are implemented and
> verified on this machine — see the [acceptance table](#final-acceptance-status).
> AT-13 to AT-17 (the PipeGen / TensorRT / live-NVIDIA criteria) are **BLOCKED**:
> the official CraftifAI PipeGen tooling and a compatible NVIDIA GPU were not
> available. No NVIDIA performance, TensorRT or compliance result is claimed
> anywhere in this repository. See [`evidence/pipegen/README.md`](evidence/pipegen/README.md).

---

## 1. Overview

| | |
|---|---|
| **Language** | Python 3.14 (see [Pinned environment](#pinned-development-environment)) |
| **Video** | OpenCV 5 (`opencv-python`) |
| **Model** | YOLOX (`object_detection_yolox_2022nov.onnx`) |
| **Intel inference** | OpenVINO (GPU, else CPU) |
| **NVIDIA inference** | CraftifAI PipeGen → TensorRT → DeepStream (BLOCKED, by design non-fallback) |
| **Interface** | Streamlit dashboard (`app.py`) + native OpenCV viewer (`run_monitor.py`) |
| **Incident store** | SQLite (WAL) + validated JSON export |
| **Tests** | pytest — 216 tests, no camera or GPU required |

The application is layered so that the same zone/alert/storage code is provably
agnostic to which engine produced the detections:

```
video source ──► detection backend ──► shared Detection events ──► tracking
      │                                           │
 webcam / file / rtsp                 Intel OpenVINO GPU/CPU  or  PipeGen
                                                  │
                                                  ▼
                    zone monitoring ──► alert manager ──► SQLite ──► JSON
                          │                                            │
                          └──────────────► dashboard ◄────────────────┘
```

**The one architecture rule that matters most:** every backend normalises its
output into the same frozen `Detection` / `DetectionEvent` / `ZoneEvent` /
`Incident` schema (`src/vision_sentinel/events.py`). Nothing downstream of the
detector — tracking, zones, alerts, storage, metrics, UI — branches on which
engine is running. On NVIDIA the perception results must come from the real
PipeGen pipeline; a local YOLO detector can never stand in for it.

---

## 2. Features

- **Video input** — built-in/USB webcam, video file, and RTSP, with bounded
  reopen retries and actionable error messages (FR-01, FR-10).
- **Real detection** — pretrained YOLOX via OpenVINO, with bounding boxes,
  labels and confidence scores (FR-02).
- **Restricted zones** — normalised polygons editable in the dashboard, so a
  zone means the same physical area at any resolution (FR-03).
- **Foot-point anchoring + debounce** — occupancy is decided by the bottom-centre
  of the box, with `enter_frames` / `exit_frames` hysteresis, so a swaying torso
  or a one-frame flicker cannot raise an alert (FR-04, FR-05).
- **De-duplicated incidents** — per `(zone, track)` cooldown plus a local IoU
  tracker for engines that do not track (FR-05).
- **SQLite incident history** — unique ids, UTC timestamps, camera/zone/event
  fields, retention option (FR-06).
- **Validated JSON export** — an atomic, schema-checked export envelope (FR-07).
- **Dashboard** — live annotated feed, KPI row, occupancy, active alerts,
  incident log with filters, zone editor, performance report (FR-08).
- **Runtime-measured performance** — real FPS and per-stage latency percentiles
  written to `evidence/`; no baked-in numbers (FR-11).
- **Automated tests** — 216 tests covering geometry, transitions, cooldowns,
  missing detections, persistence, export validity, timestamps, and backend /
  camera / config failures (FR-12).

---

## 3. Prerequisites

- Windows 10/11 (developed and verified on Windows 11), Linux, or macOS.
- Python 3.11+ (verified on 3.14.2).
- A webcam, a video file, or an RTSP stream.
- ~40 MB free for the ONNX model (downloaded on first run).
- No CUDA, TensorRT, Docker, or WSL is required for the Intel/CPU path.

---

## 4. Windows installation

PowerShell, from the project folder:

```powershell
# 1. Create and enter a virtual environment (any location; the verified one is shown)
py -3.14 -m venv C:\Users\samyu\.venvs\vision-sentinel

# 2. Install dependencies with that interpreter
& "C:\Users\samyu\.venvs\vision-sentinel\Scripts\python.exe" -m pip install --upgrade pip
& "C:\Users\samyu\.venvs\vision-sentinel\Scripts\python.exe" -m pip install -r requirements.txt
```

`requirements.txt` deliberately pulls only `numpy`, `opencv-python`, `openvino`,
`PyYAML`, `streamlit` and `pytest`. Nothing in it needs an NVIDIA stack.

---

## 5. Model setup

The YOLOX ONNX model is downloaded automatically on first use into a folder
**outside OneDrive** (so sync never locks it):

```
%LOCALAPPDATA%\VisionSentinelAI\models\object_detection_yolox_2022nov.onnx
```

To supply your own copy instead, either drop the file at that path, or set
`detection.model_path` in `config/settings.yaml`, or set `VS_MODEL_PATH`. To
forbid downloads entirely, pass `--no-download` (the run then fails with a clear
message if the model is missing rather than silently reaching the network).

---

## 6. Running it

### Dashboard (Streamlit)

```powershell
& "C:\Users\samyu\.venvs\vision-sentinel\Scripts\python.exe" -m streamlit run app.py
```

Choose the source in the sidebar and press **Start monitoring**. The page opens
without grabbing the camera; the engine is only built when you ask for it.

### Native viewer (OpenCV)

```powershell
# Default webcam
& "C:\Users\samyu\.venvs\vision-sentinel\Scripts\python.exe" run_monitor.py

# A video file
& "C:\Users\samyu\.venvs\vision-sentinel\Scripts\python.exe" run_monitor.py `
    --source-type file --file clips\front-door.mp4

# Force the CPU backend on a different camera
& "C:\Users\samyu\.venvs\vision-sentinel\Scripts\python.exe" run_monitor.py `
    --backend cpu --camera-index 1

# Headless, no window, stop after 200 frames (CI / measurement)
& "C:\Users\samyu\.venvs\vision-sentinel\Scripts\python.exe" run_monitor.py `
    --headless --max-frames 200
```

Press `q` or `Esc` in the window to stop. Unless `--no-evidence` is given, the
session writes a runtime-measured report and a JSON incident export on exit.

`run_monitor.py` flags: `--config`, `--backend`, `--source-type`, `--file`,
`--camera-index`, `--fps-limit`, `--max-frames`, `--headless`, `--no-overlay`,
`--no-download`, `--no-evidence`.

---

## 7. NVIDIA / PipeGen deployment

The NVIDIA path is implemented but **unverified and BLOCKED** here. On a machine
with the official CraftifAI PipeGen tooling and a compatible NVIDIA GPU:

1. Complete the five PipeGen stages (Requirement → Pipeline IR → Model →
   Compiler → Run) and compile a genuine TensorRT engine.
2. Make the generated pipeline publish detection JSON to a TCP endpoint (see the
   wire protocol in `src/vision_sentinel/detection/pipegen_backend.py`).
3. Run the application against it:

```powershell
$env:VS_DEVICE = "pipegen"
$env:VS_PIPEGEN_ENDPOINT = "127.0.0.1:8554"
& "C:\Users\samyu\.venvs\vision-sentinel\Scripts\python.exe" run_monitor.py
```

The `pipegen` backend is the **only** compliance-eligible path. It contains no
detector and has **no fallback**: if the pipeline is unreachable, or publishes
nothing for 5 seconds, it raises rather than reporting a stale pipeline as an
empty zone. Save the PipeGen export, Pipeline IR graph, TensorRT engine details,
speed report and live-run evidence under `evidence/pipegen/` — the checklist is
in [`evidence/pipegen/README.md`](evidence/pipegen/README.md).

`nvidia_ort` (ONNX Runtime CUDA) exists only as a developer convenience; the code
flags it as **not** PipeGen and **not** compliant.

---

## 8. Configuration reference

All application settings live in [`config/settings.yaml`](config/settings.yaml),
which is validated **before** a camera is opened (FR-10). Relative paths resolve
against the project root.

| Key | Default | Meaning |
|-----|---------|---------|
| `video.source_type` | `webcam` | `webcam` \| `file` \| `rtsp` |
| `video.camera_index` | `0` | Webcam index |
| `video.file_path` | `""` | Required when `source_type: file` |
| `video.rtsp_url` | `""` | Required when `source_type: rtsp` |
| `video.width` / `height` | `640` / `480` | Requested capture size (ignored for files) |
| `video.fps_limit` | `0` | Cap processing rate; `0` = uncapped |
| `video.reopen_attempts` | `3` | Transient read failures tolerated before raising |
| `video.reopen_delay_s` | `1.0` | Delay between reopen attempts |
| `detection.backend` | `auto` | `auto` \| `intel` \| `cpu` \| `nvidia_ort` \| `pipegen` |
| `detection.model_path` | `""` | Empty → the local app models dir |
| `detection.model_url` | OpenCV Zoo URL | Download source for the ONNX model |
| `detection.confidence_threshold` | `0.35` | Detections below this are dropped |
| `detection.nms_threshold` | `0.45` | Non-maximum-suppression IoU |
| `detection.model_size` | `640` | Square network input size |
| `detection.strides` | `[8, 16, 32]` | YOLOX feature-map strides |
| `detection.pipegen_endpoint` | `""` | `host:port` of the PipeGen detection sink |
| `tracking.enabled` | `true` | Local IoU tracker |
| `tracking.iou_threshold` | `0.30` | Tracker match threshold |
| `tracking.max_age` | `15` | Frames a track survives unmatched |
| `tracking.min_hits` | `1` | Hits before a track is confirmed |
| `alerts.enter_frames` | `3` | Consecutive inside frames before ENTER |
| `alerts.exit_frames` | `5` | Consecutive outside frames before EXIT |
| `alerts.cooldown_seconds` | `20.0` | Per `(zone, track)` re-entry suppression |
| `alerts.min_confidence` | `0.35` | Detections below this never inform occupancy |
| `alerts.record_exits` | `true` | Also persist `person_exited_zone` rows |
| `zones[]` | `zone-1` | `id`, `name`, `enabled`, `polygon_normalized` (0.0–1.0) |
| `storage.db_path` | `""` | Empty → `<local app dir>/data/incidents.db` |
| `storage.export_dir` | `""` | Empty → `<local app dir>/data/exports` |
| `storage.save_snapshots` | `false` | Off by default (imagery can contain people) |
| `storage.snapshot_dir` | `""` | Empty → `<local app dir>/data/snapshots` |
| `storage.retention_days` | `0` | `0` = keep everything |
| `metrics.window` | `120` | Samples retained for rolling p50/p95 |
| `metrics.report_path` | `""` | Empty → `evidence/performance_report.json` |

Environment overrides (see [`.env.example`](.env.example)): `VS_DEVICE`,
`VS_MODEL_PATH`, `VS_PIPEGEN_ENDPOINT`, `VS_SOURCE_TYPE`, `VS_CAMERA_INDEX`,
`VS_VIDEO_FILE`, `VS_DB_PATH`, `VS_APP_DIR`. The application reads the process
environment; it does not load a `.env` file itself.

---

## 9. Testing

```powershell
& "C:\Users\samyu\.venvs\vision-sentinel\Scripts\python.exe" -m pytest
```

216 tests. They need **no** camera, GPU, network, or real model: the detector and
video source are injected. `tests/test_acceptance.py` drives the whole pipeline
end to end; `tests/` also covers geometry, the zone state machine, cooldown,
tracking, storage, export validation, configuration, and failure handling.

The tests assert behaviour, not performance — real webcam and hardware runs are
separate, and their results live in [`evidence/`](evidence/README.md).

---

## 10. Pinned development environment

The single verified-working combination (this is what produced every figure in
`evidence/`):

| Package | Version |
|---------|---------|
| Python | 3.14.2 |
| numpy | 2.5.4 |
| opencv-python | 5.0.0.93 |
| openvino | 2026.4.1 |
| streamlit | 1.66.0 |
| pandas | 3.0.6 |
| PyYAML | 6.0.3 |
| pytest | 9.1.1 |
| OS | Windows 11 (build 26300), Intel Iris Xe + CPU |

Interpreter: `C:\Users\samyu\.venvs\vision-sentinel\Scripts\python.exe`.

> **Note on Python version.** The original blueprint suggested Python 3.11 as a
> compatibility-first target. This machine has a 3.14 toolchain with the
> OpenVINO/OpenCV wheels already working, so the project targets 3.14 and pins
> the verified versions above. Nothing in the code requires 3.14 specifically.

### Intentional deviation from the baseline

The preserved Intel baseline decoded YOLOX boxes with integer truncation
(`int()`), which produced occasional 1-pixel off-by-one boxes because of
sub-pixel `exp`/`log` error. `src/vision_sentinel/detection/yolox.py` rounds to
nearest instead. This is documented in the module docstring and covered by
`tests/test_yolox.py`.

---

## 11. Troubleshooting

| Symptom | Cause / fix |
|---------|-------------|
| "Could not open webcam index N" | Windows camera privacy settings, another app holding the camera, or a different index (`--camera-index 1`). |
| "The Intel GPU backend was unavailable, so inference is running on the CPU" | Normal fallback under `backend: auto`. Numbers from that run describe CPU inference. Install/repair the Intel OpenVINO GPU driver for GPU execution. |
| "Could not download the model" | No network / proxy. Download it manually to `%LOCALAPPDATA%\VisionSentinelAI\models\` or set `VS_MODEL_PATH`; use `--no-download` to fail fast. |
| "Could not open the incident database … file is locked" | Another instance is running, or OneDrive is syncing the DB. Runtime data is kept outside OneDrive by default; or set `VS_DB_PATH`. |
| "Could not connect to the PipeGen pipeline" | Expected unless a real generated pipeline is running. See `evidence/pipegen/README.md`. |
| `import cv2` fails | The system Python does not have the dependencies; use the project virtual environment. |
| Tests "fail" with `ModuleNotFoundError: cv2` | Same cause — run pytest with the virtual-environment interpreter. |

---

## 12. Licensing and third-party attribution

Original source is licensed **Apache-2.0** (SPDX headers on every module).

| Component | Licence | Use |
|-----------|---------|-----|
| OpenCV (`opencv-python`) | Apache-2.0 | Video capture and rendering |
| OpenVINO | Apache-2.0 | Intel GPU / CPU inference |
| Streamlit | Apache-2.0 | Dashboard |
| NumPy | BSD-3-Clause | Array maths |
| PyYAML | MIT | Configuration |
| pandas | BSD-3-Clause | Dashboard tables (transitive via Streamlit) |
| pytest | MIT | Tests |
| YOLOX `object_detection_yolox_2022nov.onnx` | Apache-2.0 (OpenCV Zoo) | Detection model |
| `https://ultralytics.com/images/bus.jpg` | see Ultralytics | Local inference sanity-check image used in `evidence/intel/` only; not part of the project |

---

## 13. Known limitations

- **No NVIDIA / PipeGen result.** AT-13 to AT-17 are blocked; the code path is
  implemented but unexercised. No TensorRT or NVIDIA performance is claimed.
- **Single camera, single process** per engine instance.
- **Foot-point heuristic.** Occupancy assumes a standing/visible lower body; a
  person fully behind an occluder may be missed.
- **Local tracker is IoU-only** for the Intel/CPU backends. Fast, crossing tracks
  can swap ids; the PipeGen path uses upstream NvDCF ids instead.
- **RTSP** depends on the OpenCV FFmpeg build and the camera's URL/credentials.
- **No PPE detection.** As required, no PPE violation is claimed.
- **`nvidia_ort`** is a convenience backend only and is never compliance-eligible.

---

## 14. Hackathon compliance checklist

Core application

- [x] Webcam and video-file input work
- [x] Person detection is tested (real model, real photo — see `evidence/intel/`)
- [x] Restricted-zone entry and exit work
- [x] Duplicate alerts are controlled
- [x] SQLite incident history works
- [x] JSON export is validated
- [x] Dashboard errors are handled

Intel environment

- [x] OpenVINO inference runs on the Windows laptop (Intel GPU)
- [x] CPU fallback is tested
- [x] Actual FPS and latency are measured (runtime report in `evidence/`)

PipeGen / NVIDIA

- [ ] Official PipeGen account and version are verified — **BLOCKED**
- [ ] Requirement stage is complete — **BLOCKED**
- [ ] Pipeline IR is generated — **BLOCKED**
- [ ] Model stage is complete — **BLOCKED**
- [ ] TensorRT engine and speed report are genuine — **BLOCKED**
- [ ] Live pipeline runs on supported NVIDIA hardware — **BLOCKED**
- [ ] Actionable output is connected to the pipeline — **BLOCKED**

Submission

- [x] Source code and setup instructions are included
- [ ] PipeGen export folder is included — **BLOCKED**
- [ ] Required NVIDIA evidence is saved — **BLOCKED**
- [x] All claims match verified results (this document claims no NVIDIA result)

---

## 15. Requirement-to-code mapping

| ID | Requirement | Where it lives |
|----|-------------|----------------|
| FR-01 | Live video input | `src/vision_sentinel/video/source.py` |
| FR-02 | Real-time object detection | `src/vision_sentinel/detection/*`, `render.py` |
| FR-03 | Restricted-zone configuration | `config.py` (`zones`), `zones/polygon.py`, `app.py` zone editor |
| FR-04 | Zone-entry detection | `zones/monitor.py`, `zones/state_machine.py` |
| FR-05 | Duplicate-alert prevention | `alerts/cooldown.py`, `alerts/manager.py`, `zones/state_machine.py` |
| FR-06 | Incident management | `storage/sqlite_store.py`, `events.py` |
| FR-07 | JSON export | `storage/json_export.py` |
| FR-08 | Live dashboard | `app.py` (Streamlit), `run_monitor.py` (native) |
| FR-09 | Hardware selection | `detection/__init__.py`, `detection/openvino_backend.py`, `detection/pipegen_backend.py` |
| FR-10 | Error handling | `config.py` validation, `video/source.py`, `detection/base.py` |
| FR-11 | Performance reporting | `metrics/performance.py` |
| FR-12 | Testing | `tests/` (216 tests) |

---

## 16. Final acceptance status

Statuses are `OK` (verified), `NOT OK` (verified failing/incomplete) or `BLOCKED`
(cannot be attempted here). AT-01 to AT-12 are the application criteria; AT-13 to
AT-17 are the mandatory PipeGen / NVIDIA compliance criteria.

| ID | Status | Key files | Test performed | Actual result | Evidence / remaining action |
|----|--------|-----------|----------------|---------------|------------------------------|
| **AT-01** | OK | `app.py`, `run_monitor.py` | Headless `AppTest` run across all five views; `run_monitor.py` smoke run | Both entry points start with no exceptions | `run_monitor.py` output; AppTest smoke run |
| **AT-02** | OK | `video/source.py` | End-to-end run on a video file; `tests/test_video_source.py` | Video-file input works end to end; webcam path implemented (needs a device) | `evidence/intel/end_to_end_report_intel_gpu.json` (source: file) |
| **AT-03** | OK | `detection/*`, `render.py` | Real YOLOX inference on a real photo, Intel GPU and CPU | 4 persons detected with boxes/confidences (0.91, 0.90, 0.89, 0.60) | `evidence/intel/sample_people_intel_openvino.jpg` |
| **AT-04** | OK | `zones/monitor.py`, `alerts/manager.py` | End-to-end run: real frame → zone → alert | 1 confirmed entry → 1 incident, zone marked OCCUPIED | `evidence/intel/end_to_end_annotated_frame.jpg`, `incidents_end_to_end.json` |
| **AT-05** | OK | `zones/state_machine.py` | `tests/test_state_machine.py`, `tests/test_acceptance.py` | Staying inside does not re-alert (debounce + cooldown); `suppressed_duplicates` tracked | `pytest` (216 passed) |
| **AT-06** | OK | `alerts/cooldown.py` | `test_acceptance.py::test_a_brief_exit_then_re_entry_is_suppressed_by_the_cooldown` | Exit then re-entry inside the cooldown is suppressed; after it, a new incident is created | `pytest` |
| **AT-07** | OK | `storage/sqlite_store.py` | End-to-end run; `tests/test_storage.py` | 1 incident persisted (`store.count() == 1`), UTC ids, queryable | `evidence/intel/incidents_end_to_end.json` |
| **AT-08** | OK | `storage/json_export.py` | End-to-end export; `load_export` validation | Valid, atomic, downloadable JSON export (envelope + schema check) | `evidence/intel/incidents_end_to_end.json` |
| **AT-09** | OK | `config.py`, `video/source.py`, `detection/base.py` | `tests/test_config.py`, `tests/test_video_source.py`; failure paths | Invalid config / camera loss / missing model produce actionable messages, no crash | `pytest` |
| **AT-10** | OK | `detection/openvino_backend.py` | Real inference probe on this laptop | `intel_openvino` on **Intel GPU (GPU)**, 61.7 ms mean, 16.2 FPS | `evidence/intel/intel_gpu_probe.json`, `sample_people_intel_openvino.json` |
| **AT-11** | OK | `detection/openvino_backend.py` | Same probe with `backend: cpu` | `cpu_openvino` on **CPU (CPU)**, 164.8 ms mean, 6.1 FPS; same 4 persons | `evidence/intel/sample_people_cpu_openvino.json` |
| **AT-12** | OK | `metrics/performance.py` | End-to-end run writes a report | Runtime-measured: 20 frames, inference p50 61.4 ms, throughput 4.67 FPS, environment captured | `evidence/intel/end_to_end_report_intel_gpu.json` |
| **AT-13** | BLOCKED | `detection/pipegen_backend.py` | — | Not attempted: no official PipeGen access | Complete the five PipeGen stages on NVIDIA hardware |
| **AT-14** | BLOCKED | `detection/pipegen_backend.py` | — | No genuine TensorRT engine or PipeGen speed report exists | Compile the real engine; save its details and the speed report |
| **AT-15** | BLOCKED | `detection/pipegen_backend.py` | — | No live run: no compatible NVIDIA GPU | Run the generated pipeline live on supported hardware |
| **AT-16** | BLOCKED | `detection/pipegen_backend.py`, `engine.py` | — | Consumer implemented (TCP JSON → shared schema) but never fed by a real pipeline | Connect the live pipeline to the endpoint and capture the events |
| **AT-17** | BLOCKED | whole repo, `evidence/` | — | Intel source, tests and evidence are prepared; the required PipeGen exports/evidence cannot be produced here | Save PipeGen export, IR graph, engine details, speed report, logs into `evidence/pipegen/` |

> AT-13 to AT-17 are **not** marked OK. Doing so without the real pipeline, a
> genuine TensorRT engine, a live NVIDIA run and saved evidence would be a false
> claim, which this project deliberately refuses to make.

---

## 17. Project layout

```
Sence AI/
├── app.py                     # Streamlit dashboard (entry point)
├── run_monitor.py             # native OpenCV viewer (entry point)
├── requirements.txt
├── README.md
├── .env.example
├── .gitignore
├── pytest.ini
├── config/
│   └── settings.yaml          # validated configuration
├── src/vision_sentinel/
│   ├── config.py  events.py  engine.py  paths.py  render.py
│   ├── video/       # webcam, file, rtsp
│   ├── detection/   # base, openvino (intel/cpu), nvidia_ort, pipegen, yolox
│   ├── tracking/    # IoU tracker
│   ├── zones/       # polygons, state machine, monitor
│   ├── alerts/      # cooldown, manager
│   ├── storage/     # sqlite store, json export
│   └── metrics/     # fps, latency, performance report
├── tests/                     # 216 automated tests
└── evidence/                  # real, reproducible runtime evidence
    ├── README.md
    ├── intel/                 # measured results + annotated frames
    └── pipegen/               # BLOCKED — checklist only
```
