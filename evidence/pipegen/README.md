# CraftifAI PipeGen / NVIDIA evidence — BLOCKED

**Status: BLOCKED. This folder intentionally contains no pipeline artefacts.**

AT-13 through AT-17 require the genuine CraftifAI PipeGen workflow (Requirement
→ Pipeline IR → Model → Compiler → Run), a TensorRT engine compiled for the
target GPU, the real PipeGen speed report, and a live run on compatible NVIDIA
hardware. None of those were available while building this project:

* no access to the official CraftifAI PipeGen tooling, and
* no compatible NVIDIA GPU machine (the development laptop is Intel Iris Xe).

The project rules are explicit: a local YOLO detector may **not** be presented as
proof that PipeGen is running, and AT-13 to AT-17 may **not** be marked complete
without genuine evidence. So this folder stays empty of results by design. The
application code path for NVIDIA is implemented and documented below; it simply
cannot be exercised here.

Everything the Intel path can prove is captured one level up in `evidence/intel/`.

---

## The eight steps that must be completed on NVIDIA hardware

These are the exact steps the `pipegen` backend waits for
(`src/vision_sentinel/detection/pipegen_backend.py`):

1. Install the official CraftifAI PipeGen version and its confirmed
   prerequisites on a compatible NVIDIA Linux machine.
2. Sign in to the official CraftifAI tool.
3. Complete the **Requirement** stage from the project specification.
4. Generate and export the **Pipeline IR** graph.
5. Configure the model in the **Model** stage.
6. Compile through PipeGen to a genuine **TensorRT** engine for the target GPU
   (TensorRT engines are specific to the GPU/runtime they were built for — do
   not assume an engine moves between GPUs).
7. **Run** the generated pipeline live and publish detection JSON to the
   configured `detection.pipegen_endpoint`.
8. Save the PipeGen export, Pipeline IR graph, speed report, TensorRT engine
   details and runtime log **into this folder**.

Do not guess PipeGen's installation requirements; use the version and
environment confirmed by CraftifAI.

---

## What this folder must contain when it is unblocked

- [ ] `pipegen_export/` — the genuine PipeGen export folder.
- [ ] `pipeline_ir.*` — the exported Pipeline IR graph.
- [ ] `tensorrt_engine.txt` — the engine details: model, precision, target GPU,
      build log summary.
- [ ] `speed_report.*` — the actual PipeGen speed report.
- [ ] `runtime.log` — the live-run log showing all five stages completed.
- [ ] `live_demo.mp4` — a screen capture of the pipeline running (DoS/DON'T
      fabricate; capture the real window).
- [ ] `live_events.jsonl` — a capture of the detection JSON the pipeline
      published on the wire.
- [ ] `vision_sentinel_pipegen_report.json` — the report this application writes
      with `VS_DEVICE=pipegen`, showing `compliance_eligible: true`,
      `backend: pipegen` and a measured FPS/latency for the live run.
- [ ] `incidents_pipegen.json` — an incident export from that live run.

---

## How the NVIDIA path is wired

The shared pipeline never runs a second detector on the NVIDIA machine. The
generated PipeGen pipeline owns the camera and publishes detections; this
application consumes them:

```
[ PipeGen / DeepStream / TensorRT pipeline ]        [ VisionSentinel AI ]
        camera ─► inference ─► tracker   ──TCP──►   PipeGenBackend
                                                        │
                                            shared Detection events
                                                        │
                                      zones ─► alerts ─► SQLite ─► JSON / dashboard
```

The transport is newline-delimited JSON over TCP. The pipeline is expected to
publish one object per processed batch:

```json
{"frame_id": 4171,
 "timestamp_utc": "2026-10-17T09:31:04.221Z",
 "source": "pipegen",
 "objects": [{"label": "person", "confidence": 0.91,
              "bbox_xywh": [120, 80, 140, 310], "track_id": 7}]}
```

`track_id` carries the ids assigned upstream by the DeepStream tracker (for
example NvDCF), so no local tracking is needed in this mode.

This backend has **no fallback**. If the socket is unreachable, or if no fresh
batch has arrived within `DEFAULT_STALE_AFTER_S` (5 s), it raises
`PipeGenUnavailableError`. Zone evaluation pauses rather than reporting a stale
pipeline as an empty zone — "no data" must never become "the zone is clear".

Run it once the pipeline is live:

```powershell
# on the NVIDIA machine (PowerShell Core or with the env exported)
$env:VS_DEVICE = "pipegen"
$env:VS_PIPEGEN_ENDPOINT = "127.0.0.1:8554"
python run_monitor.py            # or: streamlit run app.py
```

Only when this runs against the real pipeline, and the artefacts above are
saved, may AT-13 to AT-17 move from BLOCKED to OK.

---

## Why this is not a shortcut

The hackathon rule is that on NVIDIA the perception results must come from the
real PipeGen-generated pipeline. A backend that silently fell back to a local
OpenVINO/OpenCV detector whenever the pipeline was unreachable would let a
non-compliant run be mistaken for a compliant one. That is why
`PipeGenBackend` contains no detector, why it is never selected by
`backend: auto`, and why it must be requested by name.
