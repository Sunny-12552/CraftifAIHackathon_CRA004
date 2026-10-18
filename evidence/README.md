# Evidence

Everything in this directory was produced by actually running VisionSentinel AI
on the development laptop (Intel Iris Xe, Windows 11, Python 3.14.2,
OpenVINO 2026.4.1). Nothing here is a mock-up, a design target, or a hand-written
number. Each file is reproducible with the commands in the section below it.

The NVIDIA / CraftifAI PipeGen evidence required by AT-13 to AT-17 does **not**
exist here, because the tooling and compatible hardware were not available. See
`pipegen/README.md`. That absence is deliberate: an empty folder is honest, a
fabricated one is not.

## `intel/`

| File | What it shows | Reproduce with |
|------|----------------|----------------|
| `intel_gpu_probe.json` | The Intel OpenVINO GPU backend initialised and ran 20 timed inferences on a synthetic frame (no detections expected). Records the device, model path and measured latency. | the probe script in "Reproducing the evidence" |
| `sample_people_intel_openvino.jpg` | A real photo with four people, each detected by the pretrained model with a bounding box and confidence, annotated by `render_frame`. | `sample_probe.py` |
| `sample_people_intel_openvino.json` | The raw detections and measured latency behind that image. | `sample_probe.py` |
| `sample_people_cpu_openvino.jpg` / `.json` | The same image through the CPU fallback backend (AT-11). | `sample_probe.py` |
| `end_to_end_annotated_frame.jpg` | The engine's own output frame: the alert banner, the occupied restricted zone, the intruder boxed in red, other people in green, and the yellow foot points that anchor the zone test. | `end_to_end.py` |
| `incidents_end_to_end.json` | The valid JSON export of the single confirmed incident from that run. | `end_to_end.py` |
| `end_to_end_report_intel_gpu.json` | The runtime-measured performance report: per-stage latency percentiles, throughput, and the machine's environment. | `end_to_end.py` |

### Headline numbers (measured, this machine)

| Backend | Device | Mean inference | Throughput | People found |
|---------|--------|----------------|------------|--------------|
| `intel_openvino` | Intel GPU (GPU) | 61.7 ms | 16.2 FPS | 4 |
| `cpu_openvino` | CPU (CPU) | 164.8 ms | 6.1 FPS | 4 |

End-to-end run (real video → Intel GPU → zones → SQLite → export): 20 frames,
1 confirmed entry, 1 incident, inference p50 61.4 ms.

### Attribution

`sample_people_*.jpg` are annotated copies of `https://ultralytics.com/images/bus.jpg`,
used only as a locally-run inference sanity check. The underlying photograph is
not part of this project and is included here solely as test evidence.

## Reproducing the evidence

The scripts used to generate `intel/` were ad-hoc probes; the equivalent
commands in the shipped application are:

```powershell
# Live dashboard (Intel GPU, else CPU)
streamlit run app.py

# Native viewer, real inference on a video file
python run_monitor.py --source-type file --file clips/your-clip.mp4

# Native viewer, headless, CPU fallback
python run_monitor.py --headless --source-type file --file clips/your-clip.mp4 --backend cpu
```

Each run writes a fresh runtime-measured report to `evidence/performance_report.json`
and a timestamped incident export to `<local app dir>/data/exports/` (unless
`--no-evidence` is passed).

## `pipegen/`

Reserved for the NVIDIA evidence. Currently contains only the checklist of what
must be captured. See `pipegen/README.md`.
