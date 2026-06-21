# Pan-Tilt Tracking Camera on Raspberry Pi 4 + Intel NCS2

A web-controlled pan-tilt camera for the Raspberry Pi 4 that streams live video and tracks
faces or COCO objects, with **YOLOv8 object detection accelerated on an Intel Neural Compute
Stick 2 (NCS2 / Myriad X)** and an automatic **CPU fallback**.

The harder half of this repo is the **build recipe**: getting the (discontinued) NCS2 working
on a modern 64-bit Pi running Debian Trixie with GCC 14, which needs OpenVINO 2022.3.2 compiled
from source - see [`docs/NCS2-OPENVINO-BUILD-GUIDE.md`](docs/NCS2-OPENVINO-BUILD-GUIDE.md).

## Features
- Live MJPEG video stream in the browser
- Manual pan/tilt control + centre
- **Face tracking** - NCS2 SSD detector (`face-detection-retail-0004`, ~45 FPS) when the stick
  is present, automatic fallback to an OpenCV Haar cascade on CPU when it isn't
- **Object tracking** (YOLOv8, 80 COCO classes; selectable target class)
- **Runtime model switching** between YOLOv8n (fast) and YOLOv8s (accurate) - both pre-compiled
  on the stick at startup, so switching is instant (no recompile). Selectable from the UI.
- PD servo controller for smooth tracking
- Detection runs in a worker thread, decoupled from the stream, so inference never stutters video
- NCS2 acceleration with seamless CPU fallback and self-heal; the active backend + model are
  shown in the overlay and at `/status`

## Performance (640x640, Pi 4): {nano, small} x {NCS2, CPU}
| Model | NCS2 | CPU (4 threads) | NCS2 speedup |
|---|---|---|---|
| YOLOv8n (~3.2M params) | 153 ms · **6.5 FPS** | 536 ms · 1.9 FPS | 3.4x |
| YOLOv8s (~11M params) | 307 ms · **3.3 FPS** | 1416 ms · 0.7 FPS | 4.6x |
| face-detection-retail-0004 (SSD, face mode) | 22 ms · **45 FPS** | Haar cascade (fallback) | dedicated detector |

The headline result: **YOLOv8s on the NCS2 (3.3 FPS) is faster than YOLOv8n on the CPU
(1.9 FPS)** while being a more accurate model. The stick doesn't just speed things up - it
unlocks a model class the Pi's CPU can't run in real time, and frees the CPU for
capture/streaming/servo control.

## Hardware
- Raspberry Pi 4 (4 GB+), 64-bit OS (developed on Debian Trixie)
- Pimoroni Pan-Tilt HAT (I2C, `pantilthat` library)
- Pi Camera Module v2 (IMX219)
- Intel Neural Compute Stick 2 (optional - CPU fallback works without it)

## Architecture
OpenVINO's Python bindings are built for Python 3.10; picamera2/libcamera on Trixie are tied
to the system Python 3.13. The two can't share an interpreter, so inference runs as a separate
process and the app talks to it over a Unix socket:

```
  ┌─────────────────────────────┐         ┌──────────────────────────────┐
  │ app.py  (system Python 3.13)│  Unix   │ ncs_infer.py (Python 3.10)   │
  │  picamera2, Flask, servos   │ socket  │  OpenVINO + NCS2 (Myriad X)  │
  │  capture + detect threads   │ ──────► │  YOLOv8n inference           │
  │  CPU fallback (onnxruntime) │ ◄────── │  returns detections          │
  └─────────────────────────────┘         └──────────────────────────────┘
        if the service/stick is down, app.py uses its in-process CPU path
```

## Setup
1. **Build OpenVINO 2022.3.2 + NCS2 support** (only needed for the NCS2 path):
   follow [`docs/NCS2-OPENVINO-BUILD-GUIDE.md`](docs/NCS2-OPENVINO-BUILD-GUIDE.md). This
   creates `~/ov-venv` (Python 3.10 + OpenVINO) and `~/openvino-2022/install/setupvars.sh`.
   Then add OpenCV to that venv: `~/ov-venv/bin/pip install "opencv-python-headless<4.13"`.

2. **App environment** (system Python, with camera + HAT libraries):
   ```bash
   python3 -m venv --system-site-packages ~/pantilt-env   # inherits system picamera2/libcamera
   ~/pantilt-env/bin/pip install flask opencv-python-headless onnxruntime pantilthat
   ```

3. **Model**: see [`models/README.md`](models/README.md) (export `yolov8n.onnx`).

## Run
```bash
./start.sh
# open http://<pi-ip>:5000
```
`start.sh` launches both processes (paths are overridable via `OV_VENV`, `APP_VENV`,
`OV_SETUPVARS`, `APP_DIR` env vars). To run **CPU-only** without the NCS2, just start the app:
```bash
~/pantilt-env/bin/python3 app.py
```

## API
| Route | Method | Purpose |
|---|---|---|
| `/` | GET | Web UI |
| `/stream` | GET | MJPEG stream |
| `/move` | POST | `{pan_delta, tilt_delta}` |
| `/centre` | POST | Recentre servos |
| `/track/toggle` | POST | Tracking on/off |
| `/track/mode` | POST | `{mode: "face"\|"object"}` |
| `/track/class` | POST | `{class_id: <COCO index>}` |
| `/track/model` | POST | `{model: "yolov8n"\|"yolov8s"}` - switch model at runtime |
| `/status` | GET | pan/tilt/tracking/mode/backend/model/available_models |

## Notes
- Camera is mounted upside-down in this build (`cv2.flip(frame, -1)`); remove that line if yours isn't.
- If the NCS2 drops off USB mid-run, the app falls back to CPU automatically; re-run `start.sh`
  to bring the stick back (the service recompiles the model on restart).

## License
MIT - see [LICENSE](LICENSE).
