#!/usr/bin/env python3
"""
Pan-Tilt HAT web controller with MJPEG stream, face tracking, and YOLOv8 object detection.
Runs on a Raspberry Pi 4 with a Pimoroni Pan-Tilt HAT + Pi Camera Module v2.

Object detection backend:
  - Primary: Intel NCS2 (Myriad X) via the ncs_infer.py microservice over a Unix socket.
    OpenVINO is built for Python 3.10; this app runs on the system Python (3.13 on Trixie,
    where picamera2/libcamera live), so NCS2 inference runs in a separate process and we
    talk to it over a Unix socket.
  - Fallback: onnxruntime CPU (in-process), used automatically if the NCS2 service/stick
    is unavailable. Self-heals back to NCS2 when the service returns.

Detection runs in a dedicated worker thread so inference latency never stutters the stream.

Config via environment variables:
  YOLO_MODEL  path to yolov8n.onnx        (default: ./models/yolov8n.onnx)
  NCS_SOCK    Unix socket to the service  (default: /tmp/ncs_infer.sock)
"""

import os
import glob
import threading
import time
import socket
import struct
import pickle
import numpy as np
import cv2
import pantilthat
import onnxruntime as ort
from flask import Flask, Response, render_template_string, request, jsonify
from picamera2 import Picamera2

app = Flask(__name__)

APP_DIR = os.path.dirname(os.path.abspath(__file__))

# --- State ---
pan_angle = 0
tilt_angle = 0
tracking_enabled = False
track_mode = "face"       # "face" or "object"
track_target_class = 0    # COCO class index to track (0=person)

stream_lock = threading.Lock()
latest_frame = None       # encoded JPEG bytes for the MJPEG stream

raw_lock = threading.Lock()
latest_raw = None         # most recent flipped BGR frame, for the detector to consume

det_lock = threading.Lock()
latest_dets = []          # list of (x, y, w, h, label, color) to overlay on the stream

last_backend = "-"        # "NCS" / "CPU" / "-" : which engine served the last object detection

# Pan-tilt limits (degrees)
PAN_MIN, PAN_MAX = -90, 90
TILT_MIN, TILT_MAX = -90, 90

# Tracking PD gains
KP = 0.08
KD = 0.006
DEADZONE = 0.05
last_pan_err = 0
last_tilt_err = 0

# COCO class names (80 classes)
COCO_CLASSES = [
    "person","bicycle","car","motorbike","aeroplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
    "sofa","pottedplant","bed","diningtable","toilet","tvmonitor","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator",
    "book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
]

# --- Camera ---
picam = Picamera2()
picam.configure(picam.create_video_configuration(main={"size": (640, 480)}))
picam.start()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# --- NCS2 inference client (talks to ncs_infer.py over a Unix socket) ---
NCS_SOCK = os.environ.get("NCS_SOCK", "/tmp/ncs_infer.sock")
USE_NCS = True            # master toggle for the NCS2 path
ncs_conn = None
ncs_lock = threading.Lock()


def _ncs_connect():
    """(Re)establish the Unix-socket connection to the NCS2 service. Returns True on success."""
    global ncs_conn
    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(3.0)
        s.connect(NCS_SOCK)
        ncs_conn = s
        return True
    except Exception:
        ncs_conn = None
        return False


def _recvall(conn, n):
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def ncs_detect(frame, model_name, conf_thresh=0.45, target_class=None):
    """Run detection on the NCS2 service. Returns a detection list, or None to signal fallback."""
    global ncs_conn
    with ncs_lock:
        if ncs_conn is None and not _ncs_connect():
            return None
        try:
            req = {"frame": frame, "conf": conf_thresh, "target_class": target_class, "model": model_name}
            data = pickle.dumps(req, protocol=pickle.HIGHEST_PROTOCOL)
            ncs_conn.sendall(struct.pack(">I", len(data)) + data)
            raw = _recvall(ncs_conn, 4)
            if not raw:
                raise IOError("no length header")
            n = struct.unpack(">I", raw)[0]
            payload = _recvall(ncs_conn, n)
            if payload is None:
                raise IOError("short response")
            resp = pickle.loads(payload)
            if not resp.get("ok"):
                return None  # service reported an inference error -> fall back
            return resp["dets"]
        except Exception:
            try:
                ncs_conn.close()
            except Exception:
                pass
            ncs_conn = None
            return None


# --- YOLO models (CPU fallback path; NCS2 path selects the same model by name) ---
MODELS_DIR = os.environ.get("MODELS_DIR", os.path.join(APP_DIR, "models"))
AVAILABLE_MODELS = [os.path.splitext(os.path.basename(p))[0]
                    for p in sorted(glob.glob(os.path.join(MODELS_DIR, "*.onnx")))]
current_model = os.environ.get(
    "YOLO_MODEL_NAME",
    "yolov8s" if "yolov8s" in AVAILABLE_MODELS else (AVAILABLE_MODELS[0] if AVAILABLE_MODELS else "yolov8n"))

cpu_sessions = {}         # name -> (session, input_name), lazily loaded
cpu_sess_lock = threading.Lock()


def _cpu_session(name):
    with cpu_sess_lock:
        if name not in cpu_sessions:
            path = os.path.join(MODELS_DIR, name + ".onnx")
            opts = ort.SessionOptions()
            opts.inter_op_num_threads = 2
            opts.intra_op_num_threads = 4
            sess = ort.InferenceSession(path, sess_options=opts, providers=["CPUExecutionProvider"])
            cpu_sessions[name] = (sess, sess.get_inputs()[0].name)
            print(f"CPU session loaded for {name}")
        return cpu_sessions[name]


def load_yolo():
    """Pre-load the default model's CPU session so the first fallback isn't slow."""
    try:
        _cpu_session(current_model)
    except Exception as e:
        print(f"YOLO CPU preload failed for {current_model}: {e}")


def yolo_detect_cpu(frame, model_name, conf_thresh=0.45, target_class=None):
    """CPU onnxruntime fallback. Returns list of (x, y, w, h, class_id, conf)."""
    try:
        sess, input_name = _cpu_session(model_name)
    except Exception as e:
        print(f"CPU session load failed for {model_name}: {e}")
        return []
    fh, fw = frame.shape[:2]
    size = 640
    img = cv2.resize(frame, (size, size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))[np.newaxis]

    outputs = sess.run(None, {input_name: img})[0][0]  # (84, N)
    outputs = outputs.T

    boxes = []
    for det in outputs:
        cx, cy, w, h = det[:4]
        scores = det[4:]
        cls = int(np.argmax(scores))
        conf = float(scores[cls])
        if conf < conf_thresh:
            continue
        if target_class is not None and cls != target_class:
            continue
        x1 = int((cx - w / 2) * fw / size)
        y1 = int((cy - h / 2) * fh / size)
        bw = int(w * fw / size)
        bh = int(h * fh / size)
        boxes.append((x1, y1, bw, bh, cls, conf))
    return boxes


def detect_objects(frame, target_class):
    """Try NCS2 first, fall back to CPU. Updates last_backend. Returns detection list."""
    global last_backend
    model_name = current_model
    if USE_NCS:
        dets = ncs_detect(frame, model_name, 0.45, target_class)
        if dets is not None:
            last_backend = "NCS"
            return dets
    dets = yolo_detect_cpu(frame, model_name, target_class=target_class)
    last_backend = "CPU"
    return dets


# Face mode: NCS2 SSD detector (fast, accurate) with OpenCV Haar cascade as the CPU fallback.
FACE_MODEL = "face-detection-retail-0004"
FACE_CONF = 0.6


def detect_faces(frame):
    """Try the NCS2 face detector, fall back to the Haar cascade on CPU.
    Returns list of (x, y, w, h). Updates last_backend."""
    global last_backend
    if USE_NCS:
        dets = ncs_detect(frame, FACE_MODEL, conf_thresh=FACE_CONF, target_class=None)
        if dets is not None:
            last_backend = "NCS"
            return [(x, y, w, h) for (x, y, w, h, _cls, _conf) in dets]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
    last_backend = "CPU"
    return [tuple(int(v) for v in f) for f in faces]


def clamp(val, lo, hi):
    return max(lo, min(hi, val))


def set_pan_tilt(p, t):
    global pan_angle, tilt_angle
    pan_angle = clamp(p, PAN_MIN, PAN_MAX)
    tilt_angle = clamp(t, TILT_MIN, TILT_MAX)
    pantilthat.pan(pan_angle)
    pantilthat.tilt(tilt_angle)


def track_target(frame, cx, cy):
    """Drive servos toward target centre point (cx, cy) in frame."""
    global last_pan_err, last_tilt_err
    fh, fw = frame.shape[:2]
    pan_err = (fw // 2 - cx) / (fw / 2)
    tilt_err = (cy - fh // 2) / (fh / 2)
    if abs(pan_err) > DEADZONE or abs(tilt_err) > DEADZONE:
        new_pan = pan_angle + KP * pan_err * 90 + KD * (pan_err - last_pan_err) * 90
        new_tilt = tilt_angle + KP * tilt_err * 90 + KD * (tilt_err - last_tilt_err) * 90
        set_pan_tilt(new_pan, new_tilt)
    last_pan_err = pan_err
    last_tilt_err = tilt_err


def capture_loop():
    """Capture + flip + overlay detections + encode. Stays fast; never blocks on inference."""
    global latest_frame, latest_raw
    while True:
        try:
            frame = picam.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame = cv2.flip(frame, -1)  # camera mounted upside-down

            with raw_lock:
                latest_raw = frame  # detector copies before mutating

            with det_lock:
                dets = list(latest_dets)
            for (x, y, w, h, label, color) in dets:
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.circle(frame, (x + w // 2, y + h // 2), 4, color, -1)
                if label:
                    cv2.putText(frame, label, (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            if tracking_enabled and track_mode == "object":
                backend = f" {last_backend}/{current_model}"
            elif tracking_enabled and track_mode == "face":
                backend = f" {last_backend}"
            else:
                backend = ""
            status = (f"{'TRACKING' if tracking_enabled else 'MANUAL'} [{track_mode}{backend}]  "
                      f"pan:{pan_angle:.0f} tilt:{tilt_angle:.0f}")
            cv2.putText(frame, status, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

            _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            with stream_lock:
                latest_frame = jpeg.tobytes()

            time.sleep(0.02)
        except Exception as e:
            print(f"capture_loop error: {e}", flush=True)
            time.sleep(1)


def detect_loop():
    """Detection + servo control, decoupled from the capture/stream loop."""
    global latest_dets
    while True:
        try:
            if not tracking_enabled:
                with det_lock:
                    latest_dets = []
                time.sleep(0.1)
                continue

            with raw_lock:
                frame = None if latest_raw is None else latest_raw.copy()
            if frame is None:
                time.sleep(0.05)
                continue

            if track_mode == "face":
                faces = detect_faces(frame)
                if faces:
                    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                    with det_lock:
                        latest_dets = [(int(x), int(y), int(w), int(h), "", (0, 255, 0))]
                    track_target(frame, x + w // 2, y + h // 2)
                else:
                    with det_lock:
                        latest_dets = []
                time.sleep(0.005)

            elif track_mode == "object":
                dets = detect_objects(frame, track_target_class)
                if dets:
                    x, y, w, h, cls, conf = max(dets, key=lambda d: d[2] * d[3])
                    label = f"{COCO_CLASSES[cls]} {conf:.2f}"
                    with det_lock:
                        latest_dets = [(x, y, w, h, label, (0, 200, 255))]
                    track_target(frame, x + w // 2, y + h // 2)
                else:
                    with det_lock:
                        latest_dets = []
                # inference latency is the natural throttle; no extra sleep
        except Exception as e:
            print(f"detect_loop error: {e}", flush=True)
            time.sleep(0.5)


def gen_frames():
    while True:
        with stream_lock:
            frame = latest_frame
        if frame:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        time.sleep(0.033)


# --- Routes ---

HTML = """<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pan-Tilt Controller</title>
<style>
  body { background:#111; color:#eee; font-family:monospace; text-align:center; margin:0; padding:16px; }
  img  { width:100%; max-width:640px; border:2px solid #333; border-radius:4px; }
  .controls { display:flex; flex-wrap:wrap; justify-content:center; gap:10px; margin:12px 0; }
  button { padding:11px 18px; font-size:15px; border:none; border-radius:4px; cursor:pointer; background:#333; color:#eee; }
  button:active { background:#555; }
  .active { background:#2a7 !important; }
  .row { display:flex; justify-content:center; gap:8px; margin:4px 0; }
  h2 { margin:6px 0; font-size:0.9rem; color:#888; }
  select { background:#333; color:#eee; border:1px solid #555; padding:8px; border-radius:4px; font-size:14px; }
</style>
</head>
<body>
<h1 style="font-size:1.1rem; margin-bottom:6px;">Pan-Tilt Camera</h1>
<img src="/stream" />

<div class="controls">
  <div>
    <h2>Move</h2>
    <div class="row"><button onclick="move(0,10)">&#8679; Up</button></div>
    <div class="row">
      <button onclick="move(-10,0)">&#8678; Left</button>
      <button onclick="centre()">Centre</button>
      <button onclick="move(10,0)">Right &#8680;</button>
    </div>
    <div class="row"><button onclick="move(0,-10)">Down &#8681;</button></div>
  </div>
</div>

<div class="controls">
  <button id="track-btn" onclick="toggleTrack()">&#128247; Track: OFF</button>
  <select id="mode-select" onchange="setMode(this.value)">
    <option value="face">Face</option>
    <option value="object">Object (YOLO)</option>
  </select>
  <select id="class-select" onchange="setClass(parseInt(this.value))" style="display:none">
    <option value="0">Person</option>
    <option value="14">Bird</option>
    <option value="15">Cat</option>
    <option value="16">Dog</option>
    <option value="2">Car</option>
    <option value="39">Bottle</option>
  </select>
  <select id="model-select" onchange="setModel(this.value)" style="display:none">
    <option value="yolov8n">YOLOv8n (fast)</option>
    <option value="yolov8s">YOLOv8s (accurate)</option>
  </select>
</div>

<p id="status" style="color:#888; font-size:0.82rem;"></p>

<script>
async function move(dp, dt) {
  const r = await fetch('/move', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({pan_delta: dp, tilt_delta: dt})});
  const d = await r.json();
  document.getElementById('status').textContent = `pan: ${d.pan}°  tilt: ${d.tilt}°`;
}
async function centre() {
  const r = await fetch('/centre', {method:'POST'});
  const d = await r.json();
  document.getElementById('status').textContent = `pan: ${d.pan}°  tilt: ${d.tilt}°`;
}
async function toggleTrack() {
  const r = await fetch('/track/toggle', {method:'POST'});
  const d = await r.json();
  const btn = document.getElementById('track-btn');
  btn.textContent = 'Track: ' + (d.tracking ? 'ON' : 'OFF');
  btn.className = d.tracking ? 'active' : '';
}
async function setMode(mode) {
  await fetch('/track/mode', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({mode})});
  const show = mode === 'object' ? 'inline-block' : 'none';
  document.getElementById('class-select').style.display = show;
  document.getElementById('model-select').style.display = show;
}
async function setClass(cls) {
  await fetch('/track/class', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({class_id: cls})});
}
async function setModel(m) {
  await fetch('/track/model', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({model: m})});
}
// Sync the model dropdown to the server's current model on load
fetch('/status').then(r => r.json()).then(d => {
  if (d.model) document.getElementById('model-select').value = d.model;
}).catch(() => {});
</script>
</body>
</html>"""


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/stream")
def stream():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/move", methods=["POST"])
def move():
    data = request.get_json()
    # Camera is mounted upside-down (frame flipped 180deg), so BOTH axes are inverted
    # relative to the displayed image - mirror pan and tilt to match the user's view.
    set_pan_tilt(pan_angle - data.get("pan_delta", 0), tilt_angle - data.get("tilt_delta", 0))
    return jsonify(pan=round(pan_angle), tilt=round(tilt_angle))


@app.route("/centre", methods=["POST"])
def centre():
    set_pan_tilt(0, 0)
    return jsonify(pan=0, tilt=0)


@app.route("/track/toggle", methods=["POST"])
def track_toggle():
    global tracking_enabled
    tracking_enabled = not tracking_enabled
    return jsonify(tracking=tracking_enabled)


@app.route("/track/mode", methods=["POST"])
def track_mode_set():
    global track_mode
    track_mode = request.get_json().get("mode", "face")
    return jsonify(mode=track_mode)


@app.route("/track/class", methods=["POST"])
def track_class_set():
    global track_target_class
    track_target_class = int(request.get_json().get("class_id", 0))
    return jsonify(class_id=track_target_class)


@app.route("/track/model", methods=["POST"])
def track_model_set():
    global current_model
    m = request.get_json().get("model", current_model)
    if (not AVAILABLE_MODELS) or (m in AVAILABLE_MODELS):
        current_model = m
    return jsonify(model=current_model)


@app.route("/status")
def status():
    return jsonify(pan=round(pan_angle), tilt=round(tilt_angle),
                   tracking=tracking_enabled, mode=track_mode,
                   backend=last_backend, model=current_model,
                   available_models=AVAILABLE_MODELS)


if __name__ == "__main__":
    load_yolo()
    threading.Thread(target=capture_loop, daemon=True).start()
    threading.Thread(target=detect_loop, daemon=True).start()
    time.sleep(1)
    set_pan_tilt(0, 0)
    print("Pan-Tilt app running at http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, threaded=True)
