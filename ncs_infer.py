#!/usr/bin/env python3
"""
NCS2 (Intel Movidius Myriad X) inference microservice.

Runs in the OpenVINO Python 3.10 venv with setupvars.sh sourced, because the OpenVINO
2022.3.2 Python bindings are built for Python 3.10 and CANNOT be imported in the pan-tilt
app's system Python (3.13 on Trixie, where picamera2/libcamera are 3.13-locked C extensions).

Serves two kinds of model, both pre-compiled on the stick at startup (the Myriad X holds
several small models simultaneously), so the app switches at runtime with no recompile stall:
  - "yolo": YOLOv8 ONNX detectors in MODELS_DIR (640x640, RGB/255, transposed grid output)
  - "ssd" : OpenVINO Model Zoo SSD detectors in OMZ_MODELS (e.g. face-detection-retail-0004;
            raw BGR 0-255 at the model's own input size, [1,1,N,7] output)

Protocol: Unix domain socket, length-prefixed pickle.
  request : {"frame": np.ndarray(H,W,3) BGR uint8, "conf": float, "target_class": int|None,
             "model": "<name>"}
  response: {"ok": True, "dets": [(x,y,w,h,cls,conf), ...]}  on success
            {"ok": False}                                     on error / model not loaded
                                                              (client -> CPU fallback)

If no MYRIAD device or no model compiles, the service exits non-zero and the app uses its
built-in CPU paths (onnxruntime for objects, Haar cascade for faces).

Config via environment variables:
  MODELS_DIR  folder holding YOLO <name>.onnx files  (default: ./models)
  NCS_SOCK    Unix socket to bind                    (default: /tmp/ncs_infer.sock)
"""
import os
import socket
import struct
import pickle
import sys
import time
import glob
import numpy as np
import cv2
import openvino.runtime as ov

HERE = os.path.dirname(os.path.abspath(__file__))
SOCK_PATH = os.environ.get("NCS_SOCK", "/tmp/ncs_infer.sock")
MODELS_DIR = os.environ.get("MODELS_DIR", os.path.join(HERE, "models"))

# OpenVINO Model Zoo SSD detectors (IR .xml). Skipped automatically if the file isn't present.
OMZ_MODELS = {
    "face-detection-retail-0004": {
        "path": os.path.join(HERE, "omz", "face-detection-retail-0004", "face-detection-retail-0004.xml"),
        "kind": "ssd",
    },
}


def recvall(conn, n):
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def recv_msg(conn):
    raw = recvall(conn, 4)
    if not raw:
        return None
    n = struct.unpack(">I", raw)[0]
    data = recvall(conn, n)
    if data is None:
        return None
    return pickle.loads(data)


def send_msg(conn, obj):
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    conn.sendall(struct.pack(">I", len(data)) + data)


def infer(entry, frame, conf_thresh, target_class):
    """Dispatch on model kind. Returns list of (x, y, w, h, cls, conf) in frame coords."""
    compiled, out_port, kind, (in_h, in_w) = entry
    fh, fw = frame.shape[:2]

    if kind == "yolo":
        img = cv2.resize(frame, (in_w, in_h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))[np.newaxis]
        outputs = compiled([img])[out_port][0]  # (84, N)
        outputs = outputs.T                      # (N, 84)
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
            x1 = int((cx - w / 2) * fw / in_w)
            y1 = int((cy - h / 2) * fh / in_h)
            bw = int(w * fw / in_w)
            bh = int(h * fh / in_h)
            boxes.append((x1, y1, bw, bh, cls, conf))
        return boxes

    # kind == "ssd": raw BGR at model input size, output [1,1,N,7] normalized
    img = cv2.resize(frame, (in_w, in_h)).astype(np.float32)  # BGR, no normalization
    img = np.transpose(img, (2, 0, 1))[np.newaxis]
    outputs = compiled([img])[out_port]
    dets = np.array(outputs).reshape(-1, 7)
    boxes = []
    for d in dets:
        _, label, conf, xmin, ymin, xmax, ymax = d
        conf = float(conf)
        if conf < conf_thresh:
            continue
        cls = int(label)
        if target_class is not None and cls != target_class:
            continue
        x1 = int(xmin * fw)
        y1 = int(ymin * fh)
        bw = int((xmax - xmin) * fw)
        bh = int((ymax - ymin) * fh)
        boxes.append((x1, y1, bw, bh, cls, conf))
    return boxes


def wait_for_myriad(core, timeout=45.0, interval=2.0):
    """After a process releases the NCS2, the Myriad X re-boots its firmware and briefly
    disappears from USB. Poll until it re-enumerates rather than exiting immediately."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if "MYRIAD" in core.available_devices:
            return True
        print("waiting for MYRIAD to enumerate...", flush=True)
        time.sleep(interval)
    return "MYRIAD" in core.available_devices


def model_registry():
    """{name: {"path", "kind"}} for YOLO ONNX in MODELS_DIR + present OMZ SSD models."""
    reg = {}
    for p in sorted(glob.glob(os.path.join(MODELS_DIR, "*.onnx"))):
        reg[os.path.splitext(os.path.basename(p))[0]] = {"path": p, "kind": "yolo"}
    for name, info in OMZ_MODELS.items():
        if os.path.exists(info["path"]):
            reg[name] = info
    return reg


def main():
    core = ov.Core()
    if not wait_for_myriad(core):
        print("ERROR: MYRIAD device not present after wait; exiting so app uses CPU fallback.", flush=True)
        sys.exit(1)

    reg = model_registry()
    if not reg:
        print(f"ERROR: no models found (MODELS_DIR={MODELS_DIR}); exiting.", flush=True)
        sys.exit(1)

    compiled = {}  # name -> (compiled_model, out_port, kind, (in_h, in_w))
    for name, info in reg.items():
        try:
            t0 = time.time()
            model = core.read_model(info["path"])
            ish = list(model.input().shape)  # [1,3,H,W]
            in_h, in_w = int(ish[2]), int(ish[3])
            cm = core.compile_model(model, "MYRIAD")
            cm([np.zeros((1, 3, in_h, in_w), np.float32)])  # warmup
            compiled[name] = (cm, cm.output(0), info["kind"], (in_h, in_w))
            print(f"compiled {name} [{info['kind']} {in_w}x{in_h}] on MYRIAD ({time.time()-t0:.1f}s)", flush=True)
        except Exception as e:
            print(f"WARN: could not compile {name}: {repr(e)[:160]}", flush=True)

    if not compiled:
        print("ERROR: no model compiled on MYRIAD; exiting so app uses CPU fallback.", flush=True)
        sys.exit(1)

    default_model = "yolov8s" if "yolov8s" in compiled else next(iter(compiled))
    print(f"NCS2 service ready on {SOCK_PATH} | models: {list(compiled)} | default: {default_model}", flush=True)

    if os.path.exists(SOCK_PATH):
        os.unlink(SOCK_PATH)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(SOCK_PATH)
    os.chmod(SOCK_PATH, 0o660)
    srv.listen(4)

    while True:
        conn, _ = srv.accept()
        try:
            while True:
                req = recv_msg(conn)
                if req is None:
                    break
                entry = compiled.get(req.get("model", default_model))
                if entry is None:
                    send_msg(conn, {"ok": False})  # model not loaded -> app falls back
                    continue
                try:
                    dets = infer(entry, req["frame"], req.get("conf", 0.45), req.get("target_class"))
                    send_msg(conn, {"ok": True, "dets": dets})
                except Exception as e:
                    print("inference error:", repr(e)[:200], flush=True)
                    send_msg(conn, {"ok": False})
        except Exception as e:
            print("connection error:", repr(e)[:200], flush=True)
        finally:
            conn.close()


if __name__ == "__main__":
    main()
