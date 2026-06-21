#!/usr/bin/env python3
"""
NCS2 (Intel Movidius Myriad X) inference microservice.

Runs in the OpenVINO Python 3.10 venv with setupvars.sh sourced, because the OpenVINO
2022.3.2 Python bindings are built for Python 3.10 and CANNOT be imported in the pan-tilt
app's system Python (3.13 on Trixie, where picamera2/libcamera are 3.13-locked C extensions).

Protocol: Unix domain socket, length-prefixed pickle.
  request : {"frame": np.ndarray(H,W,3) BGR uint8, "conf": float, "target_class": int|None}
  response: {"ok": True, "dets": [(x,y,w,h,cls,conf), ...]}  on success
            {"ok": False}                                     on inference error (client -> CPU fallback)

If the MYRIAD device can't be acquired at startup, the service exits non-zero and the app
simply never connects, so it uses its built-in CPU onnxruntime path.

Config via environment variables:
  YOLO_MODEL  path to yolov8n.onnx        (default: ./models/yolov8n.onnx)
  NCS_SOCK    Unix socket to bind         (default: /tmp/ncs_infer.sock)
"""
import os
import socket
import struct
import pickle
import sys
import time
import numpy as np
import cv2
import openvino.runtime as ov

HERE = os.path.dirname(os.path.abspath(__file__))
SOCK_PATH = os.environ.get("NCS_SOCK", "/tmp/ncs_infer.sock")
MODEL_PATH = os.environ.get("YOLO_MODEL", os.path.join(HERE, "models", "yolov8n.onnx"))
SIZE = 640  # YOLOv8n fixed input


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


def infer(compiled, out_port, frame, conf_thresh, target_class):
    """Same preprocessing/postprocessing as the app's CPU path, run on MYRIAD."""
    fh, fw = frame.shape[:2]
    img = cv2.resize(frame, (SIZE, SIZE))
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
        x1 = int((cx - w / 2) * fw / SIZE)
        y1 = int((cy - h / 2) * fh / SIZE)
        bw = int(w * fw / SIZE)
        bh = int(h * fh / SIZE)
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


def main():
    core = ov.Core()
    if not wait_for_myriad(core):
        print("ERROR: MYRIAD device not present after wait; exiting so app uses CPU fallback.", flush=True)
        sys.exit(1)
    model = core.read_model(MODEL_PATH)
    compiled = core.compile_model(model, "MYRIAD")
    out_port = compiled.output(0)
    # Warmup
    compiled([np.zeros((1, 3, SIZE, SIZE), np.float32)])
    print("NCS2 inference service ready on", SOCK_PATH, flush=True)

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
                try:
                    dets = infer(compiled, out_port, req["frame"],
                                 req.get("conf", 0.45), req.get("target_class"))
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
