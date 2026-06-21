# Models

This project uses **YOLOv8** exported to ONNX (640x640 fixed input, 80 COCO classes). It can
hold multiple models and switch between them at runtime; by default it ships `yolov8n` (fast)
and `yolov8s` (more accurate). Model files are not committed - export them yourself:

```bash
pip install ultralytics
yolo export model=yolov8n.pt format=onnx imgsz=640   # -> yolov8n.onnx
yolo export model=yolov8s.pt format=onnx imgsz=640   # -> yolov8s.onnx
# place the .onnx files in this folder
```

Any `<name>.onnx` you drop in this folder is auto-discovered: the NCS2 service pre-compiles it
and the app exposes it via `/track/model`. Or download prebuilt ONNX files from the Ultralytics
releases and drop them here.

Both the NCS2 service (`ncs_infer.py`) and the CPU fallback (`app.py`) load this same ONNX
file. OpenVINO reads the ONNX directly and compiles it to FP16 for the Myriad X at runtime -
no separate IR conversion step is needed.

Override the path with the `YOLO_MODEL` environment variable if you keep it elsewhere.
