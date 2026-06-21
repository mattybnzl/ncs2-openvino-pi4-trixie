# Model

This project uses **YOLOv8n** exported to ONNX (640x640 fixed input, 80 COCO classes).
The model file is not committed - export it yourself:

```bash
pip install ultralytics
yolo export model=yolov8n.pt format=onnx imgsz=640
# produces yolov8n.onnx -> place it in this folder
```

Or download a prebuilt `yolov8n.onnx` from the Ultralytics releases and drop it here as
`models/yolov8n.onnx`.

Both the NCS2 service (`ncs_infer.py`) and the CPU fallback (`app.py`) load this same ONNX
file. OpenVINO reads the ONNX directly and compiles it to FP16 for the Myriad X at runtime -
no separate IR conversion step is needed.

Override the path with the `YOLO_MODEL` environment variable if you keep it elsewhere.
