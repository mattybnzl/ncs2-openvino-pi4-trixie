# Open Model Zoo (OMZ) models

Purpose-built FP16 detectors that run on the NCS2, used in addition to the YOLO models.
The IR files (`.xml` + `.bin`) are not committed - download them here (no `openvino-dev`
needed; pulled straight from the OMZ storage):

```bash
# face-detection-retail-0004 - 300x300 SSD face detector (~45 FPS on NCS2)
mkdir -p face-detection-retail-0004 && cd face-detection-retail-0004
BASE=https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.3/models_bin/1/face-detection-retail-0004/FP16
curl -fsSLO $BASE/face-detection-retail-0004.xml
curl -fsSLO $BASE/face-detection-retail-0004.bin
cd ..

# (optional) person-detection-retail-0013 - 320x544 SSD person detector (~7 FPS on NCS2)
mkdir -p person-detection-retail-0013 && cd person-detection-retail-0013
BASE=https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.3/models_bin/1/person-detection-retail-0013/FP16
curl -fsSLO $BASE/person-detection-retail-0013.xml
curl -fsSLO $BASE/person-detection-retail-0013.bin
```

The app uses `face-detection-retail-0004` for **face mode** when the NCS2 is present, and falls
back to the OpenCV Haar cascade on CPU when it isn't. The NCS service registers OMZ SSD models
in `OMZ_MODELS` (in `ncs_infer.py`) and skips any whose files are missing - so if you don't
download these, the app still runs (face mode just uses Haar).

Models are Intel Open Model Zoo, Apache-2.0 licensed.
