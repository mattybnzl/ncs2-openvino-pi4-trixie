#!/bin/bash
# Launch the NCS2 inference service (OpenVINO 3.10 venv) + the pan-tilt app (system venv).
# The two run as separate processes because OpenVINO (py3.10) and picamera2 (system py3.13)
# can't share an interpreter. They talk over a Unix socket. The app falls back to CPU if the
# service or stick is unavailable.
#
# Override any of these via environment variables before running:
OV_SETUPVARS="${OV_SETUPVARS:-$HOME/openvino-2022/install/setupvars.sh}"  # OpenVINO setupvars.sh
OV_VENV="${OV_VENV:-$HOME/ov-venv}"          # venv with OpenVINO (Python 3.10) + opencv
APP_VENV="${APP_VENV:-$HOME/pantilt-env}"    # venv with flask/picamera2/pantilthat/onnxruntime
APP_DIR="${APP_DIR:-$(cd "$(dirname "$0")" && pwd)}"  # this repo (where app.py / ncs_infer.py live)

# Stop any running instances
pkill -f "$APP_DIR/app.py" 2>/dev/null
pkill -f "$APP_DIR/ncs_infer.py" 2>/dev/null
sleep 1

# Start the NCS2 inference service.
# Activate the venv BEFORE sourcing setupvars.sh, so setupvars sets PYTHONPATH to the
# python3.10 bindings (it keys off the active python3). setupvars is sourced in a subshell
# so its PYTHONPATH/LD_LIBRARY_PATH do NOT leak into the app process.
nohup bash -c "source '$OV_VENV/bin/activate'; source '$OV_SETUPVARS' >/dev/null 2>&1; exec python '$APP_DIR/ncs_infer.py'" \
  < /dev/null >> "$APP_DIR/ncs.log" 2>&1 & disown

# Give the service time to acquire MYRIAD and compile the model before the app connects
sleep 5

# Start the pan-tilt app (clean env, system Python + picamera2)
nohup "$APP_VENV/bin/python3" "$APP_DIR/app.py" < /dev/null >> "$APP_DIR/app.log" 2>&1 & disown

echo "Started - http://$(hostname -I | awk '{print $1}'):5000"
echo "  NCS2 service log: $APP_DIR/ncs.log"
echo "  App log:          $APP_DIR/app.log"
