# Building OpenVINO 2022.3.2 with NCS2 (Myriad X) support on Raspberry Pi 4 / Debian Trixie / GCC 14

The Intel Neural Compute Stick 2 (NCS2 / Myriad X) was discontinued and its **MYRIAD plugin
support ends after the OpenVINO 2022.3 LTS line**. There are **no prebuilt aarch64 packages**
that include the MYRIAD device plugin - `pip install openvino==2022.3.*` gives you the Python
package but not the device plugin. The only way to use an NCS2 on a modern 64-bit Pi is to
**build OpenVINO 2022.3.2 from source**.

This guide reproduces that build on **Debian Trixie (13) / aarch64 / GCC 14.2**, where the
2022-era codebase hits two compiler issues that don't occur on the older GCC 12 toolchains
most existing guides assume. Both are fixed with global compiler flags - no per-file patching.

## Why 2022.3.2
- NCS2 / MYRIAD support is dropped in OpenVINO 2023+. Those builds will not see the stick.
- Prebuilt `openvino==2022.3.*` wheels do not ship the MYRIAD device plugin on aarch64.

## Prerequisites
- Raspberry Pi 4 (4 GB+), 64-bit OS, NCS2 plugged in (`lsusb` shows `03e7:2485`).
- ~8 GB free disk, a few hours of build time.

## Step 0 - apt dependencies
```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake ninja-build git \
  python3-dev libusb-1.0-0-dev libudev-dev pkg-config lld fdupes curl \
  libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev libffi-dev liblzma-dev
```

## Step 1 - Python 3.10 (the system Python on Trixie is 3.13, too new for these bindings)
OpenVINO 2022.3.2's Python bindings target 3.7-3.10 (3.11 works with a warning). Build 3.10
with pyenv:
```bash
curl -fsSL https://pyenv.run | bash
~/.pyenv/bin/pyenv install 3.10.14
~/.pyenv/versions/3.10.14/bin/python -m venv ~/ov-venv
source ~/ov-venv/bin/activate
python -m pip install -U pip setuptools wheel cython "numpy>=2,<2.3.0"
```

## Step 2 - clone OpenVINO 2022.3.2
```bash
cd ~
git clone --branch 2022.3.2 --recursive --depth 1 --shallow-submodules \
  https://github.com/openvinotoolkit/openvino.git openvino-2022
cd openvino-2022
```

## Step 3 - configure (core + MYRIAD only, GCC-14 fixes applied)
```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_PYTHON=ON -DENABLE_WHEEL=OFF \
  -DENABLE_INTEL_MYRIAD=ON -DENABLE_INTEL_CPU=OFF \
  -DENABLE_GAPI_PREPROCESSING=OFF \
  -DENABLE_TESTS=OFF -DENABLE_SAMPLES=OFF -DENABLE_DOCS=OFF \
  -DCMAKE_C_FLAGS="-D_GNU_SOURCE -include stdint.h" \
  -DCMAKE_CXX_FLAGS="-include cstdint" \
  -DCMAKE_EXE_LINKER_FLAGS="-fuse-ld=lld -Wl,--no-keep-memory" \
  -DCMAKE_SHARED_LINKER_FLAGS="-fuse-ld=lld -Wl,--no-keep-memory" \
  -DPYTHON_EXECUTABLE="$(which python)" \
  -DCMAKE_INSTALL_PREFIX="$PWD/install"
```

### The GCC-14 fixes (why those flags matter)
1. **`error: 'uint8_t' was not declared`** - GCC 13+ stopped transitively including
   `<cstdint>`, so headers that use `uint8_t` without including it fail. Fixed globally with
   `-include cstdint` (C++) and `-include stdint.h` (C). No source edits.
2. **`mvnc_api.c: unknown type name 'Dl_info' / implicit declaration of 'dladdr'`** - these
   are GNU extensions gated behind `_GNU_SOURCE`. The file already `#define _GNU_SOURCE`s
   itself, but the forced `-include stdint.h` pulls in glibc's `features.h` *before* that
   `#define` is seen, so the extensions stay hidden. Fixed by defining it on the command line
   (`-D_GNU_SOURCE`), which applies before any include.
3. `-DENABLE_GAPI_PREPROCESSING=OFF` sidesteps a narrowing error in a bundled OpenCV-HAL NEON
   header (also a GCC 12+ issue).

If you change `CMAKE_CXX_FLAGS` after a partial build, ninja rebuilds every C++ object.
Changing only `CMAKE_C_FLAGS` rebuilds just the (few) C files - useful for fix #2.

## Step 4 - build + install
```bash
# 4 GB Pi: add swap for the link step, then build with -j3 (leaves a core for the OS)
sudo fallocate -l 3G /swapfile-ov && sudo chmod 600 /swapfile-ov \
  && sudo mkswap /swapfile-ov && sudo swapon /swapfile-ov
cmake --build build -j3
cmake --install build
```

## Step 5 - udev rule for the NCS2, then replug the stick
```bash
sudo cp install/install_dependencies/97-myriad-usbboot.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger && sudo ldconfig
# unplug and replug the NCS2
```

## Step 6 - smoke test
```bash
source install/setupvars.sh
python - <<'PY'
import openvino.runtime as ov
ie = ov.Core()
print("Devices:", ie.available_devices)
if "MYRIAD" in ie.available_devices:
    print("Myriad:", ie.get_property("MYRIAD", "FULL_DEVICE_NAME"))
PY
# Expect: Devices: ['MYRIAD']  +  Intel Movidius Myriad X VPU
```

## Troubleshooting
- `Devices: []` - source `setupvars.sh` in the **same shell**; check `lsusb | grep 03e7`;
  confirm the udev rule is installed; replug; try once with `sudo -E` to rule out permissions.
- `Failed to find booted device after boot` - USB power / re-enumeration. Use a powered port,
  replug, retry. After any process that held the stick exits, the Myriad X re-boots its
  firmware and briefly disappears - wait ~10-30 s before retrying.
- Link step appears stuck - it's RAM pressure; `lld` + `--no-keep-memory` + swap + `-j2/3`
  is what keeps a 4 GB Pi alive through it.

## Result on this setup
- Build: OpenVINO 2022.3.2, MYRIAD plugin, on Pi 4 / Trixie / GCC 14.2.
- YOLOv8n 640x640: **~151 ms/frame (6.6 FPS) on the NCS2** vs **~545 ms (1.8 FPS) on CPU**
  (onnxruntime, 4 threads) - roughly 3.6x faster, and it frees the CPU cores.
