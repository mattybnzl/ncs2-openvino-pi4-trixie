#!/bin/bash
# Backup the time-consuming, NOT-in-git work on this Pi into one timestamped tarball:
#   - the from-source OpenVINO 2022.3.2 build (install + build cache + source)
#   - the two venvs (ov-venv py3.10+OpenVINO, pantilt-env py3.13+picamera2)
#   - the pyenv-built Python 3.10.14
#   - a reproducibility manifest (pip freezes, udev rule, swap, toolchain versions)
# The app code + models live in GitHub, so they are NOT re-bundled here by default.
#
# Restore = untar to the SAME home path on the same / an identical Pi (venvs and
# setupvars.sh hold absolute paths). For a different layout, rebuild from the manifest
# + docs/NCS2-OPENVINO-BUILD-GUIDE.md.
#
# Usage:   ./pi-backup.sh            # full bundle (~1-1.5 GB compressed)
#          LEAN=1 ./pi-backup.sh     # skip openvino-2022/build + source (runtime-only, smaller)
set -u

DEST="${BACKUP_DIR:-$HOME/pi-backups}"
STAMP="$(date +%Y-%m-%d)"
WORK="$DEST/pi4-ncs2-$STAMP"
mkdir -p "$WORK"

echo "Backup workdir: $WORK"

# --- Reproducibility manifest ---------------------------------------------
MAN="$WORK/MANIFEST.txt"
{
  echo "=== Pi4 NCS2 backup manifest ($STAMP) ==="
  echo "host:   $(hostname)"
  echo "uname:  $(uname -a)"
  echo "gcc:    $(gcc --version | head -1)"
  echo "pyenv:  $(ls ~/.pyenv/versions 2>/dev/null | tr '\n' ' ')"
  echo
  echo "=== udev rule (/etc/udev/rules.d/97-myriad-usbboot.rules) ==="
  cat /etc/udev/rules.d/97-myriad-usbboot.rules 2>/dev/null || echo "(not readable)"
  echo
  echo "=== swap ==="
  ls -lh /swapfile-ov 2>/dev/null
  swapon --show 2>/dev/null
} > "$MAN"

# Copy the udev rule itself (so restore doesn't need manual retyping)
cp /etc/udev/rules.d/97-myriad-usbboot.rules "$WORK/97-myriad-usbboot.rules" 2>/dev/null

# pip freezes (so the venvs can be rebuilt if a binary restore won't do)
if [ -x "$HOME/ov-venv/bin/pip" ]; then
  "$HOME/ov-venv/bin/pip" freeze > "$WORK/requirements-ov-venv.txt" 2>/dev/null
  echo "  froze ov-venv ($(wc -l < "$WORK/requirements-ov-venv.txt") pkgs)"
fi
if [ -x "$HOME/pantilt-env/bin/pip" ]; then
  "$HOME/pantilt-env/bin/pip" freeze > "$WORK/requirements-pantilt-env.txt" 2>/dev/null
  echo "  froze pantilt-env ($(wc -l < "$WORK/requirements-pantilt-env.txt") pkgs)"
fi

# --- The bundle -----------------------------------------------------------
BUNDLE="$DEST/pi4-ncs2-backup-$STAMP.tar.gz"
echo "Creating $BUNDLE ..."

# Paths relative to $HOME so the tar restores cleanly under any home dir
cd "$HOME" || exit 1
EXCLUDES=(--exclude='*/__pycache__' --exclude='*.log')
TARGETS=(ov-venv pantilt-env .pyenv/versions/3.10.14)

if [ "${LEAN:-0}" = "1" ]; then
  echo "  LEAN mode: install tree only (no build cache / source)"
  TARGETS+=(openvino-2022/install)
else
  TARGETS+=(openvino-2022)
fi

# Include the manifest dir (relative path) in the same tar
tar czf "$BUNDLE" "${EXCLUDES[@]}" \
  -C "$DEST" "pi4-ncs2-$STAMP" \
  -C "$HOME" "${TARGETS[@]}" 2>/dev/null

if [ -f "$BUNDLE" ]; then
  echo "Done: $BUNDLE  ($(du -h "$BUNDLE" | cut -f1))"
  echo
  echo "Pull it to your main machine with:"
  echo "  rsync -avP -e 'ssh -i ~/.ssh/ubuntu_to_pi' matt@$(hostname -I | awk '{print $1}'):$BUNDLE  <dest-on-main>/"
else
  echo "ERROR: bundle not created" >&2; exit 1
fi
