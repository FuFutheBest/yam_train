#!/usr/bin/env bash
# Install openpi (pinned commit) with its own uv venv, and download the pi05_base checkpoint.
#
#   bash scripts/setup_openpi.sh            # -> ../openpi, ../.cache/openpi   (see scripts/pi05_env.sh)
#
# Steps (each is skipped when already done):
#   1. git clone openpi at $OPENPI_COMMIT (GitHub; retried, the training server's TLS to GitHub is flaky)
#   2. uv (in ../.tools) + `uv sync` = openpi's locked dependencies (jax[cuda12] 0.5.3, torch 2.7.1, lerobot ...)
#      PyPI is slow from the server -> packages come from the Huawei Cloud PyPI mirror
#   3. gs://openpi-assets/checkpoints/pi05_base (~11 GB) + the PaliGemma tokenizer from Google Cloud Storage
#      over plain HTTPS, checked against the bucket's crc32c / md5
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
source scripts/pi05_env.sh 2>/dev/null || true
OPENPI_COMMIT="${OPENPI_COMMIT:-215abfb}"
PYPI_MIRROR="${PYPI_MIRROR:-https://repo.huaweicloud.com/repository/pypi/simple}"
TOOLS="$(dirname "$OPENPI_DIR")/.tools"
CACHE="$(dirname "$OPENPI_DATA_HOME")"

# 1. openpi source
if [ ! -d "$OPENPI_DIR/.git" ]; then
  for i in 1 2 3 4 5; do
    git clone -q https://github.com/Physical-Intelligence/openpi.git "$OPENPI_DIR" && break
    rm -rf "$OPENPI_DIR"; sleep 3
  done
fi
git -C "$OPENPI_DIR" checkout -q "$OPENPI_COMMIT"
echo "openpi at $(git -C "$OPENPI_DIR" log --oneline -1)"

# 2. venv
if [ ! -x "$TOOLS/bin/uv" ]; then
  python3 -m venv "$TOOLS"
  "$TOOLS/bin/pip" install -q -i "$PYPI_MIRROR" uv
fi
(
  cd "$OPENPI_DIR"
  export UV_CACHE_DIR="$CACHE/uv" UV_PYTHON_INSTALL_DIR="$CACHE/uv-python"
  export UV_DEFAULT_INDEX="$PYPI_MIRROR"
  export UV_PYTHON_INSTALL_MIRROR=https://registry.npmmirror.com/-/binary/python-build-standalone
  export UV_HTTP_TIMEOUT=300 GIT_LFS_SKIP_SMUDGE=1
  "$TOOLS/bin/uv" sync
)

# 3. checkpoint (openpi's cache layout: $OPENPI_DATA_HOME/<bucket>/<path>)
mkdir -p "$OPENPI_DATA_HOME" && cd "$OPENPI_DATA_HOME"
API="https://storage.googleapis.com/storage/v1/b/openpi-assets/o?prefix=checkpoints/pi05_base/"
curl -sf "$API&fields=items(name,size,md5Hash,crc32c)" > pi05_base_listing.json
"$PI05_PY" - <<'PY' > pi05_files.txt
import json
for i in json.load(open("pi05_base_listing.json"))["items"]:
    if not i["name"].endswith("/"):
        print(i["name"])
PY
# 6 parallel downloads; a stalled connection (<1 MB/s for 15 s) is restarted and resumed
xargs -P 6 -I{} sh -c 'f=openpi-assets/{}; mkdir -p "$(dirname "$f")"; for i in $(seq 200); do
  curl -sf --speed-limit 1000000 --speed-time 15 -C - -o "$f" "https://storage.googleapis.com/openpi-assets/{}" && break
  sleep 2; done' < pi05_files.txt
mkdir -p big_vision
[ -s big_vision/paligemma_tokenizer.model ] ||
  curl -sf -o big_vision/paligemma_tokenizer.model https://storage.googleapis.com/big_vision/paligemma_tokenizer.model
"$PI05_PY" - <<'PY'
import base64, hashlib, json, os
import google_crc32c
bad = 0
for i in json.load(open("pi05_base_listing.json"))["items"]:
    f = "openpi-assets/" + i["name"]
    if i["name"].endswith("/"):
        continue
    if not os.path.exists(f) or os.path.getsize(f) != int(i["size"]):
        print("size mismatch", f); bad += 1; continue
    crc, md5 = google_crc32c.Checksum(), hashlib.md5()
    with open(f, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            crc.update(b); md5.update(b)
    if base64.b64encode(crc.digest()).decode() != i["crc32c"] or (
            "md5Hash" in i and base64.b64encode(md5.digest()).decode() != i["md5Hash"]):
        print("checksum mismatch", f); bad += 1
print("pi05_base checkpoint:", "OK" if bad == 0 else f"{bad} BAD FILES (delete them and re-run)")
PY
