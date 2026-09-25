#!/usr/bin/env bash
# Develop the raws, verify them against photo-manifest.json, and survey:
# intake (EXIF from the raws beside the JPEGs), card detection, survey.
#   survey.sh [raw-directory [work-directory]]
# Developing needs rawpy and OpenCV (index_scanner's venv: set PYTHON).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"
raws="${1:-/ceph/christian/Photos/library/2026/2026-09-25}"
work="${2:-$here/work}"
python="${PYTHON:-$HOME/workspace/playground/index_scanner/.venv/bin/python}"
mkdir -p "$work/photos"
"$python" - "$raws" "$here/photo-manifest.json" "$work/photos" <<'PY'
import hashlib, json, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
raws, manifest, out = map(Path, sys.argv[1:])
rows = json.loads(manifest.read_text())["photos"]
for row in rows:
    path = raws / row["file"]
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if path.stat().st_size != row["bytes"] or digest != row["sha256"]:
        raise SystemExit(f"Raw differs from the saved observations: {path}")
print(f"Verified {len(rows)} raws")
sys.path.insert(0, str(Path.home() / "workspace/playground/index_scanner"))
from index_scanner.develop import develop  # LibRaw, camera white balance, camera JPEG frame

def one(row):
    raw = raws / row["file"]
    jpg = out / (raw.stem + ".JPG")
    link = out / raw.name
    if not link.exists():
        link.symlink_to(raw)  # the intake reads the EXIF from the raw beside the JPEG
    if not jpg.exists():
        develop(raw, jpg)
with ThreadPoolExecutor(8) as ex:
    list(ex.map(one, rows))
print(f"Developed {len(rows)} raws into {out}")
PY
v="$root/target/release/volumetric_cli"
"$v" view-import --stills "$work/photos" --session keytag-dslr-0 -o "$work/intake.vviews" > "$work/intake.log"
"$v" view-detect -i "$work/intake.vviews" --card "$here/card.json" -o "$work/detected.vviews" > "$work/detect.log"
"$v" view-survey -i "$work/detected.vviews" -o "$work/survey.vviews" --report "$work/survey.json" > "$work/survey.log"
"$v" view-list -i "$work/survey.vviews" --json > "$work/cameras.json"
cat "$work/intake.log" "$work/survey.log"
