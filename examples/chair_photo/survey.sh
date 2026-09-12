#!/usr/bin/env bash
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"
photos="${1:-/ceph/christian/index_scanner/sessions/chairbase-dslr-1}"
work="${2:-$here/work}"
mkdir -p "$work"
python3 - "$photos" "$here/photo-manifest.json" <<'PY'
import hashlib, json, sys
from pathlib import Path
photos, manifest = map(Path, sys.argv[1:])
expected = json.loads(manifest.read_text())["photos"]
actual = {p.name for p in photos.iterdir() if p.suffix.lower() in {".jpg", ".jpeg"}}
if actual != {row["file"] for row in expected}:
    raise SystemExit("Photo list differs from the saved observations' session")
for row in expected:
    path = photos / row["file"]
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if path.stat().st_size != row["bytes"] or digest != row["sha256"]:
        raise SystemExit(f"Photo differs from the saved observations: {path}")
print(f"Verified {len(expected)} source photographs")
PY
v="$root/target/release/volumetric_cli"
"$v" view-import --stills "$photos" --session chairbase-dslr-1-bright \
    --field chairbase-dslr-1-card --setup chairbase-dslr-1 \
    -o "$work/intake.vviews" > "$work/intake.log"
"$v" view-detect -i "$work/intake.vviews" --card "$here/card.json" \
    -o "$work/detected.vviews" --json > "$work/detection.json"
"$v" view-survey -i "$work/detected.vviews" -o "$work/survey.vviews" \
    --report "$work/survey.json" > "$work/survey.log"
"$v" view-list -i "$work/survey.vviews" --json > "$work/cameras.json"
cat "$work/survey.log"
