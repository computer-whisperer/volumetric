#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
V=./target/release/volumetric_cli
P=comet_racer.vproj
OUT=target/comet_racer
"$V" project-new --output "$P"
"$V" project-add-asset --project "$P" --input examples/comet_racer/car.wgsl --type wgsl --asset-id design
"$V" project-add-op --project "$P" --operator wgsl_script_operator --input asset:design --input 'json:{}' --output-id car
"$V" project-validate --project "$P"
"$V" project-export --project "$P" --output "$OUT"
"$V" mesh --input "$OUT/car.wasm" --output "$OUT/car.stl" --base-resolution 24 --max-depth 4 --no-simplify --quiet
"$V" render --input "$OUT/car.wasm" --output "$OUT/car.png" --base-resolution 24 --max-depth 4 --sharp-edges --views iso,iso-back,front,top --projection ortho --ortho-scale 0.135 --grid 0 --color e78c22 --background 141b24
