#!/bin/bash
# Check chair_base.vproj against the evidence: the posed model drawn
# through the surveyed photographs over the pictures, and beside the scan
# in plan and elevation sections. Outputs under target/chair/. The
# evidence project is the first argument (default: the session's demo).
set -e
L="$(cd "$(dirname "$0")" && pwd)"
cd "$L/../.."
V=./target/release/volumetric_cli
SCAN=${SCAN:-/ceph/christian/index_scanner}
E=${1:-$SCAN/sessions/chairbase-dslr-1/demo/chair_evidence.vproj}
O=target/chair
mkdir -p $O

# The posed base as one model in a copy of the evidence project.
$V project-export -p chair_base.vproj --asset base_world -o $O >/dev/null
cp "$E" $O/verify.vproj
$V project-add-model -p $O/verify.vproj -i $O/base_world.wasm --asset-id base_world >/dev/null

# Through three photographs: a low elevation, the top-down, an oblique.
for view in DSC00730 DSC00742 DSC00750; do
    $V render -i $O/verify.vproj --asset base_world --through $view --overlay edge \
        --width 1548 --height 1032 --resolution 256 -q -o $O/through_${view}.png
done

# Plan and elevation sections of the model over the scan's cloud, at
# 0.5 mm per pixel about the lift axis.
COMMON="--asset tsdf_cloud --asset base_world --up 0,0,1 --projection ortho --ortho-scale 0.8 --width 1600 --height 1600 --grid 0 --no-ssao --resolution 256 -q --wireframe"
$V render -i $O/verify.vproj $COMMON --camera-pos 0.2569,0.1509,2 --camera-target 0.2569,0.1509,0.2 --camera-up 0,1,0 -o $O/plan.png
$V render -i $O/verify.vproj $COMMON --camera-pos 0.2569,-2,0.25 --camera-target 0.2569,0.1509,0.25 -o $O/elevation.png
ls -la $O/*.png
