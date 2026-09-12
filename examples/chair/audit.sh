#!/bin/bash
# Measure chair_base.vproj against the scan: the signed distance from every
# point of the reference cloud to the posed model (cloud_distance), as a
# summary and as colour maps at +-15 mm in plan and elevation, under
# target/chair/. The reference is the unmasked TSDF cloud of the full run
# (first argument; default runs/chairbase-dslr-1-refuse2/cloud.ply), so
# the audit is clipped to a box around the base first: carpet and card
# would otherwise dominate the summary. The project to measure is the
# second argument (default chair_base.vproj), so two versions can be
# compared.
set -e
L="$(cd "$(dirname "$0")" && pwd)"
cd "$L/../.."
V=./target/release/volumetric_cli
SCAN=${SCAN:-/ceph/christian/index_scanner}
CLOUD=${1:-$SCAN/runs/chairbase-dslr-1-refuse2/cloud.ply}
BASE=${2:-chair_base.vproj}
O=target/chair
P=$O/audit.vproj
mkdir -p $O
rm -f $P

$V project-new --output $P >/dev/null
$V project-export -p "$BASE" --asset base_world -o $O >/dev/null
$V project-add-model -p $P -i $O/base_world.wasm --asset-id base_world >/dev/null
$V project-add-asset -p $P -i "$CLOUD" --type blob --asset-id cloud_ply >/dev/null
$V project-add-op -p $P --operator point_cloud_import_operator -i asset:cloud_ply -i none --output-id cloud --no-export >/dev/null
# The base's neighbourhood, above the carpet (z > 12 mm).
$V project-add-op -p $P --operator rectangular_prism_operator -i 'json:{}' -i 'json:[-0.15,-0.25,0.012]' -i 'json:[0.65,0.55,0.5]' --output-id region --no-export >/dev/null
$V project-add-op -p $P --operator mesh_clip_operator -i asset:cloud -i asset:region -i 'json:{"keep":"inside"}' --output-id near --no-export >/dev/null
$V project-add-op -p $P --operator cloud_distance_operator -i asset:near -i asset:base_world -i 'json:{"resolution":256}' --output-id audit >/dev/null

# Per zone, about the lift axis (0.2569, 0.1509): the arms in an annulus
# r 70..400 mm from z 80 to 200 (above the caster sockets), the casters
# in an annulus r 280..420 below z 80 (this one carries the carpet's
# fuzz, which an unmasked cloud piles 60 mm high around the wheels), the
# column within r 70 from z 120 to 400, the mechanism above z 400. Each
# zone's cloud is measured on its own so the summary is the zone's.
AX=0.2569,0.1509
$V project-add-op -p $P --operator cylinder_operator -i 'json:{"radius":0.40}' -i "json:[$AX,0.08]" -i "json:[$AX,0.20]" --output-id arms_outer --no-export >/dev/null
$V project-add-op -p $P --operator cylinder_operator -i 'json:{"radius":0.07}' -i "json:[$AX,0.12]" -i "json:[$AX,0.40]" --output-id column_zone --no-export >/dev/null
$V project-add-op -p $P --operator boolean_operator -i asset:arms_outer -i asset:column_zone -i 'json:{"op":"subtract"}' --output-id arms_zone --no-export >/dev/null
$V project-add-op -p $P --operator rectangular_prism_operator -i 'json:{}' -i 'json:[-0.15,-0.25,0.40]' -i 'json:[0.65,0.55,0.5]' --output-id mechanism_zone --no-export >/dev/null
$V project-add-op -p $P --operator cylinder_operator -i 'json:{"radius":0.42}' -i "json:[$AX,0.012]" -i "json:[$AX,0.08]" --output-id casters_outer --no-export >/dev/null
$V project-add-op -p $P --operator cylinder_operator -i 'json:{"radius":0.28}' -i "json:[$AX,0.0]" -i "json:[$AX,0.1]" --output-id casters_inner --no-export >/dev/null
$V project-add-op -p $P --operator boolean_operator -i asset:casters_outer -i asset:casters_inner -i 'json:{"op":"subtract"}' --output-id casters_zone --no-export >/dev/null
# The shell: points within 30 mm of the model, whatever zone. Room
# residual (carpet fuzz reaches 60 mm in an unmasked cloud) stays out,
# so this is the number for the parts that are modelled; parts that are
# not show only in the colour maps.
$V project-add-op -p $P --operator offset_operator -i asset:base_world -i 'json:{"distance":0.03,"resolution":192}' --output-id shell_zone --no-export >/dev/null
for zone in arms casters column mechanism shell; do
    $V project-add-op -p $P --operator mesh_clip_operator -i asset:near -i asset:${zone}_zone -i 'json:{"keep":"inside"}' --output-id ${zone}_cloud --no-export >/dev/null
    $V project-add-op -p $P --operator cloud_distance_operator -i asset:${zone}_cloud -i asset:base_world -i 'json:{"resolution":256}' --output-id ${zone}_audit >/dev/null
done
$V project-run -p $P --json > $O/audit.json
python3 - "$O/audit.json" <<'PY'
import json, sys
exports = {a["asset_id"]: a for a in json.load(open(sys.argv[1]))["exports"]}
print("zone             n  within 5 mm  10 mm  20 mm  p50 mm  inside")
for zone, label in [("audit", "all"), ("shell_audit", "shell 30 mm"), ("arms_audit", "arms"), ("casters_audit", "casters"), ("column_audit", "column"), ("mechanism_audit", "mechanism")]:
    d = exports[zone + "_summary"]["value"]
    print(f"{label:11s} {int(d['count']):7d}     {d['within_band']:.3f}  {d['within_2band']:.3f}  {d['within_4band']:.3f}  {d['abs_p50']*1000:6.1f}   {d['inside_fraction']:.3f}")
PY

C="--asset audit --color-field node:distance --color-range -0.015,0.015 --up 0,0,1 --projection ortho --grid 0 --no-ssao -q --width 1600 --height 1600 --ortho-scale 0.8"
$V render -i $P $C --camera-pos 0.2569,0.1509,2 --camera-target 0.2569,0.1509,0.2 --camera-up 0,1,0 -o $O/audit_plan.png
$V render -i $P $C --camera-pos 0.2569,-2,0.25 --camera-target 0.2569,0.1509,0.25 -o $O/audit_elev.png
echo "colour: purple -15 mm (model beyond the scan) .. teal 0 .. yellow +15 mm (scan beyond the model)"
