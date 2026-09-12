#!/bin/bash
# Rebuild chair_base.vproj: the office-chair base (five-arm star, gas lift,
# tilt mechanism) from the chairbase-dslr-1 scan's measurements (PLAN.md).
# Base frame: origin on the floor under the lift axis, z up, arm 0 along +x.
# `base_world` is the same model posed into the survey's card frame, for
# checking against the scan (verify.sh). All coordinates metres.
set -e
L="$(cd "$(dirname "$0")" && pwd)"
cd "$L/../.."
V=./target/release/volumetric_cli
P=chair_base.vproj
rm -f $P
$V project-new --output $P >/dev/null

# The evidence's photographs (evidence.sh writes the selection): ten
# surveyed views of the base, so the model can be looked through them
# here (`render -i chair_base.vproj --asset base_world --through DSC00742
# --overlay edge`) and in the GUI.
SCAN=${SCAN:-/ceph/christian/index_scanner}
VIEWS=${CHAIR_VIEWS:-$SCAN/sessions/chairbase-dslr-1-all/demo/chair_views.vviews}
if [ -f "$VIEWS" ]; then
    $V project-add-asset -p $P -i "$VIEWS" --type view-set --asset-id views >/dev/null
else
    echo "no $VIEWS; building without the photographs (run evidence.sh first)"
fi

op() { $V project-add-op --project $P "$@" >/dev/null; }
PLANE='json:{"kind":"plane"}'
UNION='json:{"op":"union"}'
ISECT='json:{"op":"intersect"}'

# --- Hub and gas lift: one section revolved about z (sketch x = radius,
# sketch y = height). Hub r 54 from z 133 to 170 with a chamfer to the
# collar at 178; collar r 30 to 190; lift outer tube r 25.5 to 265; inner
# tube r 15 to the mechanism at 410.
op --operator path_sketch_operator --input 'json:{"path":"M 0 0.133 H 0.054 V 0.170 L 0.048 0.178 H 0.030 V 0.190 H 0.0255 V 0.265 H 0.015 V 0.410 H 0 Z"}' --output-id column_sketch --no-export
op --operator revolve_operator --input asset:column_sketch --input none --output-id column --no-export

# --- One arm along +x: plan outline (x, y) extruded up, intersected with
# the elevation (x, z) extruded across, then rounded 6 mm.
# Plan: half-width 25 at the hub, 34 at r 250, a 25 round at the tip r 335.
op --operator path_sketch_operator --input 'json:{"path":"M 0.04 -0.025 L 0.25 -0.034 L 0.31 -0.025 A 0.025 0.025 0 0 1 0.31 0.025 L 0.25 0.034 L 0.04 0.025 Z"}' --output-id arm_plan_sketch --no-export
op --operator extrude_operator --input asset:arm_plan_sketch --input 'json:{"height":0.2}' --input none --output-id arm_plan --no-export
# Elevation: top 168 at the hub falling to 140 at r 270 and 100 at the
# tip; 22 thick. Plane (x, z) at y = +50 with normal x cross z = -y, so the
# 100 extrude spans y -50..50.
op --operator path_sketch_operator --input 'json:{"path":"M 0.04 0.146 L 0.04 0.168 L 0.15 0.160 L 0.27 0.140 L 0.335 0.100 L 0.335 0.080 L 0.27 0.118 L 0.15 0.138 Z"}' --output-id arm_side_sketch --no-export
op --operator subspace_operator --input "$PLANE" --input 'json:[0.0,0.05,0.0]' --input 'json:[1.0,0.0,0.0]' --input 'json:[0.0,0.0,1.0]' --output-id arm_side_plane --no-export
op --operator extrude_operator --input asset:arm_side_sketch --input 'json:{"height":0.1}' --input asset:arm_side_plane --output-id arm_side --no-export
op --operator boolean_operator --input asset:arm_plan --input asset:arm_side --input "$ISECT" --output-id arm_sharp --no-export
op --operator offset_operator --input asset:arm_sharp --input 'json:{"distance":-0.006,"resolution":256}' --output-id arm_eroded --no-export
op --operator offset_operator --input asset:arm_eroded --input 'json:{"distance":0.006,"resolution":256}' --output-id arm_round --no-export

# --- Caster at the tip: a stem into the arm and a 65 wheel, 24 wide,
# trailing 15 outward (the swivel state in the scan is arbitrary).
op --operator cylinder_operator --input 'json:{"radius":0.011}' --input 'json:[0.315,0.0,0.045]' --input 'json:[0.315,0.0,0.11]' --output-id stem --no-export
op --operator cylinder_operator --input 'json:{"radius":0.0325}' --input 'json:[0.33,-0.012,0.033]' --input 'json:[0.33,0.012,0.033]' --output-id wheel --no-export
op --operator boolean_operator --input asset:arm_round --input asset:stem --input asset:wheel --input "$UNION" --output-id arm --no-export
op --operator pattern_operator --input asset:arm --input 'json:{"circular":{"count":5,"axis":"z"}}' --output-id arms --no-export

# --- Tilt mechanism as two boxes in its rail frame (u along the rail,
# v across, origin on the lift axis), then turned to the rail's azimuth of
# 60.8 deg in the base frame (50.4 in the world, from a line fit to the
# top faces): rail 300 x 60 x 46 with the lift under its first quarter,
# top at 456; the seat bracket 30 x 230 x 8 across its near end.
op --operator rectangular_prism_operator --input 'json:{}' --input 'json:[-0.07,-0.017,0.41]' --input 'json:[0.23,0.043,0.456]' --output-id rail_box --no-export
op --operator rectangular_prism_operator --input 'json:{}' --input 'json:[-0.09,-0.115,0.448]' --input 'json:[-0.06,0.115,0.456]' --output-id bar_box --no-export
op --operator boolean_operator --input asset:rail_box --input asset:bar_box --input "$UNION" --output-id mechanism_u --no-export
op --operator pose_operator --input asset:mechanism_u --input 'json:{"rotate":{"rz_deg":60.8}}' --output-id mechanism --no-export

# --- Assembly, and the same posed into the scan's frame: arm 0 sits at
# azimuth -10.4 deg, the lift axis at (0.2569, 0.1509).
op --operator boolean_operator --input asset:column --input asset:arms --input asset:mechanism --input "$UNION" --output-id base
op --operator pose_operator --input asset:base --input 'json:{"rotate":{"rz_deg":-10.4},"translate":{"dx":0.2569,"dy":0.1509}}' --output-id base_world

$V project-run --project $P
