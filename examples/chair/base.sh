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
# the elevation (x, z) extruded across, then rounded 5 mm. Both profiles
# are the refuse2 cloud's (PLAN.md, measurements): the arm is 28 wide at
# the hub, 50 at r 260, 30 at the socket, and its top falls from 166 at
# the hub to 128 at r 300 and 90 at r 340, where the caster socket hangs.
op --operator path_sketch_operator --input 'json:{"path":"M 0.04 -0.014 L 0.14 -0.015 L 0.20 -0.022 L 0.26 -0.025 L 0.30 -0.022 L 0.335 -0.015 A 0.015 0.015 0 0 1 0.335 0.015 L 0.30 0.022 L 0.26 0.025 L 0.20 0.022 L 0.14 0.015 L 0.04 0.014 Z"}' --output-id arm_plan_sketch --no-export
op --operator extrude_operator --input asset:arm_plan_sketch --input 'json:{"height":0.2}' --input none --output-id arm_plan --no-export
# Elevation: the measured top, and an underside 18 thick at the hub, 26
# at mid-arm, 22 at the tip (unseen by every camera; assumed). Plane
# (x, z) at y = +50 with normal x cross z = -y, so the 100 extrude spans
# y -50..50.
op --operator path_sketch_operator --input 'json:{"path":"M 0.04 0.148 L 0.04 0.166 L 0.10 0.165 L 0.14 0.162 L 0.18 0.156 L 0.20 0.150 L 0.22 0.144 L 0.24 0.140 L 0.26 0.137 L 0.28 0.134 L 0.30 0.128 L 0.32 0.115 L 0.34 0.092 L 0.35 0.078 L 0.35 0.060 L 0.33 0.070 L 0.31 0.098 L 0.29 0.108 L 0.26 0.111 L 0.22 0.118 L 0.18 0.130 L 0.14 0.140 L 0.10 0.147 Z"}' --output-id arm_side_sketch --no-export
op --operator subspace_operator --input "$PLANE" --input 'json:[0.0,0.05,0.0]' --input 'json:[1.0,0.0,0.0]' --input 'json:[0.0,0.0,1.0]' --output-id arm_side_plane --no-export
op --operator extrude_operator --input asset:arm_side_sketch --input 'json:{"height":0.1}' --input asset:arm_side_plane --output-id arm_side --no-export
op --operator boolean_operator --input asset:arm_plan --input asset:arm_side --input "$ISECT" --output-id arm_sharp --no-export
op --operator offset_operator --input asset:arm_sharp --input 'json:{"distance":-0.005,"resolution":256}' --output-id arm_eroded --no-export
op --operator offset_operator --input asset:arm_eroded --input 'json:{"distance":0.005,"resolution":256}' --output-id arm_round --no-export

# --- Caster at the socket (r 340): a stem down from the arm tip and a 65
# wheel, 24 wide, trailing 20 outward (the swivel state in the scan is
# arbitrary; the cloud's wheels sit at r 360).
op --operator cylinder_operator --input 'json:{"radius":0.011}' --input 'json:[0.34,0.0,0.03]' --input 'json:[0.34,0.0,0.085]' --output-id stem --no-export
op --operator cylinder_operator --input 'json:{"radius":0.0325}' --input 'json:[0.36,-0.012,0.0325]' --input 'json:[0.36,0.012,0.0325]' --output-id wheel --no-export
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
