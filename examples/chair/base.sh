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
SUB='json:{"op":"subtract"}'

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

# --- Tilt mechanism, built in its own frame (u along the rail, v across
# to the left, w up; origin on the lift axis at z 0.456) from the
# photographs: hole centres triangulated from DSC00742 and DSC00760 (rays
# meet within 0.6 mm), surfaces from plane fits on clipped patches of the
# refuse2 cloud (PLAN.md). The rail's flat top is 50 wide from u -33 to
# +187 with the lever housing to 232, level: the two slots triangulate to
# one height, and the 3.65 deg the cloud's top-face fit showed was the
# dimples and the folds. Two dimples (14 wide,
# 3.5 deep, not through) at (46, -3) and (99, -5), and two 18 x 8
# mounting slots along the rail at u 121, v +11 and v -18. The seat
# bracket is a level 26 x 236 bar at u -51.5, skewed -1 deg, top 6.9 mm
# above the origin, with an 8 hole 53 and 57 mm each side and a formed
# end tab each side carrying a 10 x 12.5 slot: tab A (+v) rises 3.4 deg
# from a crease at v 85, tab B (-v) steps down 2.5 mm at v -88 and rises
# 6.7 deg. The frame sits at 60.8 deg in the base frame.
op --operator rectangular_prism_operator --input 'json:{}' --input 'json:[-0.033,-0.023,-0.046]' --input 'json:[0.187,0.027,0.0]' --output-id rail_body --no-export
op --operator rectangular_prism_operator --input 'json:{}' --input 'json:[0.187,-0.030,-0.046]' --input 'json:[0.232,0.030,0.0]' --output-id rail_housing --no-export
RECESS='json:{"radius":0.0068}'
op --operator cylinder_operator --input "$RECESS" --input 'json:[0.0458,-0.0028,-0.0035]' --input 'json:[0.0458,-0.0028,0.02]' --output-id rail_recess_a --no-export
op --operator cylinder_operator --input "$RECESS" --input 'json:[0.0994,-0.0050,-0.0035]' --input 'json:[0.0994,-0.0050,0.02]' --output-id rail_recess_b --no-export
# The slots: 18 x 8 stadiums along u, through the top.
RSLOT='json:{"radius":0.004}'
op --operator cylinder_operator --input "$RSLOT" --input 'json:[0.1163,0.0110,-0.05]' --input 'json:[0.1163,0.0110,0.02]' --output-id rslot_a1 --no-export
op --operator cylinder_operator --input "$RSLOT" --input 'json:[0.1263,0.0110,-0.05]' --input 'json:[0.1263,0.0110,0.02]' --output-id rslot_a2 --no-export
op --operator rectangular_prism_operator --input 'json:{}' --input 'json:[0.1163,0.0070,-0.05]' --input 'json:[0.1263,0.0150,0.02]' --output-id rslot_a3 --no-export
op --operator cylinder_operator --input "$RSLOT" --input 'json:[0.1160,-0.0184,-0.05]' --input 'json:[0.1160,-0.0184,0.02]' --output-id rslot_b1 --no-export
op --operator cylinder_operator --input "$RSLOT" --input 'json:[0.1260,-0.0184,-0.05]' --input 'json:[0.1260,-0.0184,0.02]' --output-id rslot_b2 --no-export
op --operator rectangular_prism_operator --input 'json:{}' --input 'json:[0.1160,-0.0224,-0.05]' --input 'json:[0.1260,-0.0144,0.02]' --output-id rslot_b3 --no-export
op --operator boolean_operator --input asset:rail_body --input asset:rail_housing --input "$UNION" --output-id rail_solid --no-export
op --operator boolean_operator --input asset:rail_solid --input asset:rail_recess_a --input asset:rail_recess_b --input asset:rslot_a1 --input asset:rslot_a2 --input asset:rslot_a3 --input asset:rslot_b1 --input asset:rslot_b2 --input asset:rslot_b3 --input "$SUB" --output-id rail_flat --no-export
op --operator pose_operator --input asset:rail_flat --input 'json:{}' --output-id rail --no-export
# The bracket, built on its own centreline (u = 0, top at w 0) then
# skewed and moved to u -51.5, top to +6.9. 4 mm plate.
op --operator rectangular_prism_operator --input 'json:{}' --input 'json:[-0.013,-0.085,-0.004]' --input 'json:[0.013,0.085,0.0]' --output-id bar_mid --no-export
op --operator rectangular_prism_operator --input 'json:{}' --input 'json:[-0.013,0.085,-0.004]' --input 'json:[0.013,0.118,0.0]' --output-id tab_a_flat --no-export
op --operator pose_operator --input asset:tab_a_flat --input 'json:{"pivot":{"at":"point","px":0.0,"py":0.085,"pz":0.0},"rotate":{"rx_deg":3.4}}' --output-id tab_a --no-export
op --operator rectangular_prism_operator --input 'json:{}' --input 'json:[-0.013,-0.118,-0.0065]' --input 'json:[0.013,-0.085,-0.0025]' --output-id tab_b_flat --no-export
op --operator pose_operator --input asset:tab_b_flat --input 'json:{"pivot":{"at":"point","px":0.0,"py":-0.088,"pz":-0.0025},"rotate":{"rx_deg":-6.7}}' --output-id tab_b --no-export
op --operator boolean_operator --input asset:bar_mid --input asset:tab_a --input asset:tab_b --input "$UNION" --output-id bar_body --no-export
SLOT_END='json:{"radius":0.005}'
HOLE8='json:{"radius":0.004}'
# Slot A centred (0.0005, 0.1046) in the bar's frame (skew removed), B at
# (-0.0030, -0.1089): 10 x 12.5, the long way along v.
op --operator cylinder_operator --input "$SLOT_END" --input 'json:[0.0005,0.10335,-0.03]' --input 'json:[0.0005,0.10335,0.03]' --output-id slot_a1 --no-export
op --operator cylinder_operator --input "$SLOT_END" --input 'json:[0.0005,0.10585,-0.03]' --input 'json:[0.0005,0.10585,0.03]' --output-id slot_a2 --no-export
op --operator rectangular_prism_operator --input 'json:{}' --input 'json:[-0.0045,0.10335,-0.03]' --input 'json:[0.0055,0.10585,0.03]' --output-id slot_a3 --no-export
op --operator cylinder_operator --input "$SLOT_END" --input 'json:[-0.003,-0.10765,-0.03]' --input 'json:[-0.003,-0.10765,0.03]' --output-id slot_b1 --no-export
op --operator cylinder_operator --input "$SLOT_END" --input 'json:[-0.003,-0.11015,-0.03]' --input 'json:[-0.003,-0.11015,0.03]' --output-id slot_b2 --no-export
op --operator rectangular_prism_operator --input 'json:{}' --input 'json:[-0.008,-0.11015,-0.03]' --input 'json:[0.002,-0.10765,0.03]' --output-id slot_b3 --no-export
op --operator cylinder_operator --input "$HOLE8" --input 'json:[-0.0012,0.0531,-0.03]' --input 'json:[-0.0012,0.0531,0.03]' --output-id bar_hole_a --no-export
op --operator cylinder_operator --input "$HOLE8" --input 'json:[-0.0025,-0.0568,-0.03]' --input 'json:[-0.0025,-0.0568,0.03]' --output-id bar_hole_b --no-export
op --operator boolean_operator --input asset:bar_body --input asset:slot_a1 --input asset:slot_a2 --input asset:slot_a3 --input asset:slot_b1 --input asset:slot_b2 --input asset:slot_b3 --input asset:bar_hole_a --input asset:bar_hole_b --input "$SUB" --output-id bar_cut --no-export
op --operator pose_operator --input asset:bar_cut --input 'json:{"rotate":{"rz_deg":-1.0},"translate":{"dx":-0.0515,"dz":0.0069}}' --output-id bracket --no-export
op --operator boolean_operator --input asset:rail --input asset:bracket --input "$UNION" --output-id mechanism_u --no-export
op --operator pose_operator --input asset:mechanism_u --input 'json:{"rotate":{"rz_deg":60.8},"translate":{"dz":0.456}}' --output-id mechanism --no-export

# --- Assembly, and the same posed into the scan's frame: arm 0 sits at
# azimuth -10.4 deg, the lift axis at (0.2569, 0.1509).
op --operator boolean_operator --input asset:column --input asset:arms --input asset:mechanism --input "$UNION" --output-id base
op --operator pose_operator --input asset:base --input 'json:{"rotate":{"rz_deg":-10.4},"translate":{"dx":0.2569,"dy":0.1509}}' --output-id base_world

$V project-run --project $P
