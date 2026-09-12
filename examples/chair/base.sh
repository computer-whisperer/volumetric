#!/bin/bash
# Rebuild chair_base.vproj: the office-chair base (five-arm star, gas lift,
# tilt mechanism) from the chairbase-dslr-1 scan's measurements (PLAN.md).
# Parts are built in the base frame (origin on the floor under the lift
# axis, z up, arm 0 along +x), then each is posed into the survey's card
# frame and the articulated assembly is made there: `chair` (the Assembly:
# lift, swivel, and a swivel and a roll per caster) and `chair_model` (the
# parts posed at the rest state as one model, for verify.sh and audit.sh).
# All coordinates metres.
set -e
L="$(cd "$(dirname "$0")" && pwd)"
cd "$L/../.."
V=./target/release/volumetric_cli
P=chair_base.vproj
rm -f $P
$V project-new --output $P >/dev/null

# The evidence's photographs (evidence.sh writes the selection): ten
# surveyed views of the base, so the model can be looked through them
# here (`render -i chair_base.vproj --asset chair_model --through DSC00742
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

# --- Hub and gas lift: two sections revolved about z (sketch x = radius,
# sketch y = height). The hub and outer tube: hub r 54 from z 133 to 170
# with a chamfer to the collar at 178; collar r 30 to 190; lift outer tube
# r 25.5 to 265. The piston (the lift's moving part): r 15 from inside the
# tube at 250 to the mechanism at 410.
op --operator path_sketch_operator --input 'json:{"path":"M 0 0.133 H 0.054 V 0.170 L 0.048 0.178 H 0.030 V 0.190 H 0.0255 V 0.265 H 0 Z"}' --output-id hub_tube_sketch --no-export
op --operator revolve_operator --input asset:hub_tube_sketch --input none --output-id hub_tube_base --no-export
op --operator path_sketch_operator --input 'json:{"path":"M 0 0.250 H 0.015 V 0.410 H 0 Z"}' --output-id piston_sketch --no-export
op --operator revolve_operator --input asset:piston_sketch --input none --output-id piston_base --no-export

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
# arbitrary; the cloud's wheels sit at r 360). Built once for arm 0; each
# arm's copy is posed into the world below, a part of its own so it can
# swivel and roll.
op --operator cylinder_operator --input 'json:{"radius":0.011}' --input 'json:[0.34,0.0,0.03]' --input 'json:[0.34,0.0,0.085]' --output-id stem_base --no-export
op --operator cylinder_operator --input 'json:{"radius":0.0325}' --input 'json:[0.36,-0.012,0.0325]' --input 'json:[0.36,0.012,0.0325]' --output-id wheel_base --no-export
op --operator pattern_operator --input asset:arm_round --input 'json:{"circular":{"count":5,"axis":"z"}}' --output-id arms_base --no-export

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
# Posed into the world below: 60.8 deg in the base frame is 50.4 deg in
# the survey's, 0.456 up the lift axis.
WORLD_MECH='json:{"rotate":{"rz_deg":50.4},"translate":{"dx":0.2569,"dy":0.1509,"dz":0.456}}'

# --- Into the scan's frame: arm 0 sits at azimuth -10.4 deg, the lift
# axis at (0.2569, 0.1509). Every part is posed there at the rest state
# (the state as photographed), and the casters are five copies each of
# the stem and the wheel, one per arm.
AX=0.2569
AY=0.1509
WORLD='json:{"rotate":{"rz_deg":-10.4},"translate":{"dx":0.2569,"dy":0.1509}}'
op --operator pose_operator --input asset:arms_base --input "$WORLD" --output-id arms --no-export
op --operator pose_operator --input asset:hub_tube_base --input "$WORLD" --output-id hub_tube --no-export
op --operator pose_operator --input asset:piston_base --input "$WORLD" --output-id piston --no-export
op --operator pose_operator --input asset:mechanism_u --input "$WORLD_MECH" --output-id mechanism --no-export
PARTS="--input asset:arms --input asset:hub_tube --input asset:piston --input asset:mechanism"
for k in 0 1 2 3 4; do
    AZ=$(python3 -c "print(72 * $k - 10.4)")
    CASTER="json:{\"rotate\":{\"rz_deg\":$AZ},\"translate\":{\"dx\":$AX,\"dy\":$AY}}"
    op --operator pose_operator --input asset:stem_base --input "$CASTER" --output-id stem_$k --no-export
    op --operator pose_operator --input asset:wheel_base --input "$CASTER" --output-id wheel_$k --no-export
    PARTS="$PARTS --input asset:stem_$k --input asset:wheel_$k"
done

# --- The mechanism: the star and hub fixed to the floor, the piston
# sliding along the lift axis (the scan's height is the rest, 0; the lift
# is assumed to have 20 mm below it and 80 above), the tilt mechanism
# swivelling on the piston about the same axis, and each caster swivelling
# about its stem and rolling about its axle. Axes in world coordinates at
# rest: the stem of arm k at r 340 and the axle at r 360, z 32.5, across
# the arm.
MECH=$(python3 - <<PY
import json, math
ax, ay = $AX, $AY
z = [0.0, 0.0, 1.0]
parts = ["arms", "hub_tube", "piston", "mechanism"]
joints = [
    {"name": "ground", "child": "arms"},
    {"name": "hub", "parent": "arms", "child": "hub_tube"},
    {"name": "lift", "kind": "prismatic", "parent": "hub_tube", "child": "piston",
     "axis": {"origin": [ax, ay, 0.0], "direction": z}, "min": -0.02, "max": 0.08},
    {"name": "swivel", "kind": "revolute", "parent": "piston", "child": "mechanism",
     "axis": {"origin": [ax, ay, 0.0], "direction": z}, "min": -180.0, "max": 180.0},
]
for k in range(5):
    t = math.radians(72 * k - 10.4)
    c, s = math.cos(t), math.sin(t)
    parts += [f"stem_{k}", f"wheel_{k}"]
    joints.append({"name": f"caster_{k}", "kind": "revolute", "parent": "arms", "child": f"stem_{k}",
                   "axis": {"origin": [ax + 0.34 * c, ay + 0.34 * s, 0.0], "direction": z},
                   "min": -180.0, "max": 180.0})
    joints.append({"name": f"roll_{k}", "kind": "revolute", "parent": f"stem_{k}", "child": f"wheel_{k}",
                   "axis": {"origin": [ax + 0.36 * c, ay + 0.36 * s, 0.0325], "direction": [-s, c, 0.0]},
                   "min": -180.0, "max": 180.0})
print(json.dumps({"parts": parts, "joints": joints}))
PY
)
op --operator mechanism_operator --input "json:$MECH" --input none --output-id chair_mechanism --no-export
op --operator assemble_operator --input asset:chair_mechanism $PARTS --input 'json:{}' --output-id chair

$V project-run --project $P
