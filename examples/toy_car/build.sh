#!/bin/bash
# Rebuild toy_car.vproj: a hand-sized toy car from sketches, extrude/revolve,
# booleans and offset rounding (all coordinates metres; see README.md).
set -e
L="$(cd "$(dirname "$0")" && pwd)"
cd "$L/../.."
V=./target/release/volumetric_cli
P=toy_car.vproj
rm -f $P
$V project-new --output $P >/dev/null
$V project-add-asset --project $P --input $L/side_profile.lua --asset-id side_lua >/dev/null
$V project-add-asset --project $P --input $L/plan_profile.lua --asset-id plan_lua >/dev/null
$V project-add-asset --project $P --input $L/windows.lua --asset-id windows_lua >/dev/null
$V project-add-asset --project $P --input $L/wheel_profile.lua --asset-id wheel_lua >/dev/null
$V project-add-asset --project $P --input $L/grille.lua --asset-id grille_lua >/dev/null

op() { $V project-add-op --project $P "$@" >/dev/null; }
PLANE='json:{"kind":"plane"}'
UNION='json:{"op":"union"}'
SUB='json:{"op":"subtract"}'
ISECT='json:{"op":"intersect"}'

# --- Body: side profile x plan outline (two-view intersection), then rounded.
op --operator lua_script_operator --input asset:side_lua --input 'json:{}' --output-id side_sketch --no-export
op --operator subspace_operator --input "$PLANE" --input 'json:[0.0,0.0,-0.016]' --input 'json:[1.0,0.0,0.0]' --input 'json:[0.0,1.0,0.0]' --output-id side_plane --no-export
op --operator extrude_operator --input asset:side_sketch --input 'json:{"height":0.032}' --input asset:side_plane --output-id side_solid --no-export
op --operator lua_script_operator --input asset:plan_lua --input 'json:{}' --output-id plan_sketch --no-export
# Plane basis (x, z) has normal x cross z = -y: extrude downward from y = 45mm.
op --operator subspace_operator --input "$PLANE" --input 'json:[0.0,0.045,0.0]' --input 'json:[1.0,0.0,0.0]' --input 'json:[0.0,0.0,1.0]' --output-id plan_plane --no-export
op --operator extrude_operator --input asset:plan_sketch --input 'json:{"height":0.05}' --input asset:plan_plane --output-id plan_solid --no-export
op --operator boolean_operator --input asset:side_solid --input asset:plan_solid --input "$ISECT" --output-id body_sharp --no-export
op --operator offset_operator --input asset:body_sharp --input 'json:{"distance":-0.0015,"resolution":256}' --output-id body_eroded --no-export
op --operator offset_operator --input asset:body_eroded --input 'json:{"distance":0.0015,"resolution":256}' --output-id body_round --no-export

# --- Wheel wells: per-side flat cylinders r 10.5 along z at the axle lines.
WELL='json:{"radius":0.0105,"cap":"flat"}'
op --operator cylinder_operator --input "$WELL" --input 'json:[0.025,0.009,0.009]'  --input 'json:[0.025,0.009,0.020]'  --output-id well_fr --no-export
op --operator cylinder_operator --input "$WELL" --input 'json:[0.025,0.009,-0.009]' --input 'json:[0.025,0.009,-0.020]' --output-id well_fl --no-export
op --operator cylinder_operator --input "$WELL" --input 'json:[-0.025,0.009,0.009]' --input 'json:[-0.025,0.009,0.020]' --output-id well_rr --no-export
op --operator cylinder_operator --input "$WELL" --input 'json:[-0.025,0.009,-0.009]' --input 'json:[-0.025,0.009,-0.020]' --output-id well_rl --no-export
op --operator boolean_operator --input asset:well_fr --input asset:well_fl --input "$UNION" --output-id wells_f --no-export
op --operator boolean_operator --input asset:well_rr --input asset:well_rl --input "$UNION" --output-id wells_r --no-export
op --operator boolean_operator --input asset:wells_f --input asset:wells_r --input "$UNION" --output-id wells --no-export

# --- Side window recesses: sketch extruded through, minus the inner slab.
op --operator lua_script_operator --input asset:windows_lua --input 'json:{}' --output-id windows_sketch --no-export
op --operator subspace_operator --input "$PLANE" --input 'json:[0.0,0.0,-0.025]' --input 'json:[1.0,0.0,0.0]' --input 'json:[0.0,1.0,0.0]' --output-id windows_plane --no-export
op --operator extrude_operator --input asset:windows_sketch --input 'json:{"height":0.05}' --input asset:windows_plane --output-id windows_through --no-export
op --operator rectangular_prism_operator --input 'json:{}' --input 'json:[-0.05,0.0,-0.0148]' --input 'json:[0.05,0.05,0.0148]' --output-id inner_slab --no-export
op --operator boolean_operator --input asset:windows_through --input asset:inner_slab --input "$SUB" --output-id window_recess --no-export

# --- Windscreen and rear window: prisms rotated to the face rake about z,
# then moved to the face midpoint (half outside, half recessing 1.2mm).
op --operator rectangular_prism_operator --input 'json:{"mode":"position_size"}' --input 'json:[0.0,0.0,0.0]' --input 'json:[0.0024,0.011,0.024]' --output-id ws_box --no-export
op --operator rotation_operator --input asset:ws_box --input 'json:{"rz_deg":29.745}' --output-id ws_tilted --no-export
op --operator translate_operator --input asset:ws_tilted --input 'json:{"dx":0.002,"dy":0.030,"dz":0.0}' --output-id windscreen --no-export
op --operator rectangular_prism_operator --input 'json:{"mode":"position_size"}' --input 'json:[0.0,0.0,0.0]' --input 'json:[0.0024,0.009,0.024]' --output-id rw_box --no-export
op --operator rotation_operator --input asset:rw_box --input 'json:{"rz_deg":164.055}' --output-id rw_tilted --no-export
op --operator translate_operator --input asset:rw_tilted --input 'json:{"dx":-0.026,"dy":0.030,"dz":0.0}' --output-id rear_window --no-export

# --- Grille: front-view sketch on the yz plane at x = 45mm, extruded 9mm
# back into the nose (basis (z, y) has normal z cross y = -x).
op --operator lua_script_operator --input asset:grille_lua --input 'json:{}' --output-id grille_sketch --no-export
op --operator subspace_operator --input "$PLANE" --input 'json:[0.045,0.0,0.0]' --input 'json:[0.0,0.0,1.0]' --input 'json:[0.0,1.0,0.0]' --output-id grille_plane --no-export
op --operator extrude_operator --input asset:grille_sketch --input 'json:{"height":0.009}' --input asset:grille_plane --output-id grille --no-export

op --operator boolean_operator --input asset:body_round --input asset:wells --input "$SUB" --output-id body_c1 --no-export
op --operator boolean_operator --input asset:body_c1 --input asset:window_recess --input "$SUB" --output-id body_c2 --no-export
op --operator boolean_operator --input asset:body_c2 --input asset:windscreen --input "$SUB" --output-id body_c3 --no-export
op --operator boolean_operator --input asset:body_c3 --input asset:rear_window --input "$SUB" --output-id body_c4 --no-export
op --operator boolean_operator --input asset:body_c4 --input asset:grille --input "$SUB" --output-id body

# --- Axles and headlights.
AXLE='json:{"radius":0.0015,"cap":"flat"}'
op --operator cylinder_operator --input "$AXLE" --input 'json:[0.025,0.009,-0.0175]'  --input 'json:[0.025,0.009,0.0175]'  --output-id axle_f --no-export
op --operator cylinder_operator --input "$AXLE" --input 'json:[-0.025,0.009,-0.0175]' --input 'json:[-0.025,0.009,0.0175]' --output-id axle_r --no-export
LAMP='json:{"radius":0.0025,"cap":"round"}'
op --operator cylinder_operator --input "$LAMP" --input 'json:[0.036,0.017,0.009]'  --input 'json:[0.0395,0.017,0.009]'  --output-id lamp_r --no-export
op --operator cylinder_operator --input "$LAMP" --input 'json:[0.036,0.017,-0.009]' --input 'json:[0.0395,0.017,-0.009]' --output-id lamp_l --no-export

# --- Wheels: revolve about z at the origin (outer face +z), mirror for the
# left side, translate to the four hubs.
op --operator lua_script_operator --input asset:wheel_lua --input 'json:{}' --output-id wheel_sketch --no-export
op --operator revolve_operator --input asset:wheel_sketch --input 'data:' --output-id wheel
op --operator scale_operator --input asset:wheel --input 'json:{"sz":-1.0}' --output-id wheel_mirrored --no-export
op --operator translate_operator --input asset:wheel --input 'json:{"dx":0.025,"dy":0.009,"dz":0.014}' --output-id wheel_fr --no-export
op --operator translate_operator --input asset:wheel --input 'json:{"dx":-0.025,"dy":0.009,"dz":0.014}' --output-id wheel_rr --no-export
op --operator translate_operator --input asset:wheel_mirrored --input 'json:{"dx":0.025,"dy":0.009,"dz":-0.014}' --output-id wheel_fl --no-export
op --operator translate_operator --input asset:wheel_mirrored --input 'json:{"dx":-0.025,"dy":0.009,"dz":-0.014}' --output-id wheel_rl --no-export

# --- Assembly.
op --operator boolean_operator --input asset:body --input asset:axle_f --input "$UNION" --output-id a1 --no-export
op --operator boolean_operator --input asset:a1 --input asset:axle_r --input "$UNION" --output-id a2 --no-export
op --operator boolean_operator --input asset:a2 --input asset:lamp_r --input "$UNION" --output-id a3 --no-export
op --operator boolean_operator --input asset:a3 --input asset:lamp_l --input "$UNION" --output-id a4 --no-export
op --operator boolean_operator --input asset:a4 --input asset:wheel_fr --input "$UNION" --output-id a5 --no-export
op --operator boolean_operator --input asset:a5 --input asset:wheel_rr --input "$UNION" --output-id a6 --no-export
op --operator boolean_operator --input asset:a6 --input asset:wheel_fl --input "$UNION" --output-id a7 --no-export
op --operator boolean_operator --input asset:a7 --input asset:wheel_rl --input "$UNION" --output-id car

$V project-run --project $P
