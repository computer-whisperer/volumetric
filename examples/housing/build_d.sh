#!/bin/bash
# Family D: Lua-scripted shell and parting solids — 11 nodes instead of ~50.
set -e
L="$(cd "$(dirname "$0")" && pwd)"
cd "$L/../.."
V=./target/release/volumetric_cli
P=housing_d.vproj
rm -f $P
$V project-new --output $P >/dev/null
$V project-add-asset --project $P --input board.step --asset-id board_step >/dev/null
$V project-add-asset --project $P --input $L/outer_ports.lua --type lua --asset-id outer_lua >/dev/null
$V project-add-asset --project $P --input $L/p_tray.lua --type lua --asset-id ptray_lua >/dev/null
$V project-add-asset --project $P --input $L/p_lid.lua --type lua --asset-id plid_lua >/dev/null

op() { $V project-add-op --project $P "$@" >/dev/null; }

op --operator step_import_operator --input asset:board_step --input 'json:{}' --output-id board
op --operator offset_operator --input asset:board --input 'json:{"distance":0.0004,"resolution":256}' --output-id cavity --no-export
op --operator sweep_operator --input asset:cavity --input 'json:{"axis":"z","distance":0.02,"until":0.0058,"resolution":256}' --output-id swept --no-export
op --operator lua_script_operator --input asset:outer_lua --input 'json:{}' --output-id shell --no-export
op --operator lua_script_operator --input asset:ptray_lua --input 'json:{}' --output-id p_tray --no-export
op --operator lua_script_operator --input asset:plid_lua --input 'json:{}' --output-id p_lid --no-export
op --operator boolean_operator --input asset:shell --input asset:swept --input 'json:{"op":"subtract"}' --output-id housing
op --operator boolean_operator --input asset:housing --input asset:p_tray --input 'json:{"op":"intersect"}' --output-id tray
op --operator boolean_operator --input asset:housing --input asset:p_lid --input 'json:{"op":"subtract"}' --output-id lid

$V project-run --project $P
