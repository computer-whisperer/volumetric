#!/bin/bash
# Rebuild housing.vproj: clamshell for board.step (all coordinates metres).
# Requires: board.step at repo root, CLI built with step/offset/sweep/cylinder.
set -e
cd "$(dirname "$0")/../.."
V=./target/release/volumetric_cli
P=housing_a.vproj
rm -f $P
$V project-new --output $P >/dev/null
$V project-add-asset --project $P --input board.step --asset-id board_step >/dev/null

op() { $V project-add-op --project $P "$@" >/dev/null; }

# Board -> clearance dilate -> demolding sweep (cavity, monotone in +z).
op --operator step_import_operator --input asset:board_step --input 'json:{}' --output-id board
op --operator offset_operator --input asset:board --input 'json:{"distance":0.0004,"resolution":256}' --output-id cavity --no-export
op --operator sweep_operator --input asset:cavity --input 'json:{"axis":"z","distance":0.02,"until":0.0058,"resolution":256}' --output-id swept --no-export

# Outer shell: inset core dilated 2mm = rounded verticals + rounded top rim
# (flat crown above the core footprint); slab intersect flattens the bottom.
op --operator rectangular_prism_operator --input 'json:{}' --input 'json:[-0.0175,-0.0215,-0.0025]' --input 'json:[0.0175,0.0134,0.0058]' --output-id core --no-export
op --operator offset_operator --input asset:core --input 'json:{"distance":0.002,"resolution":224}' --output-id outer_round --no-export
op --operator rectangular_prism_operator --input 'json:{}' --input 'json:[-0.0195,-0.0235,-0.0025]' --input 'json:[0.0195,0.0154,0.02]' --output-id slab --no-export
op --operator boolean_operator --input asset:outer_round --input asset:slab --input 'json:{"op":"intersect"}' --output-id outer --no-export

# Cable slots through the walls (gallery E/S/W at the JST band + USB north).
op --operator rectangular_prism_operator --input 'json:{}' --input 'json:[0.0155,-0.0165,0.0013]' --input 'json:[0.0210,0.0115,0.02]' --output-id slot_e --no-export
op --operator rectangular_prism_operator --input 'json:{}' --input 'json:[-0.0110,-0.0240,0.0013]' --input 'json:[0.0100,-0.0195,0.02]' --output-id slot_s --no-export
op --operator rectangular_prism_operator --input 'json:{}' --input 'json:[-0.0210,-0.0145,0.0013]' --input 'json:[-0.0155,0.0065,0.02]' --output-id slot_w --no-export
# USB mouth sized for the USB-IF maximum plug overmold (12.35 x 6.5mm)
# plus ~0.3mm clearance: 13.0mm wide, open through the top, floor at
# z=0.15mm with r1.0 bottom corners; the port is recessed ~2.6mm so the
# whole wall opening must pass the overmold envelope.
op --operator rectangular_prism_operator --input 'json:{}' --input 'json:[-0.0065,0.0123,0.00115]' --input 'json:[0.0065,0.0170,0.02]' --output-id usb_main --no-export
op --operator rectangular_prism_operator --input 'json:{}' --input 'json:[-0.0055,0.0123,0.00015]' --input 'json:[0.0055,0.0170,0.02]' --output-id usb_low --no-export
op --operator cylinder_operator --input 'json:{"radius":0.001,"cap":"flat"}' --input 'json:[-0.0055,0.0123,0.00115]' --input 'json:[-0.0055,0.0170,0.00115]' --output-id usb_r1 --no-export
op --operator cylinder_operator --input 'json:{"radius":0.001,"cap":"flat"}' --input 'json:[0.0055,0.0123,0.00115]' --input 'json:[0.0055,0.0170,0.00115]' --output-id usb_r2 --no-export
op --operator boolean_operator --input asset:usb_main --input asset:usb_low --input 'json:{"op":"union"}' --output-id usb_s1 --no-export
op --operator boolean_operator --input asset:usb_s1 --input asset:usb_r1 --input 'json:{"op":"union"}' --output-id usb_s2 --no-export
op --operator boolean_operator --input asset:usb_s2 --input asset:usb_r2 --input 'json:{"op":"union"}' --output-id usb_mouth --no-export
op --operator rectangular_prism_operator --input 'json:{}' --input 'json:[-0.0070,0.0123,0.0053]' --input 'json:[0.0070,0.0170,0.02]' --output-id usb_scoop --no-export
op --operator boolean_operator --input asset:usb_mouth --input asset:usb_scoop --input 'json:{"op":"union"}' --output-id slot_usb --no-export
op --operator boolean_operator --input asset:swept --input asset:slot_e --input 'json:{"op":"union"}' --output-id cut1 --no-export
op --operator boolean_operator --input asset:cut1 --input asset:slot_s --input 'json:{"op":"union"}' --output-id cut2 --no-export
op --operator boolean_operator --input asset:cut2 --input asset:slot_w --input 'json:{"op":"union"}' --output-id cut3 --no-export
op --operator boolean_operator --input asset:cut3 --input asset:slot_usb --input 'json:{"op":"union"}' --output-id cuts --no-export
op --operator boolean_operator --input asset:outer --input asset:cuts --input 'json:{"op":"subtract"}' --output-id housing

# Stepped parting: tray keeps below z=3.1mm plus a wall lip (3.1..4.3,
# inner face 17.5 to wall midline 18.55); the lid is cut with a rebate
# ring 0.2mm larger on every face for fit clearance.
op --operator rectangular_prism_operator --input 'json:{}' --input 'json:[-0.03,-0.03,-0.01]' --input 'json:[0.03,0.03,0.0031]' --output-id box_lo --no-export
op --operator rectangular_prism_operator --input 'json:{}' --input 'json:[-0.01855,-0.0225,0.0031]' --input 'json:[0.01855,0.01432,0.0043]' --output-id lip_mid --no-export
op --operator rectangular_prism_operator --input 'json:{}' --input 'json:[-0.0175,-0.0214,0]' --input 'json:[0.0175,0.01325,0.02]' --output-id lip_hole --no-export
op --operator boolean_operator --input asset:lip_mid --input asset:lip_hole --input 'json:{"op":"subtract"}' --output-id lip_ring --no-export
op --operator boolean_operator --input asset:box_lo --input asset:lip_ring --input 'json:{"op":"union"}' --output-id p_tray --no-export
op --operator rectangular_prism_operator --input 'json:{}' --input 'json:[-0.01875,-0.0227,0.0031]' --input 'json:[0.01875,0.01452,0.0045]' --output-id reb_mid --no-export
op --operator rectangular_prism_operator --input 'json:{}' --input 'json:[-0.0173,-0.0212,0]' --input 'json:[0.0173,0.01305,0.02]' --output-id reb_hole --no-export
op --operator boolean_operator --input asset:reb_mid --input asset:reb_hole --input 'json:{"op":"subtract"}' --output-id reb_ring --no-export
op --operator boolean_operator --input asset:box_lo --input asset:reb_ring --input 'json:{"op":"union"}' --output-id p_lid --no-export
op --operator boolean_operator --input asset:housing --input asset:p_tray --input 'json:{"op":"intersect"}' --output-id tray0 --no-export
op --operator boolean_operator --input asset:housing --input asset:p_lid --input 'json:{"op":"subtract"}' --output-id lid0 --no-export

# Snap fastening: capsule nubs on the lip outer face (half-proud), matching
# grooves in the lid rebate wall (r +0.15, ends +0.2). Two on the south lip
# segments, two on the north segments beside the USB opening.
CAP='json:{"radius":0.0005,"cap":"round"}'
GRV='json:{"radius":0.00065,"cap":"round"}'
op --operator cylinder_operator --input "$CAP" --input 'json:[-0.0155,-0.0225,0.0038]' --input 'json:[-0.0125,-0.0225,0.0038]' --output-id nub_s1 --no-export
op --operator cylinder_operator --input "$CAP" --input 'json:[0.0125,-0.0225,0.0038]' --input 'json:[0.0155,-0.0225,0.0038]' --output-id nub_s2 --no-export
op --operator cylinder_operator --input "$CAP" --input 'json:[-0.0135,0.01432,0.0038]' --input 'json:[-0.0105,0.01432,0.0038]' --output-id nub_n1 --no-export
op --operator cylinder_operator --input "$CAP" --input 'json:[0.0105,0.01432,0.0038]' --input 'json:[0.0135,0.01432,0.0038]' --output-id nub_n2 --no-export
op --operator cylinder_operator --input "$GRV" --input 'json:[-0.0157,-0.0225,0.0038]' --input 'json:[-0.0123,-0.0225,0.0038]' --output-id grv_s1 --no-export
op --operator cylinder_operator --input "$GRV" --input 'json:[0.0123,-0.0225,0.0038]' --input 'json:[0.0157,-0.0225,0.0038]' --output-id grv_s2 --no-export
op --operator cylinder_operator --input "$GRV" --input 'json:[-0.0137,0.01432,0.0038]' --input 'json:[-0.0103,0.01432,0.0038]' --output-id grv_n1 --no-export
op --operator cylinder_operator --input "$GRV" --input 'json:[0.0103,0.01432,0.0038]' --input 'json:[0.0137,0.01432,0.0038]' --output-id grv_n2 --no-export
op --operator boolean_operator --input asset:tray0 --input asset:nub_s1 --input 'json:{"op":"union"}' --output-id tray1 --no-export
op --operator boolean_operator --input asset:tray1 --input asset:nub_s2 --input 'json:{"op":"union"}' --output-id tray2 --no-export
op --operator boolean_operator --input asset:tray2 --input asset:nub_n1 --input 'json:{"op":"union"}' --output-id tray3 --no-export
op --operator boolean_operator --input asset:tray3 --input asset:nub_n2 --input 'json:{"op":"union"}' --output-id tray
op --operator boolean_operator --input asset:grv_s1 --input asset:grv_s2 --input 'json:{"op":"union"}' --output-id grv12 --no-export
op --operator boolean_operator --input asset:grv12 --input asset:grv_n1 --input 'json:{"op":"union"}' --output-id grv123 --no-export
op --operator boolean_operator --input asset:grv123 --input asset:grv_n2 --input 'json:{"op":"union"}' --output-id grooves --no-export
op --operator boolean_operator --input asset:lid0 --input asset:grooves --input 'json:{"op":"subtract"}' --output-id lid

$V project-run --project $P
