# Debugger-board housing (board.step)

Clamshell housing for the multi-probe debugger assembly at the repo
root (`board.step`): ten side-entry JST-SH probe connectors on three
edges, USB-C on the fourth. Each script rebuilds its `housing_<x>.vproj`
at the repo root from scratch (`cargo build --release -p volumetric_cli`
first). The cavity pipeline is shared: STEP import -> offset 0.4mm
(clearance dilate) -> sweep +z to the interior ceiling (demolding-
monotone, so socket interiors and component gaps stay open). All
variants split at z=3.1mm into a tray and lid with a 0.2mm-clearance
lip/rebate, and open the USB port as a stadium profile hugging the
connector with the top edge flush at the connector top.

- `build_a.sh` — open gallery: JST slots are open-top U-notches, roof
  pulled back past the inner wall face; snap-nub fastening.
- `build_b.sh` — enclosed JST slots behind a 45-degree flared eave
  (rotated-prism wedge cuts); snap-nub fastening.
- `build_c.sh` — family A plus corner turrets with M2 screw stacks
  (counterbore from below, pilot into the lid posts); no snap nubs.
- `build_d.sh` — family A's geometry with the shell and both parting
  solids authored as Lua scripts (`outer_ports.lua`, `p_tray.lua`,
  `p_lid.lua`) instead of primitive booleans: ~11 nodes vs ~50, and
  the corner rounding is analytic instead of a lattice bake.

Note: `json:` configs on the CLI must use float literals for float
fields (`-45.0`, not `-45`) — integers encode as CBOR ints and the
operators reject them.
