# Chair (dogfooding from evidence)

An office chair built on the scanned base of a real one. The base comes
first: its model is what everything else mounts on, and modelling it from
photographs is the forcing function for volumetric's evidence tooling.
`PLAN.md` is the ledger: steps, measurements, the gaps met and how each
was fixed.

Scripts (run from the repo root after `cargo build --release -p
volumetric_cli`; the scanner tree is expected at `$SCAN`, default
`/ceph/christian/index_scanner`):

- `evidence.sh [out.vproj]` surveys the chairbase-dslr-1 stills here and
  imports the trained splat (as surfels) and its TSDF surface cloud:
  `chair_evidence.vproj`, about 120 MB, in the session's demo directory.
- `base.sh` rebuilds `chair_base.vproj` at the repo root from the catalog
  (revolved column, two-view arms patterned five times, box mechanism) and
  runs it. Exports `base` (base frame: floor origin under the lift axis,
  z up, one arm along +x) and `base_world` (posed into the survey's card
  frame).
- `verify.sh [evidence.vproj]` draws `base_world` through three surveyed
  photographs as an edge overlay, and in plan and elevation sections over
  the scan's cloud, under `target/chair/`.
