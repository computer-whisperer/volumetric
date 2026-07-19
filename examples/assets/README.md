# Example reference assets

## Raspberry Pi 4 Model B

`raspberry_pi_4_model_b.stl` is a reference assembly from
[`atticusrussell/ballbot`](https://github.com/atticusrussell/ballbot), commit
`9998f0ff833a640d9850c5964aaeb6005cea86ab`, originally stored at
[`src/ballbot_description/meshes/raspberry_pi_4_model_b.stl`](https://github.com/atticusrussell/ballbot/blob/9998f0ff833a640d9850c5964aaeb6005cea86ab/src/ballbot_description/meshes/raspberry_pi_4_model_b.stl)
and used by that project without an additional URDF scale factor.

- Upstream license: Apache License 2.0; see `LICENSE.ballbot-Apache-2.0.txt`.
- SHA-256: `9ed7722a2c885a4046ef8909694fe9b257e9994a77853bfff04e061a1daea3a3`.
- Imported bounds: `(-0.043, -0.003, -0.028)` to `(0.046, 0.017, 0.030)` m.
- Alignment used by `raspberry_pi_4_fit.vproj`: rotate +90° about X, then
  translate by `(0.0425, 0.028, 0.0066)` m. This maps the centred 85 × 56 mm
  PCB into the tray's `[0, 0.085] × [0, 0.056]` board coordinates with its
  underside on the 5 mm standoffs.

This is a third-party visual/mechanical reference, not an official Raspberry
Pi CAD release. The board envelope and mounting pattern should still be checked
against [Raspberry Pi's official mechanical drawing](https://pip-assets.raspberrypi.com/categories/545-raspberry-pi-4-model-b/documents/RP-008343-DS-1-raspberry-pi-4-mechanical-drawing.pdf)
before manufacturing. The project's `fit_interference` export is the boolean
intersection of tray and reference; at 192³ with 16 discovery probes it meshes
to zero triangles.
