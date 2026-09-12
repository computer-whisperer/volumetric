# Chair base continuation

User accepted the upper interface on 2026-09-12 and requested the remaining
base at representative accuracy, with special care for the oblong backrest
receiver at the rear end of the T and its tilt connection.

Keep the six accepted mounting apertures and their measured frame fixed.
Continue with the 44-photo native survey; no prior chair model or scanner
reconstruction. Add independently replayable observations for the receiver,
its visible pivots, gas lift, star base, casters, and controls. Represent
unobserved internals and the mechanism's motion limits as unresolved.

WGSL remains the only new modelling language. Use separately exported parts
and explicit measured versus assumed parameters. Validate the receiver in
several top/oblique views and the assembly in side and whole-base views.
The receiver is a hollow socket, not a shallow hole in the existing rail.
Its mating opening needs its own plane, profile, and insertion direction.


## Completed milestone

The 16-part WGSL base, receiver fit, additional observations, replay scripts,
and native occupancy audit are implemented. See README.md for results and
limitations. Accepted mounting parameters remain unchanged; continuation
fit results and parameters reproduce in work/replay. The receiver entry
mouth is about 39 by 18 mm, with coherent cross-view disagreement at the
1–2 mm level. Actual throat clearance, insertion direction/depth, and
kinematics are unresolved. Keep these distinctions when designing a mate.

CLI measurement coordinates now remain f64, and the multiple-view-set error
explains the render-specific selector. The WGSL guide records the select
literal-concretization trap. CLI 29 tests pass; host full workspace tests
fail only the pre-existing operator_metadata strict-bound parser test.

A suspected missing longitudinal column surface was withdrawn after four
isolated render comparisons and independent image review. Continuous
silhouettes in all modes support shading/faceting differences, not the
claimed missing geometry. Final audit renders use denser meshes with
sharpening and simplification disabled; this is a presentation choice,
not a demonstrated engine bug fix.

Cleanup: shared project-add-operator orchestration between the two build
scripts; no change to deprecated Lua. Unrelated toy_car edits, prints,
and screen captures were present before this arc and remain untouched.
