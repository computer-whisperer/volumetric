# volumetric (Python)

volumetric's kernels from Python, one to one: projects (build, run,
read the exports as arrays), view sets and splats, marker detection, the
still solve and the survey. A PyO3 extension module built with maturin;
the package is `volumetric`, the module `volumetric._volumetric`.

The CLI stays the canonical agent surface and the operator the durable
pipeline stage; this is the workbench for validation loops and plots,
and the way the scanner's Python can call volumetric's kernels instead of
its own. Anything the CLI does through a library call the bindings do
through the same call (`volumetric::project_edit` holds the shared
project-editing rules), so a feature that exists only here is a bug.

## Build

```sh
source path/to/venv/bin/activate      # Python >= 3.12; numpy
pip install maturin
maturin develop --release -m crates/volumetric_py/Cargo.toml
python -m pytest crates/volumetric_py/tests
```

The crate is a workspace member, so `cargo build --workspace` and clippy
cover it (a Python interpreter is needed at build time for pyo3's
configuration). Its Rust tests are disabled (`[lib] test = false`): an
extension module has no libpython to link a test binary against, and the
tests are pytest. Tests needing the chair evidence skip when
`$SCAN/sessions/chairbase-dslr-1-all/demo/chair_views.vviews` is absent
(`SCAN` defaults to `/ceph/christian/index_scanner`).

## Projects

```python
import volumetric as v

p = v.Project()
p.add_op("cylinder_operator", [{"radius": 0.05}, [0, 0, 0], [0, 0, 0.2]], output="post")
p.add_op("cylinder_operator", [{"radius": 0.05}, [0.1, 0, 0], [0.1, 0, 0.2]], output="post2", export=False)
p.add_op("boolean_operator", ["post", "post2", {"op": "union"}], output="both")
out = p.run()                       # {id: Asset}, the exports
mesh = out["both"].mesh(max_depth=4)  # a Model meshed: nodes (n,3), elements (m,3), node_fields["normal"]
p.save("posts.vproj")
```

- `Project()`, `Project.open(path)`, `Project.from_bytes(b)`, `.save(path)`,
  `.to_bytes()`, `.asset_ids()`, `.exports`, `.steps()`, `.validate()`.
- `.add_model(wasm, id=None)`; `.add_asset(data, kind, id=None)` with kind
  `lua`, `wgsl`, `config`, `f64map` (a dict is accepted), `blob`, `viewset`,
  `splat`, `mechanism` or `assembly`; `v.models()` / `v.model_bytes(name)` for the bundled models.
- `.add_op(operator, inputs, output=None, export=True)`: `operator` is a
  bundled name (`v.operators()`), a `.wasm` path or wasm bytes. Each input
  is an asset id (`str`), `None` (unwired), `bytes` (raw) or a JSON-like
  value coerced by the slot's declared type: a dict for a CBOR
  configuration or an F64Map, a list for a VecF64, as the CLI's `json:`
  form. Errors name the slot. `v.operator_info(name)` shows the slots
  (with the configuration's CDDL), outputs and docs.
- `.add_views(viewset, id="views")` imports a view set for look-through
  and `render(through=)`; `.set_config(step, {field: value})` merges
  schema-checked values into a step's configuration (`step` is an index or
  an operator-id substring) and returns what changed.
- `.run(remote=None)` runs every step (on a daemon at `http://host:port`
  when `remote` is given) and returns the exports. `Asset.kind`,
  `.bytes`, `.warnings`; `.value` decodes an F64Map (dict), a VecF64
  (float array) or a Subspace (dict with `origin` and `basis` arrays);
  `.mesh(...)` gives a FeaMesh or TriMesh as arrays, or meshes a Model
  with the adaptive surface nets (`base_resolution`, `max_depth`,
  `sharp_edges`, `sharp_angle`, `simplify`); `.viewset()`, `.splat()`,
  `.mechanism()` and `.assembly()` decode those kinds.

## Measuring in photographs

```python
view = views.view("DSC00755")
px = view.project(points)                  # (n,3) world -> (n,2) pixels through the distortion; NaN behind
world, depth = view.cast(px, z=0.0)        # pixels onto a plane: z=height or plane=(point, normal)
point, gaps = views.triangulate({"DSC00755": (3728, 919), "DSC00758": (2513, 2011)})  # metres, ray misses
values = asset.sample(points); inside = asset.occupied(points); lo, hi = asset.bounds()  # a Model, in process
```

Picks live with the photograph they were made in, not in a script's
JSON: `views.with_picks({name: {view: (u, v)}}, check={...})` records
them on the views (`picks()` reads them back, `with_contours` /
`contours()` for traced rims), and `views.fit_picks()` triangulates every
recorded feature, reporting each fit pick's ray miss and each check
pick's reprojection error. Picks survive `encode`, `select` and
`detect`; the CLI records them with `view-pick --record NAME [--check]`
and fits them with `view-triangulate --all-features`.

These are the calls behind `view-pick`, `view-triangulate` and `sample`
(`view_core::measure`), so a script that shelled out per feature can loop
in Python instead. `examples/chair_photo`'s committed measurement replays
through them to the same points, ray misses and check-view errors.

## Frames

```python
r = v.render(p, views="iso,top", width=1024, height=768)          # the project is run; r.frames (h,w,4) per preset
r = v.render(out_assets, camera=([1, 1, 1], [0, 0, 0]))            # assets from a run, an explicit eye and target
r = v.render(p, through="views:DSC00742", overlay="edge")          # a surveyed photograph, the render composited
r.image, r.names, r.report                                          # first frame, preset names, stats / up / GPU / notes
```

The same call as the CLI's `render` (`volumetric_render`): presets, an
explicit camera, a pinhole (`pinhole={"fx", "fy", "cx", "cy",
"camera_to_world"}`), or a view of a view set with its photograph under
an overlay; `resolution`, `sharp`, `simplify`, `color_field`,
`color_range`, `wireframe`, `background`, `up`, `projection="ortho"`
with `ortho_scale`, `grid`, `ssao` as the CLI's flags; `marks=True`
draws what the looked-through view observed (markers, card corners,
recorded picks and contours, named beside their marks) over the frame,
as the GUI does. Needs a GPU (or a
software adapter wgpu accepts).

## Pictures and markers

```python
gray = v.Gray.open("DSC00742.JPG")            # or Gray(array), Gray.from_rgb(array), Gray.decode(bytes)
found = v.detect(gray, ["5x5_100", "36h11"])   # [{id, family, corners, rotation, distance, fit_px}]
obs = v.observe(gray)                          # swatches + survey card: detections, corners, blur, observations
solve = v.solve_still(gray, views, file=open("DSC00742.JPG", "rb").read())  # pose against a view set's field
solved, report = v.survey(views, rounds=6)     # the survey; report is SurveyReport as a dict
```

The survey from a directory of stills:

```python
views, report = v.import_stills("/photos", embed="preview", labels={"session": "chair"})  # StillsOptions kwargs
views, reports, skipped = views.detect(swatches="5x5_100", card="card.json")            # observations on the views
solved, report = v.survey(views)
```

`ViewSet.detect` stores each picture's markers, card corners and blur on
its view (`view.observations`) and the card's spec on the set; `card`
takes the survey card by default, a spec dict, JSON text, a card.json
path or `False`. `card_spec(...)` shows what a card argument parses to.

Options are keyword arguments over the crate's defaults (`DetectParams`
for `detect`, `SurveyOptions` for `survey`, `Render` for the renderers);
an unknown name is an error. `observe` takes `detect=` and `corners=`
dicts for its two parameter blocks, `board=False` to skip the card.

`render_board(camera, camera_to_world, origin, right, down, spec=None)`
and `render_markers(camera, camera_to_world, markers, family)` are
`cv_core`'s synthetic renderers, for tests that need a picture with a
known answer; `survey_card()` and `board_corners()` describe the card.

## View sets and splats

`ViewSet.load(path)` / `.decode(bytes)` / `.save` / `.encode`; `.views`
(each a `View`: `id`, `camera`, `camera_to_world` (3,4) or None,
`position`, `tags`, `shot`, `observations`, `picture()` bytes, `image()`
(h,w,3), `depth()`, `mask()`, `as_dict()`), `.view(id)`, `.cameras`
(`Camera`: `width height fx fy cx cy distortion`, constructible),
`.poses()` (n,3,4) with NaN for unposed, `.markers` (dicts),
`.marker_ids()`, `.marker_corners()` (n,4,3), `.board`, `.world_up`,
`.provenance`.

`ViewSet.select(ids=, stride=, near=, radius=, max=, posed=, tags=,
embed=, preview_px=)` is `view-select`: a subset as a set of its own with
its pictures re-embedded (`keep`, `full`, `preview`, `none`).
`View.crop(center, size=, scale=, grid=, marks=, world_marks=)` is
`view-crop`: a magnified crop of the original with a labelled grid and
crosses, as a `Crop` with `image` (h,w,3), `origin`/`end`, the grid
lines' coordinates and each world mark's `projected` pixel; `.png()` and
`.save(path)`.

`Splat.load(path)` / `.decode(bytes)`: `means`, `scales`, `quats`,
`opacities`, `sh0`, `sh_rest` (n,3,k), `normals` or None, `kind`,
`sh_degree`, `count`.

Poses are camera-to-world 3x4 row-major: columns are the camera's right,
down and forward axes in the world and its position.

## Mechanisms and assemblies

An articulated assembly (`ASSEMBLY_PLAN.md`): parts joined by fixed,
revolute and prismatic joints, authored in the world frame at rest with
axes in world coordinates at rest, posed by the product of each chain's
joint motions. `Mechanism.load(path)` / `.decode(bytes)` / `.save` /
`.encode`; `.parts`, `.joints` (dicts), `.state_keys`, `.default_state`,
`.ranges`; `.pose(state=None)` gives every part's world <- part map as an
(n,3,4) array (missing keys take their defaults, out-of-range values are
refused), `.velocity(part, point, state=None)` the world velocity of a
point on a part per unit rate of each state as a (k,3) array (the
Jacobian a drag solves against), `.pull(part, local, target, state=None,
iterations=8)` the state that brings a point of a part nearest a world
target (what the viewport solves while a part is dragged). `Assembly.load` / `.decode` / `.save` /
`.encode`; `.mechanism`, `.parts`, `.part_model(name)` (the unposed wasm),
`.state`, `.poses()` (n,3,4). Build them with the `mechanism_operator`
and `assemble_operator` steps (the operator READMEs give the config).

Pixel coordinates passed to `triangulate`, `with_picks`, `with_contours`,
and `View.crop` (center and marks) accept two-element lists as well as
tuples, so observations loaded from JSON can be passed directly.
