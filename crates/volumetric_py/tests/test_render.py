"""The headless frame from Python."""

import numpy as np
import pytest

import volumetric as v

from test_project import cylinder_project


def rendered_or_skip(*args, **kwargs):
    try:
        return v.render(*args, **kwargs)
    except RuntimeError as err:
        if "adapter" in str(err).lower() or "gpu" in str(err).lower():
            pytest.skip(f"no GPU: {err}")
        raise


def test_presets_render_the_cylinder():
    p = cylinder_project()
    r = rendered_or_skip(p, views="iso,top", width=320, height=240, resolution=32)
    assert r.names == ["iso", "top"] and len(r.frames) == 2
    iso = r.image
    assert iso.shape == (240, 320, 4) and iso.dtype == np.uint8
    background = np.array([0x2D, 0x2D, 0x2D])
    assert (np.abs(iso[..., :3].astype(int) - background).sum(axis=2) > 30).mean() > 0.02
    (entity,) = r.report["entities"]
    assert entity["id"] == "post" and entity["triangles"] > 0
    assert r.report["up_source"] == "default" and r.report["gpu"]


def test_assets_from_a_run_render_with_an_explicit_camera():
    out = cylinder_project().run()
    r = rendered_or_skip([out["post"]], camera=([0.5, 0.5, 0.5], [0, 0, 0.1]), width=64, height=64,
                         resolution=32, grid=0.0, ssao=False, background="ffffff")
    assert r.names == [""] and r.image.shape == (64, 64, 4)
    assert (r.image[..., :3] < 200).any()  # the post is darker than the white ground
    with pytest.raises(ValueError, match="one of"):
        v.render(cylinder_project(), camera=([1, 1, 1],), through="x")
    with pytest.raises(ValueError, match="unknown view"):
        v.render(cylinder_project(), views="sideways")


def test_through_a_photograph_with_an_overlay(chair_views):
    p = v.Project()
    p.add_asset(chair_views.encode(), "viewset", id="views")
    p.add_op("cylinder_operator", [{"radius": 0.026}, [0.2569, 0.1509, 0.0], [0.2569, 0.1509, 0.45]], output="column")
    r = rendered_or_skip(p, through="views:DSC00742", overlay="edge", width=800, height=533, resolution=48)
    assert r.image.shape == (533, 800, 4)
    photo = chair_views.view("DSC00742").image()
    assert r.report["up_source"] == "default"  # the view set is not drawn, only looked through
    # The frame is mostly the photograph.
    small = np.asarray(v.Gray.from_rgb(photo).downsampled(2).array, float)
    assert abs(small.mean() - v.Gray.from_rgb(r.image[..., :3].copy()).array.mean()) < 25
    with pytest.raises(ValueError, match="through"):
        v.render(p, overlay="edge")


def test_marks_draw_the_views_observations_and_picks(chair_views):
    # A pick recorded on the view is drawn over the finished overlay: a
    # green cross for a fit pick, an orange diagonal one for a check.
    # Picks near a corner, where the lens moves pixels by some 60 px at
    # full size: the cross must land on the pick as recorded in the
    # photograph, not on its ideal pinhole position.
    picked = chair_views.with_picks({"hole": {"DSC00742": (600.0, 400.0)}},
                                    check={"hole": {"DSC00730": (5600.0, 3700.0)}})
    p = v.Project()
    p.add_views(picked, id="views")
    p.add_op("cylinder_operator", [{"radius": 0.026}, [0.2569, 0.1509, 0.0], [0.2569, 0.1509, 0.45]], output="column")
    plain = rendered_or_skip(p, through="views:DSC00742", overlay="edge", width=774, height=516, resolution=32)
    marked = v.render(p, through="views:DSC00742", overlay="edge", marks=True, width=774, height=516, resolution=32)
    changed = (np.abs(marked.image.astype(int) - plain.image.astype(int)).sum(axis=2) > 0)
    assert 500 < changed.sum() < 0.2 * changed.size  # marker quads and a cross, not a wash
    # The cross is centred on the pick: its arms are green there and above.
    assert any("through the lens" in note for note in marked.report["notes"]), marked.report["notes"]
    x, y = 600 * 774 / 6192, 400 * 516 / 4128
    patch = marked.image[int(y) - 3: int(y) + 4, int(x) - 3: int(x) + 4, :3].astype(int)
    assert ((patch[..., 1] > 200) & (patch[..., 0] < 200)).any(), "no green on the pick"
    # The pick's name sits just right of its cross: a dark label box with
    # green ink, a cross-and-a-bit (41 px * 1.3 at full size) to the right.
    lx = int(x + 41 * 1.3 * 774 / 6192)
    strip = marked.image[int(y) - 6: int(y) + 7, lx: lx + 40, :3].astype(int)
    assert (strip.max(axis=2) <= 24).sum() > 50, "no label box"
    assert ((strip[..., 1] > 200) & (strip[..., 0] < 200)).any(), "no label ink"
    check = v.render(p, through="views:DSC00730", marks=True, width=774, height=516, resolution=32, grid=0.0)
    x, y = 5600 * 774 / 6192, 3700 * 516 / 4128
    patch = check.image[int(y) - 3: int(y) + 4, int(x) - 3: int(x) + 4, :3].astype(int)
    assert ((patch[..., 0] > 200) & (patch[..., 2] < 150) & (patch[..., 1] < 230)).any(), "no orange on the check"
