"""View sets and the survey on the chair evidence."""

import numpy as np
import pytest

import volumetric as v


def test_chair_views_read_as_arrays(chair_views):
    assert len(chair_views) == 10 and len(chair_views.cameras) == 1
    poses = chair_views.poses()
    assert poses.shape == (10, 3, 4) and not np.isnan(poses).any()
    corners = chair_views.marker_corners()
    assert corners.shape == (len(chair_views.marker_ids()), 4, 3)
    assert np.allclose(chair_views.world_up, [0, 0, 1])
    assert chair_views.board["spec"]["family"] == "36h11"
    view = chair_views.views[0]
    assert view.camera_to_world.shape == (3, 4)
    assert view.image().shape[2] == 3
    d = view.as_dict()
    assert d["has_image"] and not d["has_depth"]
    assert view.observations["markers"] and view.observations["board"]


def test_survey_reproduces_the_stored_poses(chair_views):
    """Ten views re-surveyed on their own solve the field again, so the
    poses move a few millimetres from the 182-view survey's, not more."""
    solved, report = v.survey(chair_views)
    assert report["unposed"] == [] and report["rms_px"] < 1.5
    assert len(report["frames"]) == 10 and report["cameras"][0]["f"] > 6000
    before, after = chair_views.poses(), solved.poses()
    assert np.abs(before[:, :, 3] - after[:, :, 3]).max() < 0.01
    assert all(view.observations is not None for view in solved.views)


def test_encode_decode_round_trip(chair_views, tmp_path):
    path = tmp_path / "copy.vviews"
    chair_views.save(str(path))
    again = v.ViewSet.load(str(path))
    assert again.encode() == chair_views.encode()
    assert v.ViewSet.decode(chair_views.encode()).views[3].id == chair_views.views[3].id


def test_select_keeps_named_views_and_reembeds(chair_views):
    a, b = chair_views.views[3].id, chair_views.views[0].id
    two = chair_views.select(ids=[a, b], embed="none")
    assert [v.id for v in two.views] == [a, b]
    assert two.views[0].picture() is None and len(two.cameras) == 1
    assert two.provenance["tools"][-1].endswith("(2 of 10 views)")
    every_third = chair_views.select(stride=3, posed=True)
    assert len(every_third) == 4 and every_third.views[0].picture() is not None
    small = chair_views.select(ids=["DSC00742"], embed="preview", preview_px=400)
    assert small.views[0].image().shape[1] == 400
    with pytest.raises(ValueError):
        chair_views.select(tags=["no-such-tag"])
    with pytest.raises(ValueError, match="embed"):
        chair_views.select(embed="thumbnail")


def test_crop_reads_the_original_with_marks(chair_views):
    view = chair_views.view("DSC00742")
    corner = chair_views.marker_corners()[0, 0]
    px = view.project(corner[None])[0]
    c = view.crop(center=px.tolist(), size=(200, 100), scale=2, grid=50, marks=[px.tolist()], world_marks=[corner])
    assert c.image.shape == (200, 400, 3) and c.image.dtype == np.uint8
    assert c.end[0] - c.origin[0] == 200 and c.scale == 2
    assert all(u % 50 == 0 and c.origin[0] <= u < c.end[0] for u in c.verticals)
    (projected,) = c.projected
    assert abs(projected[0] - px[0]) < 1e-9
    assert c.png()[:8] == b"\x89PNG\r\n\x1a\n"
    behind = view.crop(center=(100, 100), world_marks=[view.position + [0, 0, 1.0]])
    assert behind.projected == [None]
