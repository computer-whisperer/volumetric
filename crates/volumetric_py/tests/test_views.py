"""View sets and the survey on the chair evidence."""

import numpy as np

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
