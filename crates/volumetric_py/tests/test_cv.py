"""Markers and boards, on synthetic pictures and on the chair evidence."""

import numpy as np
import pytest

import volumetric as v


def look_down(position):
    """Camera-to-world for a camera at `position` looking along -z with
    +x right: rows are the camera axes' world directions and the
    translation."""
    return np.array([[1.0, 0.0, 0.0, position[0]],
                     [0.0, -1.0, 0.0, position[1]],
                     [0.0, 0.0, -1.0, position[2]]])


def project(camera, c2w, points):
    R, t = c2w[:, :3], c2w[:, 3]
    pc = (points - t) @ R  # R^T (p - t)
    return np.stack([camera.fx * pc[:, 0] / pc[:, 2] + camera.cx,
                     camera.fy * pc[:, 1] / pc[:, 2] + camera.cy], axis=1)


# The card on the z=0 plane, its right along +x and its down along -y, so
# a camera looking down -z with +x right sees it upright (mirrored
# markers do not decode).
ORIGIN = np.array([0.0, 0.2, 0.0])
RIGHT = np.array([1.0, 0.0, 0.0])
DOWN = np.array([0.0, -1.0, 0.0])


def on_card(bx, by):
    """A point of the card plane (right, down metres) in the world."""
    return ORIGIN + bx * RIGHT + by * DOWN


@pytest.fixture(scope="module")
def synthetic_card():
    camera = v.Camera(1600, 1200, 1500.0, 1500.0, 800.0, 600.0, label="synthetic")
    spec = v.survey_card()
    centre = on_card(spec["squares_x"] * spec["pitch_x_m"] / 2, spec["squares_y"] * spec["pitch_y_m"] / 2)
    c2w = look_down([centre[0], centre[1], 0.45])
    gray = v.render_board(camera, c2w, ORIGIN, RIGHT, DOWN, supersample=2)
    return camera, c2w, gray


def test_render_board_is_a_picture(synthetic_card):
    camera, c2w, gray = synthetic_card
    assert (gray.width, gray.height) == (1600, 1200)
    a = gray.array
    assert a.shape == (1200, 1600) and a.dtype == np.uint8
    assert a.min() < 40 and a.max() > 200  # ink and paper


def test_detect_finds_the_card_markers(synthetic_card):
    camera, c2w, gray = synthetic_card
    found = v.detect(gray, ["36h11"])
    ids = {d["id"] for d in found}
    assert len(ids) >= 40 and min(ids) >= 100
    assert all(d["family"] == "36h11" and d["fit_px"] < 0.5 for d in found)
    assert v.detect(gray, ["5x5_100"]) == []


def test_observe_locates_corners_where_the_camera_projects_them(synthetic_card):
    camera, c2w, gray = synthetic_card
    obs = v.observe(gray, swatches="5x5_100", board="survey_card")
    corners = {c["id"]: np.array(c["pixel"]) for c in obs["corners"]}
    truth = {cid: on_card(p[0], p[1]) for cid, p in v.board_corners()}
    assert len(corners) >= 80
    px = np.array([corners[i] for i in corners])
    expected = project(camera, c2w, np.array([truth[i] for i in corners]))
    err = np.linalg.norm(px - expected, axis=1)
    assert np.sqrt(np.mean(err ** 2)) < 0.3
    assert len(obs["observations"]["board"]) == len(obs["corners"])
    assert set(obs["blur"]) == {"horizontal", "vertical", "profiles"}
    assert v.observe(gray, swatches=None, board=False)["corners"] == []


def test_detect_params_are_checked():
    gray = v.Gray(np.full((64, 64), 200, np.uint8))
    assert v.detect(gray, threshold_constant=9) == []
    with pytest.raises(ValueError, match="unknown field"):
        v.detect(gray, thresholdconstant=9)


def test_gray_conversions_round_trip():
    rgb = np.zeros((4, 6, 3), np.uint8)
    rgb[..., 1] = 255
    g = v.Gray.from_rgb(rgb)
    assert (g.width, g.height) == (6, 4) and g.array.max() > 100
    assert v.Gray(g.array).array.tolist() == g.array.tolist()
    assert g.downsampled(2).array.shape == (2, 3)


def test_chair_still_solves_against_the_survey(chair_views):
    view = chair_views.view("DSC00742")
    gray = v.Gray.decode(view.picture())
    camera = view.camera
    scale = gray.width / camera.width
    seed = v.Camera(gray.width, gray.height, camera.fx * scale, camera.fy * scale,
                    camera.cx * scale, camera.cy * scale, distortion=camera.distortion)
    solve = v.solve_still(gray, chair_views, intrinsics=seed)
    assert solve["error"] is None, solve
    pose = solve["pose"]
    assert pose["rms_px"] < 1.0
    got = np.array(pose["camera_to_world"]).reshape(3, 4)
    assert np.linalg.norm(got[:, 3] - view.position) < 0.01
