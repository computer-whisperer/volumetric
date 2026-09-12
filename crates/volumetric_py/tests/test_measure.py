"""Measuring in photographs: project, cast, triangulate, detect."""

import json
import pathlib

import numpy as np
import pytest

import volumetric as v

HERE = pathlib.Path(__file__).resolve().parents[3] / "examples/chair_photo"


def test_project_and_cast_round_trip(chair_views):
    view = chair_views.views[0]
    camera = view.camera
    u, w = np.meshgrid(np.linspace(300, camera.width - 300, 5), np.linspace(300, camera.height - 300, 4))
    px = np.stack([u.ravel(), w.ravel()], axis=1)
    world, depth = view.cast(px, z=0.1)                      # pixels onto the plane a decimetre up
    assert np.allclose(world[:, 2], 0.1) and (depth > 0.3).all()
    assert np.abs(view.project(world) - px).max() < 1e-6     # and back through the distortion
    world2, _ = view.cast(px, plane=([0, 0, 0.1], [0, 0, 1]))
    assert np.abs(world2 - world).max() < 1e-9
    rays = view.ray(px)
    assert np.allclose(np.linalg.norm(rays, axis=1), 1.0)
    assert np.isnan(view.project(np.array([view.position + [0, 0, 1.0]]))).all()  # behind the camera
    with pytest.raises(ValueError, match="plane="):
        view.cast(px)
    with pytest.raises(ValueError, match="behind"):
        view.cast(px, z=5.0)


def test_triangulate_recovers_a_marker_corner(chair_views):
    corner = chair_views.marker_corners()[0, 0]
    a, b = chair_views.views[0], chair_views.views[1]
    picks = {a.id: tuple(a.project(corner[None])[0]), b.id: tuple(b.project(corner[None])[0])}
    point, gaps = chair_views.triangulate(picks)
    assert np.linalg.norm(point - corner) < 1e-6 and gaps.max() < 1e-6
    with pytest.raises(ValueError, match="two views"):
        chair_views.triangulate({a.id: picks[a.id]})


def test_chair_photo_report_replays_exactly():
    """The other session's committed measurement replays through the
    bindings to the same points, ray misses and check-view errors."""
    survey = HERE / "work/survey.vviews"
    if not survey.exists():
        pytest.skip("chair_photo survey not built")
    views = v.ViewSet.load(str(survey))
    obs = json.loads((HERE / "observations.json").read_text())
    report = json.loads((HERE / "measurement-report.json").read_text())
    for name, picks in obs["features"].items():
        point, gaps = views.triangulate({k: tuple(picks[k]) for k in obs["fit_views"]})
        assert np.linalg.norm(point - report["features"][name]["world"]) < 1e-9
        assert np.allclose(gaps, report["features"][name]["gaps"])
        (check,) = obs["check_views"]
        projected = views.view(check).project(point[None])[0]
        expected = report["features"][name]["reprojections"][check]["error_px"]
        # The report predates the CLI's f32 -> f64 argument fix (2d672f8).
        assert abs(np.linalg.norm(projected - picks[check]) - expected) < 1e-3


def test_detect_stores_observations(chair_views):
    stripped = v.ViewSet.decode(chair_views.encode())
    detected, reports, skipped = stripped.detect(ids=["DSC00742"], swatches="5x5_100", card=str(HERE / "card.json"))
    assert skipped == [] and len(reports) == 1
    (report,) = reports
    assert report["id"] == "DSC00742" and report["width"] == 6192
    assert len(report["detections"]) > 40 and len(report["corners"]) > 80
    obs = detected.view("DSC00742").observations
    assert len(obs["markers"]) == len(report["detections"])
    assert detected.board["spec"] == v.card_spec(str(HERE / "card.json"))
    assert v.card_spec()["family"] == "36h11"
    with pytest.raises(ValueError, match="nothing to look for"):
        stripped.detect(swatches=None, card=False)


def test_import_stills_makes_an_unposed_set(tmp_path):
    session = pathlib.Path("/ceph/christian/index_scanner/sessions/chairbase-dslr-1")
    stills = sorted(session.glob("*.JPG"))[:2] if session.exists() else []
    if not stills:
        pytest.skip("chairbase stills not present")
    for still in stills:
        (tmp_path / still.name).write_bytes(still.read_bytes())
    views, report = v.import_stills(str(tmp_path), embed="preview", preview_px=800, labels={"session": "test"})
    assert len(views) == 2 and report["total"] == 2
    assert np.isnan(views.poses()).all()
    assert views.views[0].image().shape[1] == 800
    assert views.provenance["session"] == "test"
    with pytest.raises(ValueError, match="unknown field"):
        v.import_stills(str(tmp_path), preview="none")


def test_recorded_picks_replay_the_chair_photo_report():
    """The other session's picks, recorded on the views instead of in
    JSON, fit to the report's points with the check view's errors."""
    survey = HERE / "work/survey.vviews"
    if not survey.exists():
        pytest.skip("chair_photo survey not built")
    obs = json.loads((HERE / "observations.json").read_text())
    report = json.loads((HERE / "measurement-report.json").read_text())
    fit = {name: {view: tuple(px) for view, px in picks.items() if view in obs["fit_views"]}
           for name, picks in obs["features"].items()}
    check = {name: {view: tuple(px) for view, px in picks.items() if view in obs["check_views"]}
             for name, picks in obs["features"].items()}
    views = v.ViewSet.load(str(survey)).with_picks(fit, check=check)
    recorded = views.picks()
    assert recorded["cross_a"]["DSC00756"]["role"] == "check"
    assert len(recorded) == 6 and all(len(p) == 3 for p in recorded.values())
    fits = views.fit_picks()
    for name, f in fits.items():
        ref = report["features"][name]
        assert np.linalg.norm(f["world"] - ref["world"]) < 1e-9
        assert abs(f["max_gap"] - max(ref["gaps"])) < 1e-12
        (check_view,) = obs["check_views"]
        assert abs(f["max_check_px"] - ref["reprojections"][check_view]["error_px"]) < 1e-3
        roles = {p["view"]: p["role"] for p in f["picks"]}
        assert roles[check_view] == "check" and sum(r == "fit" for r in roles.values()) == 2
    # Picks survive encoding and selection, and a re-detection.
    again = v.ViewSet.decode(views.encode())
    assert again.picks() == recorded
    subset = views.select(ids=["DSC00755"], embed="none")
    assert set(subset.picks()) == set(recorded) and all(list(p) == ["DSC00755"] for p in subset.picks().values())


def test_picks_and_contours_validate(chair_views):
    a, b = chair_views.views[0].id, chair_views.views[1].id
    one = chair_views.with_picks({"lone": {a: (10.0, 20.0)}})
    assert one.fit_picks()["lone"]["error"].startswith("feature \"lone\" has 1 fit pick")
    with pytest.raises(ValueError, match="needs two"):
        one.fit_picks(["lone"])
    with pytest.raises(ValueError, match="no view"):
        chair_views.with_picks({"x": {"nope": (1.0, 1.0)}})
    traced = chair_views.with_contours({"rim": {a: [(1.0, 2.0), (3.0, 4.0)], b: [(5.0, 6.0)]}})
    assert traced.contours() == {"rim": {a: [(1.0, 2.0), (3.0, 4.0)], b: [(5.0, 6.0)]}}
    with pytest.raises(ValueError, match="empty"):
        chair_views.with_contours({"rim": {a: []}})
    # Detection replaces the automatic observations and keeps the picks.
    detected, _, _ = one.detect(ids=[a], swatches="5x5_100", card=str(HERE / "card.json"))
    assert detected.picks() == one.picks() and detected.view(a).observations["markers"]
