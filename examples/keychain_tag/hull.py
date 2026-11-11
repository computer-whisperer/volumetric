"""Horizontal sections of a tag by visual hull from card occlusion.

Each surveyed view is rectified onto the card plane (volumetric_cli
view-rectify); the nominal ChArUco card rendered in the same chart says what
an unobstructed view shows there, so where the photo differs the tag is in
the way: the view's occlusion "shadow" on z = 0. A point P is inside the
hull when, for every view, the ray from the camera through P meets the card
inside that view's shadow. Sections at heights z give the outline per
height; concavities the cameras cannot see into stay filled.

A tag directory's tag.json names the poses: per pose the views taken
while the tag lay still, and roughly where it lay (centre on the card and
yaw, from a plan rectification). Sections are sampled on a grid in that
rough frame, so the outline comes out nearly axis-aligned.

    python3 examples/keychain_tag/hull.py <tag-dir> <pose> [--z 0.25:13.75:0.5]
    python3 examples/keychain_tag/hull.py <tag-dir> --shadows   # every view's shadow, for grouping
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CLI = ROOT / "target/release/volumetric_cli"

RES = 0.0001  # metres per pixel of the shadow charts and sections
WINDOW = 0.095  # half-size of the shadow chart about the pose's centre
GRID = 0.055  # half-size of the section grid in the pose's rough frame


class Pose:
    def __init__(self, spec: dict, size: list[float]):
        self.views = spec["views"]
        self.centre = np.array(spec["centre_m"], float)
        self.yaw = np.radians(spec["yaw_deg"])
        self.half = 0.5 * np.array(size[:2])
        c = self.centre
        self.u = (c[0] - WINDOW, c[0] + WINDOW)
        self.v = (c[1] - WINDOW, c[1] + WINDOW)

    def chart_coords(self, shape):
        h, w = shape
        u = self.u[0] + (np.arange(w) + 0.5) * RES
        v = self.v[1] - (np.arange(h) + 0.5) * RES
        return np.meshgrid(u, v)

    def local(self, x, y):
        """World card-plane coordinates to the pose's rough frame."""
        dx, dy = x - self.centre[0], y - self.centre[1]
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        return c * dx + s * dy, -s * dx + c * dy


def rectify(survey: Path, view: str, pose: Pose, out: Path) -> np.ndarray:
    if not out.exists():
        subprocess.run(
            [CLI, "view-rectify", "-i", survey, "--view", view, "--plane-z", "0",
             "--u-range", f"{pose.u[0]},{pose.u[1]}", "--v-range", f"{pose.v[0]},{pose.v[1]}",
             "--mm-per-px", str(RES * 1000), "--grid-mm", "0", "-o", out],
            check=True, capture_output=True)
    return cv2.cvtColor(cv2.imread(str(out)), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0


def card_render(board: dict, pose: Pose, shape: tuple[int, int]) -> np.ndarray:
    """The nominal card in the rectified chart: 1 white, 0 black. World x is
    the board's x, world y its -y (the survey's card frame)."""
    spec = board["spec"]
    sq = 200  # render pixels per square
    b = cv2.aruco.CharucoBoard(
        (spec["squares_x"], spec["squares_y"]), 1.0, spec["marker_m"] / spec["pitch_y_m"],
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11),
        np.arange(spec["first_id"], spec["first_id"] + spec["squares_x"] * spec["squares_y"] // 2))
    img = b.generateImage((spec["squares_x"] * sq, spec["squares_y"] * sq), marginSize=0).astype(np.float32) / 255.0
    u, v = pose.chart_coords(shape)
    mapx = (u / spec["pitch_x_m"] * sq).astype(np.float32)
    mapy = (-v / spec["pitch_y_m"] * sq).astype(np.float32)
    # Off the printed squares the card is its white margin.
    return cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=1.0)


def shadow(photo: np.ndarray, card: np.ndarray, pose: Pose, threshold: float) -> tuple[np.ndarray, dict]:
    """Where the photo is not the card, as the component under the tag with
    its holes filled; gain and offset fitted on the card away from the tag."""
    U, V = pose.chart_coords(photo.shape)
    a, b = pose.local(U, V)
    # Away from the tag: outside its rough outline grown by 35 mm, past the
    # longest shadow a low view casts.
    far = (np.abs(a) > pose.half[0] + 0.035) | (np.abs(b) > pose.half[1] + 0.035)
    # Blur both alike: the photo is soft at a few pixels.
    cb = cv2.GaussianBlur(card, (0, 0), 3)
    pb = cv2.GaussianBlur(photo, (0, 0), 1.5)
    seen = pb > 0.004  # black = unseen by the view
    sel = far & seen
    A = np.stack([cb[sel], np.ones(sel.sum())], 1)
    gain, offset = np.linalg.lstsq(A, pb[sel], rcond=None)[0]
    pred = gain * cb + offset
    diff = np.abs(pb - pred) / max(gain, 1e-6)
    # The tag is smooth where the card is textured: a white tag over a
    # white square still differs from the marker inside it.
    def local_std(x):
        mu = cv2.GaussianBlur(x, (0, 0), 6)
        return np.sqrt(np.maximum(cv2.GaussianBlur(x * x, (0, 0), 6) - mu * mu, 0))
    smooth = (local_std(pb) / max(gain, 1e-6) < 0.06) & (local_std(cb) > 0.15)
    m = (((diff > threshold) | smooth) & seen).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n, lab = cv2.connectedComponents(m)
    t = int(WINDOW / RES)
    r = int(0.008 / RES)
    ids = np.unique(lab[t - r:t + r, t - r:t + r])
    keep = np.isin(lab, ids[ids > 0]).astype(np.uint8)
    # Fill holes: a shadow is simply connected but for see-through slots.
    ff = keep.copy()
    cv2.floodFill(ff, None, (0, 0), 2)
    keep[ff == 0] = 1
    resid = float(np.median(np.abs(pb[sel] - gain * cb[sel] - offset)) / gain)
    return keep.astype(bool), {"gain": float(gain), "offset": float(offset), "card_residual": resid}


def view_positions(survey: Path) -> tuple[dict, dict]:
    cams = json.loads(subprocess.run([CLI, "view-list", "-i", survey, "--json"], check=True,
                                     capture_output=True, text=True).stdout)
    return {v["id"]: np.array(v["position"]) for v in cams["views"] if v["position"]}, cams["board"]


def shadows_of(work: Path, name: str, pose: Pose, threshold: float) -> dict:
    survey = work / "survey.vviews"
    positions, board = view_positions(survey)
    out = work / "hull" / name
    out.mkdir(parents=True, exist_ok=True)
    card = None
    shadows = {}
    for view in pose.views:
        if view not in positions:
            print(f"{view}: not posed, left out")
            continue
        photo = rectify(survey, view, pose, out / f"z0_{view}.png")
        if card is None:
            card = card_render(board, pose, photo.shape)
            cv2.imwrite(str(out / "card.png"), (card * 255).astype(np.uint8))
        m, fit = shadow(photo, card, pose, threshold)
        shadows[view] = (positions[view], m)
        cv2.imwrite(str(out / f"shadow_{view}.png"), m.astype(np.uint8) * 255)
        print(f"{view}: card residual {fit['card_residual']:.3f}, shadow {m.sum() * RES * RES * 1e6:.0f} mm^2")
    return shadows


def sections(pose: Pose, shadows: dict, zs, misses: int, out: Path) -> None:
    n = int(2 * GRID / RES)
    xs = -GRID + (np.arange(n) + 0.5) * RES  # along the rough frame's x
    ys = GRID - (np.arange(n) + 0.5) * RES  # rows run down
    A, B = np.meshgrid(xs, ys)
    c, s = np.cos(pose.yaw), np.sin(pose.yaw)
    X = pose.centre[0] + c * A - s * B
    Y = pose.centre[1] + s * A + c * B
    strict_all, robust_all = [], []
    for z in zs:
        out_votes = np.zeros(X.shape, np.int32)
        for cam, m in shadows.values():
            k = cam[2] / (cam[2] - z)
            qx = cam[0] + (X - cam[0]) * k
            qy = cam[1] + (Y - cam[1]) * k
            col = ((qx - pose.u[0]) / RES).astype(int)
            row = ((pose.v[1] - qy) / RES).astype(int)
            ok = (col >= 0) & (col < m.shape[1]) & (row >= 0) & (row < m.shape[0])
            hit = np.zeros(X.shape, bool)
            hit[ok] = m[row[ok], col[ok]]
            out_votes += ~(hit | ~ok)
        robust = out_votes <= misses
        strict = out_votes == 0
        robust_all.append(robust)
        strict_all.append(strict)
        cv2.imwrite(str(out / f"section_{z * 1000:05.2f}.png"), robust.astype(np.uint8) * 255)
        if robust.any():
            cols = np.where(robust.any(0))[0]
            rows = np.where(robust.any(1))[0]
            print(f"z {z * 1000:5.2f} mm: x {xs[cols[0]] * 1000:7.2f}..{xs[cols[-1]] * 1000:7.2f}, "
                  f"y {ys[rows[-1]] * 1000:7.2f}..{ys[rows[0]] * 1000:7.2f}, "
                  f"area {robust.sum() * RES * RES * 1e6:6.0f} mm^2")
    np.savez_compressed(out / "sections.npz", z=np.array(zs), robust=np.array(robust_all),
                        strict=np.array(strict_all), misses=misses, xs=xs, ys=ys,
                        centre=pose.centre, yaw_deg=np.degrees(pose.yaw))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("tag", type=Path, help="tag directory (tag.json, work/survey.vviews)")
    ap.add_argument("pose", nargs="?", help="pose name in tag.json")
    ap.add_argument("--shadows", action="store_true", help="every posed view's shadow about the card's centre")
    ap.add_argument("--z", help="z0:z1:dz in mm (default from tag.json)")
    ap.add_argument("--threshold", type=float, default=0.30)
    ap.add_argument("--misses", type=int, default=2,
                    help="views allowed to vote a point out (segmentation slips at card square edges)")
    args = ap.parse_args()
    spec = json.loads((args.tag / "tag.json").read_text())
    work = args.tag / "work"
    if args.shadows:
        positions, _ = view_positions(work / "survey.vviews")
        pose = Pose({"views": sorted(positions), **spec["survey_area"]}, spec["size_m"])
        shadows_of(work, "all", pose, args.threshold)
        return 0
    pose = Pose(spec["poses"][args.pose], spec["size_m"])
    shadows = shadows_of(work, args.pose, pose, args.threshold)
    z0, z1, dz = (float(s) / 1000 for s in (args.z or spec["sections_mm"]).split(":"))
    sections(pose, shadows, np.arange(z0, z1 + 1e-9, dz), args.misses, work / "hull" / args.pose)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
