"""Horizontal sections of the tag by visual hull from card occlusion.

Each surveyed view is rectified onto the card plane (volumetric_cli
view-rectify); the nominal ChArUco card rendered in the same chart says what
an unobstructed view shows there, so where the photo differs the tag is in
the way: the view's occlusion "shadow" on z = 0. A point P is inside the
hull when, for every view, the ray from the camera through P meets the card
inside that view's shadow. Sections at heights z give the outline per
height; concavities the cameras cannot see into stay filled.

    python3 examples/keychain_tag/hull.py [--work DIR] [--z 0.5:13.5:0.5]
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

# The rectified chart window on the card (metres) and its resolution.
WIN_U = (0.030, 0.190)
WIN_V = (-0.160, -0.035)
RES = 0.0001
# Where the tag is, for picking the shadow component (from view-pick casts).
TAG_XY = (0.110, -0.097)


def rectify(survey: Path, view: str, out: Path) -> np.ndarray:
    if not out.exists():
        subprocess.run(
            [CLI, "view-rectify", "-i", survey, "--view", view, "--plane-z", "0",
             "--u-range", f"{WIN_U[0]},{WIN_U[1]}", "--v-range", f"{WIN_V[0]},{WIN_V[1]}",
             "--mm-per-px", str(RES * 1000), "--grid-mm", "0", "-o", out],
            check=True, capture_output=True)
    return cv2.cvtColor(cv2.imread(str(out)), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0


def card_render(board: dict, shape: tuple[int, int]) -> np.ndarray:
    """The nominal card in the rectified chart: 1 white, 0 black. World x is
    the board's x, world y its -y (the survey's card frame)."""
    spec = board["spec"]
    sq = 200  # render pixels per square
    b = cv2.aruco.CharucoBoard(
        (spec["squares_x"], spec["squares_y"]), 1.0, spec["marker_m"] / spec["pitch_y_m"],
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11),
        np.arange(spec["first_id"], spec["first_id"] + spec["squares_x"] * spec["squares_y"] // 2))
    img = b.generateImage((spec["squares_x"] * sq, spec["squares_y"] * sq), marginSize=0).astype(np.float32) / 255.0
    h, w = shape
    u = WIN_U[0] + (np.arange(w) + 0.5) * RES
    v = WIN_V[1] - (np.arange(h) + 0.5) * RES
    mx = (u / spec["pitch_x_m"] * sq).astype(np.float32)
    my = (-v / spec["pitch_y_m"] * sq).astype(np.float32)
    mapx, mapy = np.meshgrid(mx, my)
    # Off the printed squares the card is its white margin.
    return cv2.remap(img, mapx, mapy, cv2.INTER_AREA if False else cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=1.0)


def shadow(photo: np.ndarray, card: np.ndarray, threshold: float) -> tuple[np.ndarray, dict]:
    """Where the photo is not the card, as the component under the tag with
    its holes filled; gain and offset fitted on the card away from the tag."""
    h, w = photo.shape
    u = WIN_U[0] + (np.arange(w) + 0.5) * RES
    v = WIN_V[1] - (np.arange(h) + 0.5) * RES
    U, V = np.meshgrid(u, v)
    far = (np.hypot(U - TAG_XY[0], (V - TAG_XY[1]) * 2.0) > 0.070)
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
    ty = int((WIN_V[1] - TAG_XY[1]) / RES)
    tx = int((TAG_XY[0] - WIN_U[0]) / RES)
    r = int(0.008 / RES)
    ids = np.unique(lab[ty - r:ty + r, tx - r:tx + r])
    keep = np.isin(lab, ids[ids > 0]).astype(np.uint8)
    # Fill holes: a shadow is simply connected but for see-through slots.
    ff = keep.copy()
    cv2.floodFill(ff, None, (0, 0), 2)
    keep[ff == 0] = 1
    resid = float(np.median(np.abs(pb[sel] - gain * cb[sel] - offset)) / gain)
    return keep.astype(bool), {"gain": float(gain), "offset": float(offset), "card_residual": resid}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--work", type=Path, default=HERE / "work")
    ap.add_argument("--z", default="0.25:13.75:0.5", help="z0:z1:dz in mm")
    ap.add_argument("--threshold", type=float, default=0.30)
    ap.add_argument("--misses", type=int, default=2,
                    help="views allowed to vote a point out (segmentation slips at card square edges)")
    args = ap.parse_args()
    work = args.work
    survey = work / "survey.vviews"
    cams = json.loads(subprocess.run([CLI, "view-list", "-i", survey, "--json"], check=True,
                                     capture_output=True, text=True).stdout)
    (work / "hull").mkdir(exist_ok=True)
    views = [v for v in cams["views"] if v["position"]]
    card = None
    shadows = {}
    for v in views:
        photo = rectify(survey, v["id"], work / "hull" / f"z0_{v['id']}.png")
        if card is None:
            card = card_render(cams["board"], photo.shape)
            cv2.imwrite(str(work / "hull" / "card.png"), (card * 255).astype(np.uint8))
        m, fit = shadow(photo, card, args.threshold)
        shadows[v["id"]] = (np.array(v["position"]), m)
        cv2.imwrite(str(work / "hull" / f"shadow_{v['id']}.png"), m.astype(np.uint8) * 255)
        print(f"{v['id']}: card residual {fit['card_residual']:.3f}, shadow {m.sum() * RES * RES * 1e6:.0f} mm^2")
    z0, z1, dz = (float(s) / 1000 for s in args.z.split(":"))
    # Section grid over the tag's neighbourhood.
    xs = np.arange(0.060, 0.160, RES) + RES / 2
    ys = np.arange(-0.125, -0.070, RES) + RES / 2
    X, Y = np.meshgrid(xs, ys[::-1])
    sections = []
    for z in np.arange(z0, z1 + 1e-9, dz):
        out_votes = np.zeros(X.shape, np.int32)
        for c, m in shadows.values():
            s = c[2] / (c[2] - z)
            qx = c[0] + (X - c[0]) * s
            qy = c[1] + (Y - c[1]) * s
            col = ((qx - WIN_U[0]) / RES).astype(int)
            row = ((WIN_V[1] - qy) / RES).astype(int)
            ok = (col >= 0) & (col < m.shape[1]) & (row >= 0) & (row < m.shape[0])
            hit = np.zeros(X.shape, bool)
            hit[ok] = m[row[ok], col[ok]]
            out_votes += ~(hit | ~ok)
        inside = out_votes <= args.misses
        sections.append((z, inside, out_votes == 0))
        cv2.imwrite(str(work / "hull" / f"section_{z * 1000:05.2f}.png"), inside.astype(np.uint8) * 255)
        if inside.any():
            cols = np.where(inside.any(0))[0]
            rows = np.where(inside.any(1))[0]
            print(f"z {z * 1000:5.2f} mm: x {xs[cols[0]] * 1000:7.2f}..{xs[cols[-1]] * 1000:7.2f}, "
                  f"y {ys[::-1][rows[-1]] * 1000:7.2f}..{ys[::-1][rows[0]] * 1000:7.2f}, "
                  f"area {inside.sum() * RES * RES * 1e6:6.0f} mm^2")
    np.savez_compressed(work / "hull" / "sections.npz", z=np.array([s[0] for s in sections]),
                        robust=np.array([s[1] for s in sections]), strict=np.array([s[2] for s in sections]),
                        misses=args.misses, xs=xs, ys=ys[::-1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
