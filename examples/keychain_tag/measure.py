"""Measure the tag's envelope from the surveyed photographs.

Three independent readings, all through volumetric_cli view-rectify:

1. Height of the back face: plane sweep of the printed "B" on it. At the
   right height every view's rectification puts the logo in one place.
2. Edges: per-row gradient edges in metric rectifications. Contact lines at
   z = 0 on the sides that face a camera, and the back face's outline at its
   height where it is a silhouette against the card's black squares.
3. Sections: the visual hull from hull.py (run it first), as the outline per
   height. It bounds the part from outside; concavities stay filled.

The report (work/measurements.json) gives the tag frame (origin on the face
plane at the centre of the outline, x along the length, z out of the back)
and the envelope in it, in metres.

    python3 examples/keychain_tag/measure.py [--work DIR]
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

# Views that look down on the logo (elevation 55-75 deg).
LOGO_VIEWS = ["DSC02147", "DSC02148", "DSC02149", "DSC02150", "DSC02152", "DSC02153"]
# The logo's neighbourhood on the card plane, from a first rectification.
LOGO_WINDOW = ((0.075, 0.100), (-0.110, -0.085))


def rectify(survey, view, z, u_range, v_range, res_mm, out):
    subprocess.run(
        [CLI, "view-rectify", "-i", survey, "--view", view, "--plane-z", repr(float(z)),
         "--u-range", f"{u_range[0]},{u_range[1]}", "--v-range", f"{v_range[0]},{v_range[1]}",
         "--mm-per-px", repr(float(res_mm)), "--grid-mm", "0", "-o", out],
        check=True, capture_output=True)
    return cv2.imread(str(out)).astype(np.float32)


def logo_height(survey, scratch):
    """The z where the views agree on the blue logo, parabola-refined."""
    def disagreement(z):
        blues = []
        for view in LOGO_VIEWS:
            im = rectify(survey, view, z, *LOGO_WINDOW, 0.1, scratch / f"logo_{view}.png")
            b = im[:, :, 0] - (im[:, :, 1] + im[:, :, 2]) / 2
            blues.append((b - b.mean()) / (b.std() + 1e-6))
        return float(np.stack(blues).std(0).mean())

    zs = np.arange(0.0120, 0.0156, 0.0002)
    scores = np.array([disagreement(z) for z in zs])
    i = int(np.clip(np.argmin(scores), 1, len(zs) - 2))
    a, b, _ = np.polyfit(zs[i - 1:i + 2], scores[i - 1:i + 2], 2)
    return float(-b / (2 * a)), [[float(z), float(s)] for z, s in zip(zs, scores)]


# Edge readings: (side, plane z, expected coordinate, band along the edge).
SIDES = {"+x": (1, 0), "-x": (-1, 0), "+y": (0, 1), "-y": (0, -1)}


def edge(survey, scratch, view, z, side, expect, band, half=0.005, res=0.00005, dark=70):
    """Median per-row position of the bright-inside to dark-outside edge, with
    the rows where the outside is a black card square."""
    ax = 0 if side[1] == "x" else 1
    window = [(expect - half, expect + half), band] if ax == 0 else [band, (expect - half, expect + half)]
    im = rectify(survey, view, z, *window, res * 1000, scratch / f"edge_{view}_{side}.png")
    g = cv2.GaussianBlur(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), (0, 0), 2)
    if ax == 1:
        g = g[::-1].T  # rows now run along +y
    coord = expect - half + (np.arange(g.shape[1]) + 0.5) * res
    sign = SIDES[side][ax]
    found = []
    for row in g:
        d = -sign * np.gradient(row)
        j = int(np.argmax(d))
        k = j + sign * int(0.0015 / res)
        if 0 <= k < len(row) and row[k] < dark and d[j] > 2:
            found.append(coord[j])
    if len(found) < 0.25 * len(g):
        return None
    found = np.array(found)
    return float(np.median(found)), float(np.subtract(*np.percentile(found, [75, 25]))), len(found)


def card_lines(board, value, axis):
    """Distance to the nearest card square boundary: an edge that close may be
    the card's own, not the tag's."""
    pitch = board["spec"]["pitch_x_m"] if axis == 0 else board["spec"]["pitch_y_m"]
    t = abs(value) / pitch
    return abs(t - round(t)) * pitch


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--work", type=Path, default=HERE / "work")
    args = ap.parse_args()
    work = args.work
    survey = work / "survey.vviews"
    scratch = work / "measure"
    scratch.mkdir(exist_ok=True)
    cams = json.loads(subprocess.run([CLI, "view-list", "-i", survey, "--json"], check=True,
                                     capture_output=True, text=True).stdout)
    positions = {v["id"]: np.array(v["position"]) for v in cams["views"] if v["position"]}
    report = {"survey": {"views_posed": len(positions)}}

    # 1. The back face's height.
    top, sweep = logo_height(survey, scratch)
    report["back_face_z_m"] = top
    report["logo_sweep"] = sweep
    print(f"back face (logo sweep): z = {top * 1000:.2f} mm")

    # 3. Sections (first: they give the frame the edges are read in).
    npz = np.load(work / "hull" / "sections.npz")
    xs, ys, zs = npz["xs"], npz["ys"], npz["z"]
    res = float(xs[1] - xs[0])
    rows = []
    for z, robust, strict in zip(zs, npz["robust"], npz["strict"]):
        entry = {"z_m": float(z)}
        for name, mask in (("robust", robust), ("strict", strict)):
            m = mask.astype(np.uint8)
            n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
            if n < 2:
                continue
            big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            pts = np.argwhere(lab == big)[:, ::-1].astype(np.float32)  # (col, row)
            (cx, cy), (w, h), angle = cv2.minAreaRect(pts)
            if w < h:
                w, h, angle = h, w, angle - 90
            entry[name] = {
                "center_m": [float(xs[0] + cx * res), float(ys[0] - cy * res)],
                "length_m": float(w * res), "width_m": float(h * res),
                # Image rows run down: a positive image angle is clockwise.
                "angle_deg": float(-angle), "area_m2": float(stats[big, cv2.CC_STAT_AREA] * res * res)}
        rows.append(entry)
    report["sections"] = rows
    base = [r["robust"] for r in rows if 0.001 <= r["z_m"] <= 0.005]
    center = np.mean([b["center_m"] for b in base], axis=0)
    angle = float(np.median([b["angle_deg"] for b in base]))
    report["frame"] = {"origin_m": [float(center[0]), float(center[1]), 0.0], "yaw_deg": angle,
                       "note": "x along the tag's length toward the plain end, z out of the back"}
    print(f"frame: origin ({center[0] * 1000:.2f}, {center[1] * 1000:.2f}) mm, yaw {angle:+.2f} deg")

    # 2. Edges, read along the frame's axes (the yaw is small: the windows
    # stay world-aligned and the edge positions are compared to the hull).
    board = cams["board"]
    edges = []
    b0 = base[len(base) // 2]
    half_l, half_w = b0["length_m"] / 2, b0["width_m"] / 2
    last = rows[-1]["robust"]
    t_half_l, t_half_w = last["length_m"] / 2, last["width_m"] / 2
    for plane, (hl, hw), mode in ((0.0, (half_l, half_w), "facing"), (top, (t_half_l, t_half_w), "silhouette")):
        for side, n in SIDES.items():
            ax = 0 if side[1] == "x" else 1
            expect = center[ax] + n[ax] * (hl if ax == 0 else hw)
            band = ((center[1] - 0.6 * hw, center[1] + 0.6 * hw) if ax == 0
                    else (center[0] - 0.6 * hl, center[0] + 0.6 * hl))
            for view, c in sorted(positions.items()):
                d = (c[:2] - center) / np.linalg.norm(c[:2] - center)
                f = float(d @ np.array(n))
                if (mode == "facing" and f < 0.3) or (mode == "silhouette" and f > -0.3):
                    continue
                r = edge(survey, scratch, view, plane, side, expect, band)
                if r is None:
                    continue
                near = card_lines(board, r[0], ax)
                edges.append({"view": view, "side": side, "plane_z_m": plane, "mode": mode,
                              "edge_m": r[0], "iqr_m": r[1], "rows": r[2],
                              "card_line_m": near, "suspect": bool(near < 0.0004 or r[1] > 0.0005)})
    report["edges"] = edges
    summary = {}
    for plane_name, plane in (("face", 0.0), ("back", top)):
        for side in SIDES:
            vals = [e["edge_m"] for e in edges if e["side"] == side and e["plane_z_m"] == plane and not e["suspect"]]
            if vals:
                summary[f"{plane_name}{side}"] = {"median_m": float(np.median(vals)), "n": len(vals),
                                                  "spread_m": float(np.ptp(vals))}
    report["edge_summary"] = summary
    for k, v in summary.items():
        print(f"edge {k:7s}: {v['median_m'] * 1000:8.2f} mm over {v['n']} views (spread {v['spread_m'] * 1000:.2f})")

    # The envelope in the tag frame: the hull's outline per height, with the
    # edges as the check. Lengths from the robust hull (an outer bound).
    env = []
    for r in rows:
        if "robust" in r and r["z_m"] <= top:
            env.append([r["z_m"], r["robust"]["length_m"], r["robust"]["width_m"]])
    env = np.array(env)
    report["envelope"] = {
        "z_m": env[:, 0].tolist(), "length_m": env[:, 1].tolist(), "width_m": env[:, 2].tolist(),
        "max_length_m": float(env[:, 1].max()), "max_width_m": float(env[:, 2].max()),
        "top_length_m": float(env[-1, 1]), "top_width_m": float(env[-1, 2]),
        "height_m": top,
    }
    e = report["envelope"]
    print(f"envelope: {e['max_length_m'] * 1000:.2f} x {e['max_width_m'] * 1000:.2f} mm, back face "
          f"{e['top_length_m'] * 1000:.2f} x {e['top_width_m'] * 1000:.2f} mm, {top * 1000:.2f} mm thick")
    (work / "measurements.json").write_text(json.dumps(report, indent=1) + "\n")
    print(f"wrote {work / 'measurements.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
