"""Measure the Stellar-L3N from the surveyed photographs.

Every reading samples the developed pictures through the surveyed cameras
(../cameras.py, checked against view-rectify to 0.005 px):

1. Frames: each display-down pose's rough frame (tag.json) refined by its
   visual hull's lowest sections (../hull.py, run first).
2. Heights by parallax: a window rectified onto horizontal planes in every
   view of a pose, phase-correlated against a reference view; each view's
   shift grows linearly with the plane's error, and the height where it
   vanishes is that view's reading. The label on the back (the back face),
   the raised logo panel, and with the display up, the barcode (the face).
3. Contact lines: the display face's outline where it meets the card. From
   outside inward, the first place a facing view stops showing the white
   card squares; the face's dark edge strip reads as the black squares do,
   so only rows over white squares count.
4. Edges on the back by triangulation: a straight edge's position in planes
   at several heights, per view, follows that view's parallax; the height
   where the views agree is the edge's, and their spread there its quality.
5. Wall heights: brightness up the plane of the face's outline. The wall is
   on that plane (one level) up to where the body steps in.
6. The display, seen with the display up: its edges on the face plane,
   against the face's far edge (a silhouette on the card).
7. From the hulls: the face's corner radius (circles fitted to the lowest
   sections' corners) and how far the clip strip on the +y wall stands
   out (the hull's reach over it beyond its reach over the plain wall).

Writes work/measurements.json (everything, in the tag frame) and
work/parameters.json (the measured_* values keychain.wgsl takes).

The tag frame: origin on the display face at the centre of its outline,
x along the length toward the end whose back carries the model label (the
back body is flush with the face there), y toward the side with the clip
strip (flush too), z out of the back.

    python3 examples/keychain_tag/stellar_l3n/measure.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import cameras  # noqa: E402

WORK = HERE / "work"
SPEC = json.loads((HERE / "tag.json").read_text())
UP = np.array([0.0, 0.0, 1.0])


class Frame:
    """A pose's frame on the card: centre (world x, y) and yaw."""

    def __init__(self, centre, yaw_deg: float):
        self.centre = np.array([centre[0], centre[1], 0.0])
        self.yaw = float(yaw_deg)
        y = np.radians(yaw_deg)
        self.ax = np.array([np.cos(y), np.sin(y), 0.0])
        self.ay = np.array([-np.sin(y), np.cos(y), 0.0])

    def world(self, x, y, z=0.0):
        x, y, z = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float), np.asarray(z, float))
        return self.centre + x[..., None] * self.ax + y[..., None] * self.ay + z[..., None] * UP

    def shifted(self, dx: float, dy: float) -> "Frame":
        c = self.centre + dx * self.ax + dy * self.ay
        return Frame(c[:2], self.yaw)

    def side(self, name: str):
        """(outward normal, along) of a side: +x, -x, +y, -y."""
        return {"+x": (self.ax, self.ay), "-x": (-self.ax, self.ay),
                "+y": (self.ay, self.ax), "-y": (-self.ay, self.ax)}[name]


def facing(view, point, normal) -> float:
    """How squarely a view faces a side: the horizontal direction to the
    eye against the outward normal, -1..1."""
    d = (view.eye - point)[:2]
    return float(d @ normal[:2] / np.linalg.norm(d))


def elevation(view, point) -> float:
    d = view.eye - point
    return float(np.degrees(np.arctan2(d[2], np.linalg.norm(d[:2]))))


# 1. Frames ----------------------------------------------------------------

def hull_frame(pose: str) -> tuple[Frame, list]:
    """The pose's frame from its robust hull sections up to 2.25 mm: the
    mean centre and median angle of their minimum-area rectangles."""
    d = np.load(WORK / "hull" / pose / "sections.npz")
    xs, ys, zs = d["xs"], d["ys"], d["z"]
    res = xs[1] - xs[0]
    rows = []
    for z, m in zip(zs, d["robust"]):
        if z > 0.00226:
            continue
        n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8))
        big = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        pts = np.argwhere(lab == big)[:, ::-1].astype(np.float32)
        (cx, cy), (w, h), ang = cv2.minAreaRect(pts)
        if w < h:
            w, h, ang = h, w, ang - 90
        # Image rows run down: a positive image angle is clockwise.
        rows.append({"z_m": float(z), "centre_m": [float(xs[0] + cx * res), float(ys[0] - cy * res)],
                     "angle_deg": float(((-ang + 90) % 180) - 90), "length_m": float(w * res),
                     "width_m": float(h * res)})
    rough = Frame(d["centre"], float(d["yaw_deg"]))
    a = np.mean([r["centre_m"] for r in rows], axis=0)
    c = rough.world(a[0], a[1])[:2]
    return Frame(c, rough.yaw + float(np.median([r["angle_deg"] for r in rows]))), rows


def hull_reach(pose: str, frame: Frame, side: str, span, z_range) -> float:
    """How far the robust hull reaches out on a side over `span` along it
    and heights `z_range`: an outer bound on anything there."""
    d = np.load(WORK / "hull" / pose / "sections.npz")
    rough = Frame(d["centre"], float(d["yaw_deg"]))
    A, B = np.meshgrid(d["xs"], d["ys"])
    W = rough.world(A, B)
    rel = W - frame.centre
    normal, along = frame.side(side)
    n, t = rel @ normal, rel @ along
    sel = (t >= span[0]) & (t <= span[1])
    reach = -np.inf
    for z, m in zip(d["z"], d["robust"]):
        if z_range[0] <= z <= z_range[1] and (m & sel).any():
            reach = max(reach, float(n[m & sel].max()))
    return reach


# 2. Heights by parallax ---------------------------------------------------

def plane_patch(view, frame: Frame, window, z, res=0.00005):
    """The view's picture on the plane at height z over a window in the
    frame (x0, x1, y0, y1), normalised."""
    xs = np.arange(window[0], window[1], res)
    ys = np.arange(window[3], window[2], -res)
    X, Y = np.meshgrid(xs, ys)
    g = view.sample(frame.world(X, Y, np.full_like(X, z)))
    g = cv2.GaussianBlur(np.nan_to_num(g, nan=float(np.nanmean(g))), (0, 0), 1.0)
    return ((g - g.mean()) / (g.std() + 1e-6)).astype(np.float32)


def parallax_height(views, ids, frame: Frame, window, ref: str, z_lo: float, z_hi: float, res=0.00005):
    """Each view's zero-shift height against `ref`, from phase correlation
    at two planes: (readings, median, IQR)."""
    hann = None
    shifts = {}
    for z in (z_lo, z_hi):
        r = plane_patch(views[ref], frame, window, z, res)
        if hann is None:
            hann = cv2.createHanningWindow(r.shape[::-1], cv2.CV_32F)
        for vid in ids:
            if vid == ref:
                continue
            g = plane_patch(views[vid], frame, window, z, res)
            (dx, dy), resp = cv2.phaseCorrelate(r * hann, g * hann)
            shifts.setdefault(vid, []).append((np.array([dx, -dy]) * res, resp))
    readings = {}
    for vid, ((s0, q0), (s1, q1)) in shifts.items():
        d = s1 - s0
        if np.linalg.norm(d) < 1e-6 or min(q0, q1) < 0.1:
            continue
        n = d / np.linalg.norm(d)
        a0, a1 = s0 @ n, s1 @ n
        readings[vid] = {"z_m": float(z_lo - a0 * (z_hi - z_lo) / (a1 - a0)), "response": float(min(q0, q1))}
    z = np.array([r["z_m"] for r in readings.values()])
    return readings, float(np.median(z)), float(np.subtract(*np.percentile(z, [75, 25])))


# 3. Contact lines -----------------------------------------------------------

def contact_lines(views, ids, frame: Frame, half: dict, card: cameras.Card, res=0.00002):
    """Per side and facing view, the outline of the face on the card."""
    out = {}
    for side in ("+x", "-x", "+y", "-y"):
        normal, along = frame.side(side)
        h, other = (half["x"], half["y"]) if side[1] == "x" else (half["y"], half["x"])
        ns = np.arange(-0.003, 0.004, res)
        ts = np.linspace(-0.9 * other, 0.9 * other, 400)
        N, T = np.meshgrid(ns, ts)
        P = frame.centre + (h + N)[..., None] * normal + T[..., None] * along
        c = card.sample(P)
        band = (ns > -0.0010) & (ns < 0.0030)
        white_rows = (np.ptp(c[:, band], axis=1) < 0.05) & (np.median(c[:, band], axis=1) > 0.95)
        outside = (ns > 0.0020) & (ns < 0.0029)
        found = []
        for vid in ids:
            v = views[vid]
            if facing(v, frame.centre + h * normal, normal) < 0.3 or white_rows.sum() < 10:
                continue
            g = cv2.GaussianBlur(np.nan_to_num(v.sample(P)), (0, 0), 1.5)
            white = np.median(g[white_rows][:, outside])
            black = np.percentile(g, 2)
            rows = []
            for row in g[white_rows]:
                dev = row < black + 0.75 * (white - black)
                run = 0
                for j in range(len(ns) - 1, -1, -1):
                    run = run + 1 if dev[j] else 0
                    if run >= int(0.0002 / res):
                        rows.append(ns[j + run - 1])
                        break
            if len(rows) > 15:
                rows = np.array(rows)
                found.append({"view": vid, "edge_m": float(h + np.median(rows)),
                              "iqr_m": float(np.subtract(*np.percentile(rows, [75, 25]))), "rows": len(rows)})
        out[side] = found
    return out


def corner_radii(pose: str) -> list[float]:
    """Circles fitted to the corners of the strict hull's sections up to
    1.25 mm (where it hugs the face): the face's corner radius per corner
    and height. Contact readings at the corners need white card there,
    which the poses did not give."""
    d = np.load(WORK / "hull" / pose / "sections.npz")
    xs, ys = d["xs"], d["ys"]
    res = xs[1] - xs[0]
    radii = []
    for z, m in zip(d["z"], d["strict"]):
        if z > 0.00126:
            continue
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        c = max(cnts, key=cv2.contourArea)[:, 0, :].astype(float)
        X, Y = xs[0] + c[:, 0] * res, ys[0] - c[:, 1] * res
        for sx, ex in ((1, X.max()), (-1, X.min())):
            for sy, ey in ((1, Y.max()), (-1, Y.min())):
                # The arc: within 6 mm of the corner, off both straight sides.
                u, v = sx * (X - ex), sy * (Y - ey)
                sel = (u > -0.006) & (v > -0.006) & (u < -0.00015) & (v < -0.00015)
                if sel.sum() < 10:
                    continue
                A = np.stack([X[sel], Y[sel], np.ones(sel.sum())], 1)
                (a1, a2, a3), *_ = np.linalg.lstsq(A, X[sel] ** 2 + Y[sel] ** 2, rcond=None)
                radii.append(float(np.sqrt(a3 + (a1 / 2) ** 2 + (a2 / 2) ** 2)))
    return radii


# 4. Edges by triangulation --------------------------------------------------

def edge_positions(view, frame: Frame, side: str, seed: float, span, z, search=0.0010, res=0.00002):
    """Where the strongest edge near `seed` (an offset along the side's
    normal) lies in the plane at z, averaged along `span` of the side."""
    normal, along = frame.side(side)
    ns = np.arange(seed - search, seed + search, res)
    ts = np.linspace(span[0], span[1], 120)
    N, T = np.meshgrid(ns, ts)
    P = frame.centre + N[..., None] * normal + T[..., None] * along + z * UP
    prof = np.nanmedian(view.sample(P), axis=0)
    if np.isnan(prof).any():
        return None
    g = np.abs(np.gradient(cv2.GaussianBlur(prof.reshape(1, -1).astype(np.float32), (0, 0), 3).ravel()))
    j = int(np.argmax(g[5:-5])) + 5
    if g[j] < 3 * np.median(g):
        return None
    # Sub-sample peak.
    a, b, c = g[j - 1], g[j], g[j + 1]
    off = 0.5 * (a - c) / (a - 2 * b + c) if (a - 2 * b + c) != 0 else 0.0
    return float(ns[j] + off * res)


def triangulate_edge(views, ids, frame: Frame, side: str, seed: float, span, zs):
    """The edge's height and offset: per view, the edge offset at planes
    zs is linear in z; the z where the views agree best is the edge's."""
    lines = {}
    for vid in ids:
        pos = [edge_positions(views[vid], frame, side, seed, span, z) for z in zs]
        ok = [(z, p) for z, p in zip(zs, pos) if p is not None]
        if len(ok) >= 3:
            z_, p_ = np.array(ok).T
            k, b = np.polyfit(z_, p_, 1)
            fit = float(np.abs(p_ - (k * z_ + b)).max())
            if fit < 0.0002:
                lines[vid] = (k, b)
    if len(lines) < 3:
        return None
    grid = np.linspace(min(zs) - 0.003, max(zs) + 0.003, 601)
    spread = [np.std([k * z + b for k, b in lines.values()]) for z in grid]
    i = int(np.argmin(spread))
    z = float(grid[i])
    offs = [k * z + b for k, b in lines.values()]
    return {"z_m": z, "edge_m": float(np.median(offs)), "spread_m": float(np.std(offs)), "views": len(lines)}


# 5. Wall heights --------------------------------------------------------------

def wall_height(views, ids, frame: Frame, side: str, offset: float, span, z_top: float):
    """Per facing view, where the brightness up the plane of the face's
    outline leaves the wall's level: the wall's top (z_top when it
    never does)."""
    normal, along = frame.side(side)
    zs = np.arange(0.0, z_top, 0.0001)
    ts = np.linspace(span[0], span[1], 160)
    Z, T = np.meshgrid(zs, ts, indexing="ij")
    P = frame.centre + offset * normal + T[..., None] * along + Z[..., None] * UP
    found = []
    for vid in ids:
        v = views[vid]
        if facing(v, frame.centre + offset * normal, normal) < 0.5:
            continue
        prof = cv2.GaussianBlur(np.nanmedian(v.sample(P), axis=1).reshape(1, -1).astype(np.float32),
                                (0, 0), 2).ravel()
        level = np.median(prof[(zs > 0.002) & (zs < 0.003)])
        off = np.abs(prof - level) > 0.12 * level
        above = np.where(off & (zs > 0.0025))[0]
        found.append({"view": vid, "z_m": float(zs[above[0]]) if len(above) else float(z_top)})
    return found


# 6. The display, display up ----------------------------------------------------

def face_edges(views, ids, frame: Frame, z_face: float, seeds: dict, res=0.00002):
    """Edges on the face plane per view: {name: (axis, seed, span)}."""
    out = {}
    for name, (axis, seed, span) in seeds.items():
        side = "+x" if axis == "x" else "+y"
        vals = []
        for vid in ids:
            p = edge_positions(views[vid], frame, side, seed, span, z_face, search=0.0010, res=res)
            if p is not None:
                vals.append(p)
        out[name] = {"edge_m": float(np.median(vals)), "spread_m": float(np.ptp(vals)), "views": len(vals)}
    return out


def summary(values):
    v = np.array(values)
    return {"median_m": float(np.median(v)), "spread_m": float(np.ptp(v)), "n": int(len(v))}


def main() -> int:
    views = cameras.load(WORK)
    card = cameras.Card(cameras.board(WORK))
    poses = SPEC["poses"]
    report = {"poses": {}}
    half0 = {"x": SPEC["size_m"][0] / 2, "y": SPEC["size_m"][1] / 2}

    for name in SPEC["display_down"]:
        ids = poses[name]["views"]
        frame, sections = hull_frame(name)
        r = {"hull_frame": {"centre_m": frame.centre[:2].tolist(), "yaw_deg": frame.yaw, "sections": sections}}
        # Contact lines; their mid-points re-centre the frame on the face.
        lines = contact_lines(views, ids, frame, half0, card)
        med = {s: float(np.median([f["edge_m"] for f in v])) for s, v in lines.items() if v}
        frame = frame.shifted((med["+x"] - med["-x"]) / 2, (med["+y"] - med["-y"]) / 2)
        r["contact"] = {s: {"readings": v, **summary([f["edge_m"] for f in v])} for s, v in lines.items()}
        r["face_length_m"] = med["+x"] + med["-x"]
        r["face_width_m"] = med["+y"] + med["-y"]
        r["frame"] = {"centre_m": frame.centre[:2].tolist(), "yaw_deg": frame.yaw}
        r["corner_radii_m"] = corner_radii(name)
        print(f"{name}: face {r['face_length_m'] * 1000:.2f} x {r['face_width_m'] * 1000:.2f} mm "
              f"(per side, views spread {', '.join(f'{s} {r['contact'][s]['spread_m'] * 1000:.2f}' for s in med)})")
        # Heights: the label (back face) and the logo panel.
        ref = poses[name]["reference_view"]
        for key, window, lo, hi in (("back_face", SPEC["label_window_m"], 0.0115, 0.0130),
                                    ("panel", SPEC["panel_window_m"], 0.0115, 0.0140)):
            readings, z, iqr = parallax_height(views, ids, frame, window, ref, lo, hi)
            r[f"{key}_z"] = {"median_m": z, "iqr_m": iqr, "readings": readings}
            print(f"  {key}: z {z * 1000:.2f} mm (IQR {iqr * 1000:.2f}, {len(readings)} views)")
        # Edges on the back.
        r["edges"] = {}
        for key, (side, seed, span) in SPEC["back_edges"].items():
            e = triangulate_edge(views, ids, frame, side, seed, span, [0.0110, 0.0118, 0.0126, 0.0134, 0.0142])
            r["edges"][key] = e
            if e:
                print(f"  edge {key}: {e['edge_m'] * 1000:6.2f} mm at z {e['z_m'] * 1000:5.2f} "
                      f"(views spread {e['spread_m'] * 1000:.2f}, {e['views']} views)")
            else:
                print(f"  edge {key}: no reading")
        # Wall heights on the plane of the face's outline.
        r["walls"] = {}
        for side, span in SPEC["wall_spans"].items():
            found = wall_height(views, ids, frame, side, med[side], span, 0.0124)
            r["walls"][side] = {"readings": found, **summary([f["z_m"] for f in found])}
            print(f"  wall {side}: top {r['walls'][side]['median_m'] * 1000:.2f} mm "
                  f"(spread {r['walls'][side]['spread_m'] * 1000:.2f}, {len(found)} views)")
        # The clip strip on the +y wall: the hull's reach over it and beside it.
        strip = SPEC["clip_strip"]
        r["clip_strip_reach_m"] = hull_reach(name, frame, "+y", strip["x_m"], strip["z_m"])
        r["beside_strip_reach_m"] = hull_reach(name, frame, "+y", strip["beside_x_m"], strip["z_m"])
        print(f"  hull on +y at the clip strip {r['clip_strip_reach_m'] * 1000:.2f} mm, beside it "
              f"{r['beside_strip_reach_m'] * 1000:.2f} mm")
        report["poses"][name] = r

    # The display, with the display up.
    up = SPEC["display_up"]
    ids = poses[up]["views"]
    frame = Frame(poses[up]["centre_m"], poses[up]["yaw_deg"])
    readings, z_face, iqr = parallax_height(views, ids, frame, SPEC["barcode_window_m"],
                                            poses[up]["reference_view"], 0.0120, 0.0145)
    edges = face_edges(views, ids, frame, z_face, SPEC["face_edges"])
    report["display_up"] = {"face_z": {"median_m": z_face, "iqr_m": iqr, "readings": readings}, "edges": edges}
    print(f"{up}: face plane z {z_face * 1000:.2f} mm (IQR {iqr * 1000:.2f}); "
          + ", ".join(f"{k} {v['edge_m'] * 1000:+.2f} ({v['spread_m'] * 1000:.2f})" for k, v in edges.items()))

    # Combined, in the tag frame. The face outline from the pose in diffuse
    # light (the lamp's cast shadow widens the other on one side).
    main_pose = report["poses"][SPEC["outline_pose"]]
    other = [p for k, p in report["poses"].items() if k != SPEC["outline_pose"]]
    length, width = main_pose["face_length_m"], main_pose["face_width_m"]
    z_back = float(np.median([p["back_face_z"]["median_m"] for p in report["poses"].values()]))
    z_panel = float(np.median([p["panel_z"]["median_m"] for p in report["poses"].values()]))

    def edge(key):
        """The pose whose views agree best on the edge."""
        found = [p["edges"][key] for p in report["poses"].values() if p["edges"].get(key)]
        return min(found, key=lambda e: e["spread_m"])["edge_m"] if found else None

    radii = main_pose["corner_radii_m"]
    corner = float(np.median(radii))
    # The strip stands out from the flush wall by what the hull reaches over
    # it beyond what it reaches beside it (its overshoot of the plain wall).
    strip_out = max(p["clip_strip_reach_m"] - p["beside_strip_reach_m"] for p in report["poses"].values())

    # The display in the tag frame: the face's far edge (+x of the display-up
    # chart, a silhouette, rounded: assumed_face_edge_inset inside the outline)
    # anchors the measured length; turned over about y, x reverses.
    inset = SPEC["assumed_face_edge_inset_m"]
    far = edges["face_far_x"]["edge_m"] + inset
    cx_up = far - length / 2
    cy_up = 0.5 * (edges["face_+y"]["edge_m"] + edges["face_-y"]["edge_m"])
    display = {
        "x_m": sorted([-(edges["display_+x"]["edge_m"] - cx_up), -(edges["display_-x"]["edge_m"] - cx_up)]),
        "y_m": sorted([edges["display_-y"]["edge_m"] - cy_up, edges["display_+y"]["edge_m"] - cy_up]),
    }
    report["tag"] = {
        "face_length_m": length, "face_width_m": width,
        "face_length_check_m": [p["face_length_m"] for p in other],
        "back_face_z_m": z_back, "panel_z_m": z_panel,
        "back_edges_m": {k: edge(k) for k in SPEC["back_edges"]},
        "wall_top_m": {s: float(np.median([p["walls"][s]["median_m"] for p in report["poses"].values()]))
                       for s in SPEC["wall_spans"]},
        "display_m": display,
        "face_corner_radius_m": corner,
        "face_corner_radii_m": radii,
        "clip_strip_protrusion_m": strip_out,
    }
    t = report["tag"]
    print(f"tag: face {length * 1000:.2f} x {width * 1000:.2f} mm, back face z {z_back * 1000:.2f}, "
          f"panel z {z_panel * 1000:.2f}; display x {display['x_m'][0] * 1000:+.2f}..{display['x_m'][1] * 1000:+.2f}, "
          f"y {display['y_m'][0] * 1000:+.2f}..{display['y_m'][1] * 1000:+.2f}")
    print("back edges: " + ", ".join(f"{k} {v * 1000:+.2f}" if v is not None else f"{k} -"
                                     for k, v in t["back_edges_m"].items()))
    print(f"face corner radius {corner * 1000:.2f} mm ({len(radii)} fits, {np.min(radii) * 1000:.2f}.."
          f"{np.max(radii) * 1000:.2f}); clip strip stands out {strip_out * 1000:.2f} mm")
    # The frame overlays pose the envelope in: the outline pose's.
    report["frame"] = {**main_pose["frame"], "pose": SPEC["outline_pose"]}
    (WORK / "measurements.json").write_text(json.dumps(report, indent=1) + "\n")
    print(f"wrote {WORK / 'measurements.json'}")
    e = t["back_edges_m"]
    params = {
        "measured_face_length": length,
        "measured_face_width": width,
        "measured_face_corner_radius": corner,
        "measured_back_z": z_back,
        "measured_panel_z": z_panel,
        # The back body's top face, its edges in the tag frame.
        "measured_body_x0": -e["back_-x"],
        "measured_body_x1": e["back_+x"],
        "measured_body_y0": -e["back_-y"],
        "measured_body_y1": e["back_+y"],
        "measured_panel_x1": e["panel_+x"],
        "measured_panel_y0": -e["panel_-y"],
        "measured_panel_y1": e["panel_+y"],
        # Where the face's wall ends and the body steps in.
        "measured_step_x_z": t["wall_top_m"]["-x"],
        "measured_step_y_z": t["wall_top_m"]["-y"],
        "measured_clip_protrusion": strip_out,
        "measured_display_x0": display["x_m"][0],
        "measured_display_x1": display["x_m"][1],
        "measured_display_y0": display["y_m"][0],
        "measured_display_y1": display["y_m"][1],
    }
    (WORK / "parameters.json").write_text(json.dumps(params, indent=2) + "\n")
    print(f"wrote {WORK / 'parameters.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
