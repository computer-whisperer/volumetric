#!/usr/bin/env python3
"""What the sharp stages leave behind, from a `wasm_mesh_view --dump-stages` dump.

    wasm_mesh_view model.wasm out --no-sharp --no-simplify --depth 6 \\
        --debug-corner 9,9,9,0 --dump-stages dump/
    sharp_stages.py table dump/ [dump2/ ...]
    sharp_stages.py compare dump_before/ dump_after/
    sharp_stages.py pics dump/ <inward|nonmanifold|folds|ee|ef ...> <count> out.png

`table` counts, on the welded mesh: open and non-manifold edges; distinct
corner points; slivers (triangles with all three vertices snapped and a height
under a quarter cell); triangles facing into the material (against the summed
stage-4 vertex normals); and hard folds (two neighbouring triangles more than
135 degrees apart), sorted by what is beside them.

`compare` takes two dumps of the same model from two builds and counts, per
stage-4 vertex, the snaps lost, gained, and kept with a different target.

`pics` draws the neighbourhood of a sample of defects, looking along the
surface normal: welded triangles (red when facing inward), the stage-4 mesh
dotted, an arrow from where each snap candidate was to where it is, and each
vertex's depth in cells where it is off the picture plane. A fold class is the
two vertex classes on the edge, then the two opposite it, e.g. `ee|ef`.

Vertex classes: e/c snapped to an edge/corner, r snapped and retracted,
u unclaimed and left, k claimed candidate left, f on a face.

Needs numpy; `pics` needs matplotlib.
"""
import collections
import os
import sys

import numpy as np

CLASSES = "fkurec"  # a welded vertex takes the last of its members' classes
SNAPPED = ["e", "c"]


def load(directory):
    def array(name, dtype, width):
        return np.fromfile(os.path.join(directory, name), dtype=dtype).reshape(-1, width)

    d = dict(
        before=array("before.f64", "<f8", 3),
        normals=array("normals.f64", "<f8", 3),
        tris=array("tris.u32", "<u4", 3).astype(np.int64),
        cls=np.fromfile(os.path.join(directory, "class.u8"), dtype="S1").astype(str),
        after=array("after.f64", "<f8", 3),
        remap=np.fromfile(os.path.join(directory, "remap.u32"), dtype="<u4").astype(np.int64),
        pos=array("welded.f64", "<f8", 3),
        wtris=array("welded_tris.u32", "<u4", 3).astype(np.int64),
        cell=float(open(os.path.join(directory, "cell.txt")).read()),
    )
    count = len(d["pos"])
    rank = np.zeros(count, int)
    np.maximum.at(rank, d["remap"], np.array([CLASSES.index(c) for c in d["cls"]]))
    d["wcls"] = np.array(list(CLASSES))[rank]
    d["ref"] = np.zeros((count, 3))
    np.add.at(d["ref"], d["remap"], d["normals"])

    tri, pos = d["wtris"], d["pos"]
    corners = pos[tri]
    cross = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    doubled_area = np.linalg.norm(cross, axis=1)
    d["fn"] = cross / np.maximum(doubled_area[:, None], 1e-300)
    longest = np.max(
        [np.linalg.norm(corners[:, (k + 1) % 3] - corners[:, k], axis=1) for k in range(3)], axis=0
    )
    d["height"] = doubled_area / np.maximum(longest, 1e-300)

    # Edges, sorted so that the faces on one edge are adjacent rows.
    edges = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    face = np.concatenate([np.arange(len(tri))] * 3)
    opposite = np.concatenate([tri[:, 2], tri[:, 0], tri[:, 1]])
    key = np.sort(edges, axis=1)
    order = np.lexsort((key[:, 1], key[:, 0]))
    d["edge"], d["edge_face"], d["edge_opp"] = key[order], face[order], opposite[order]
    first = np.ones(len(order), bool)
    first[1:] = (d["edge"][1:] != d["edge"][:-1]).any(axis=1)
    d["start"] = np.flatnonzero(first)
    d["faces_on"] = np.diff(np.append(d["start"], len(order)))
    two = d["start"][d["faces_on"] == 2]
    cos = np.clip((d["fn"][d["edge_face"][two]] * d["fn"][d["edge_face"][two + 1]]).sum(axis=1), -1, 1)
    d["fold"] = two[np.degrees(np.arccos(cos)) > 135]

    d["inward"] = (d["height"] > 0.02 * d["cell"]) & ((cross * d["ref"][tri].sum(axis=1)).sum(axis=1) < 0)
    all_snapped = np.isin(d["wcls"][tri], SNAPPED).all(axis=1)
    d["sliver"] = all_snapped & (d["height"] < 0.25 * d["cell"])
    return d


def fold_class(d, e, faces=2):
    c = d["wcls"]
    return "".join(sorted(c[d["edge"][e]])) + "|" + "".join(sorted(c[d["edge_opp"][e : e + faces]]))


def corner_points(d):
    points = d["pos"][d["wcls"] == "c"]
    return np.unique(np.round(points / (0.5 * d["cell"])).astype(np.int64), axis=0)


def table(directories):
    for directory in directories:
        d = load(directory)
        cls, wcls, folds = d["cls"], d["wcls"], d["fold"]
        quad = np.concatenate(
            [d["edge"][folds], np.stack([d["edge_opp"][folds], d["edge_opp"][folds + 1]], axis=1)], axis=1
        )
        beside_sliver = d["sliver"][d["edge_face"][folds]] | d["sliver"][d["edge_face"][folds + 1]]
        unsnapped = np.isin(wcls[quad], ["u", "k", "r"]).any(axis=1) & ~beside_sliver
        corner = (wcls[quad] == "c").any(axis=1) & ~unsnapped & ~beside_sliver
        other = ~beside_sliver & ~unsnapped & ~corner
        inward = collections.Counter("".join(sorted(x)) for x in wcls[d["wtris"][d["inward"]]])
        classes = collections.Counter(fold_class(d, e) for e in folds)
        print(f"{directory}")
        print(
            f"  snapped {np.isin(cls, SNAPPED).sum()} (corner {(cls == 'c').sum()}), "
            f"retracted {(cls == 'r').sum()}, other candidates {np.isin(cls, ['u', 'k']).sum()}"
        )
        print(
            f"  triangles {len(d['wtris'])}, open edges {(d['faces_on'] == 1).sum()}, "
            f"non-manifold edges {(d['faces_on'] > 2).sum()}, corner points {len(corner_points(d))}, "
            f"slivers {d['sliver'].sum()}"
        )
        print(f"  facing inward {d['inward'].sum()}: {dict(inward.most_common(6))}")
        print(
            f"  folds {len(folds)}: beside a sliver {beside_sliver.sum()}, with an unsnapped candidate "
            f"{unsnapped.sum()}, at a corner {corner.sum()}, other {other.sum()}; "
            + ", ".join(f"{n} {c}" for c, n in classes.most_common(8))
        )


def compare(before, after):
    a, b = load(before), load(after)
    assert np.array_equal(a["before"], b["before"]), "the two dumps are of different stage-4 meshes"
    cell = a["cell"]
    was, now = np.isin(a["cls"], SNAPPED), np.isin(b["cls"], SNAPPED)
    lost, gained, kept = was & ~now, ~was & now, was & now
    moved = np.linalg.norm(a["pos"][a["remap"]] - b["pos"][b["remap"]], axis=1) / cell
    print(f"snapped {was.sum()} -> {now.sum()}")
    print(f"  lost {lost.sum()}: retracted {(lost & (b['cls'] == 'r')).sum()}, rejected {(lost & (b['cls'] != 'r')).sum()}")
    print(f"  gained {gained.sum()}")
    print(
        f"  kept {kept.sum()}: target moved more than 0.05 cell {(kept & (moved > 0.05)).sum()}, "
        f"more than 0.25 cell {(kept & (moved > 0.25)).sum()}, at most {moved[kept].max() if kept.any() else 0:.2f}"
    )
    print(f"  corner points {len(corner_points(a))} -> {len(corner_points(b))}")
    rejected = a["after"][lost & (b["cls"] != "r")]
    if len(rejected):
        block = 6 * cell
        cells, counts = np.unique(np.round(rejected / block).astype(int), axis=0, return_counts=True)
        print("  rejected snaps, largest groups (count at position):")
        for i in np.argsort(-counts)[:8]:
            print(f"    {counts[i]:4d} at {(cells[i] * block).round(5).tolist()}")


def picture(d, centre, normal, ax, title, radius=3.0, mark=None):
    cell, pos, tri, wcls = d["cell"], d["pos"], d["wtris"], d["wcls"]
    near = np.linalg.norm(pos - centre, axis=1) < 1.5 * radius * cell
    if d["ref"][near].sum(axis=0) @ normal < 0:
        normal = -normal
    u = np.cross(normal, [0, 0, 1.0])
    if np.linalg.norm(u) < 0.3:
        u = np.cross(normal, [1.0, 0, 0])
    u /= np.linalg.norm(u)
    w = np.cross(normal, u)
    project = lambda p: np.stack([(p - centre) @ u, (p - centre) @ w], axis=-1) / cell
    depth = lambda p: ((p - centre) @ normal) / cell

    was_near = np.linalg.norm(d["before"] - centre, axis=1) < 1.5 * radius * cell
    for t in d["tris"][was_near[d["tris"]].all(axis=1)]:
        q = project(d["before"][t])
        ax.fill(q[:, 0], q[:, 1], facecolor="none", edgecolor=(0.2, 0.6, 0.2, 0.5), lw=0.5, ls=":")
    shown = np.flatnonzero(near[tri].all(axis=1))
    for t in shown[np.argsort([depth(pos[tri[t]]).mean() for t in shown])]:
        q = project(pos[tri[t]])
        colour = (1, 0.2, 0.2, 0.5) if d["inward"][t] else (0.8, 0.8, 0.8, 0.3)
        ax.fill(q[:, 0], q[:, 1], facecolor=colour, edgecolor="k", lw=0.5)
    colours = dict(f="tab:gray", e="tab:blue", c="tab:green", u="tab:orange", k="tab:purple", r="tab:red")
    for v in np.flatnonzero(was_near & (d["cls"] != "f")):
        start, end = project(d["before"][v]), project(pos[d["remap"][v]])
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", color=colours[d["cls"][v]], lw=0.9))
        ax.plot(*start, ".", color=colours[d["cls"][v]], ms=4)
    for v in np.flatnonzero(near):
        p = project(pos[v])
        ax.plot(*p, "o", color=colours[wcls[v]], ms=5)
        if abs(depth(pos[v])) > 0.15:
            ax.text(p[0] + 0.05, p[1] + 0.05, f"{depth(pos[v]):+.1f}", fontsize=6)
    if mark is not None:
        q = project(pos[list(mark)])
        ax.plot(q[:, 0], q[:, 1], "r-", lw=2.5)
    ax.set_aspect("equal")
    ax.set_xlim(-radius, radius)
    ax.set_ylim(-radius, radius)
    ax.set_title(title, fontsize=8)


def pics(directory, want, count, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = load(directory)
    items = []
    if want == "inward":
        for t in np.flatnonzero(d["inward"]):
            corners = d["wtris"][t]
            items.append((d["pos"][corners].mean(axis=0), -d["fn"][t], "inward " + "".join(sorted(d["wcls"][corners])), None))
    else:
        if want == "nonmanifold":
            edges = d["start"][d["faces_on"] > 2]
        else:
            edges = [e for e in d["fold"] if want == "folds" or fold_class(d, e) == want]
        for e in edges:
            a, b = d["edge"][e]
            items.append((0.5 * (d["pos"][a] + d["pos"][b]), d["fn"][d["edge_face"][e]], fold_class(d, e), (a, b)))
    if not items:
        print("nothing of that kind")
        return
    items = items[:: max(1, len(items) // count)][:count]
    columns = (len(items) + 1) // 2
    fig, axes = plt.subplots(2, columns, figsize=(6.5 * columns, 13), squeeze=False)
    for (centre, normal, title, mark), ax in zip(items, axes.ravel()):
        picture(d, centre, normal, ax, f"{title} at {np.round(centre, 5)}", mark=mark)
    fig.tight_layout()
    fig.savefig(out, dpi=72)
    print("wrote", out, len(items))


if __name__ == "__main__":
    command, args = (sys.argv[1], sys.argv[2:]) if len(sys.argv) > 2 else ("", [])
    if command == "table":
        table(args)
    elif command == "compare" and len(args) == 2:
        compare(*args)
    elif command == "pics" and len(args) == 4:
        pics(args[0], args[1], int(args[2]), args[3])
    else:
        sys.exit(__doc__)
