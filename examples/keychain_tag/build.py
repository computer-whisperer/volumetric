"""Build the keychain bodies and the tag's envelope from the measurements.

Reads work/measurements.json (measure.py), writes work/parameters.json, the
project work/keychain.vproj (the rigid sleeve and the TPU bumper in the tag
frame, the tag envelope posed into the survey's world with a few
photographs for overlay audits), and the printable
work/keychain_{sleeve,bumper}.3mf / .stl in millimetres.

    python3 examples/keychain_tag/build.py [--work DIR] [--render]
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CLI = ROOT / "target/release/volumetric_cli"

# The bodies: (asset, WGSL part).
BODIES = [("sleeve", 1.0), ("bumper", 2.0)]

# Photographs kept in the project for overlays: views all round the tag.
AUDIT_VIEWS = ["DSC02139", "DSC02143", "DSC02147", "DSC02150", "DSC02152"]


def cli(*args) -> str:
    return subprocess.check_output([str(CLI), *map(str, args)], text=True)


def parameters(report: dict) -> dict:
    """measured_* from the report; the sleeve's assumed_* keep the WGSL
    defaults unless set here."""
    sections = report["sections"]
    strict = [s["strict"] for s in sections if "strict" in s]
    zs = np.array([s["z_m"] for s in sections if "strict" in s])
    length = np.array([s["length_m"] for s in strict])
    width = np.array([s["width_m"] for s in strict])
    top = report["back_face_z_m"]
    edges = report["edge_summary"]
    back_length = edges["back+x"]["median_m"] - edges["back-x"]["median_m"]
    back_width = edges["back+y"]["median_m"] - edges["back-y"]["median_m"]
    # The strict hull bounds the tag from outside (up to segmentation slips);
    # its largest outline is what the pocket must pass.
    params = {
        "measured_length": float(length.max()),
        "measured_width": float(width.max()),
        "measured_height": float(top),
        # The display face: the lowest section (the hull there is carved by
        # the contact line seen from every side).
        "measured_face_length": float(length[0]),
        "measured_face_width": float(width[0]),
        "measured_back_length": float(back_length),
        "measured_back_width": float(back_width),
        # Where the hull starts narrowing: last height within 0.5 mm of the
        # largest outline.
        "measured_length_taper_z": float(zs[np.where(length >= length.max() - 0.0005)[0].max()]),
        "measured_width_taper_z": float(zs[np.where(width >= width.max() - 0.0005)[0].max()]),
    }
    return params


def sample_lines(project, asset, lines):
    command = ["sample", "-i", project, "--asset", asset, "--json"]
    for line in lines:
        command += ["-l", line]
    return json.loads(cli(*command))


def report_fit(project, params, work):
    """Interference between the tag envelope and each body, sampled along x
    lines through the tag: none is the fit. The lips' overlaps onto the face
    and back (and the sleeve's detent) are the retention."""
    lines = []
    for y in np.linspace(-0.5, 0.5, 13) * params["measured_width"]:
        for z in np.linspace(0.0002, params["measured_height"] - 0.0002, 12):
            half = 0.5 * params["measured_length"] + 0.004
            lines.append(f"{-half},{y},{z} : {half},{y},{z} : 400")
    occupied = lambda rows: np.array([r["occupied"] for r in rows])
    t = occupied(sample_lines(project, "tag_local", lines)["samples"])
    fit = {"points": int(len(t)), "tag_points": int(t.sum())}
    for body, _ in BODIES:
        b = occupied(sample_lines(project, body, lines)["samples"])
        fit[f"{body}_interference_points"] = int((t & b).sum())
        print(f"fit: {fit[f'{body}_interference_points']} of {fit['tag_points']} tag points inside the {body}")
    (work / "fit.json").write_text(json.dumps(fit, indent=2) + "\n")
    if any(fit[f"{body}_interference_points"] for body, _ in BODIES):
        raise SystemExit("a body cuts into the tag envelope")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--work", type=Path, default=HERE / "work")
    ap.add_argument("--render", action="store_true")
    args = ap.parse_args()
    work = args.work.resolve()
    report = json.loads((work / "measurements.json").read_text())
    params = parameters(report)
    (work / "parameters.json").write_text(json.dumps(params, indent=2) + "\n")
    for k, v in params.items():
        print(f"{k:28s} {v * 1000:8.2f} mm")

    project = work / "keychain.vproj"
    cli("project-new", "-o", project)
    select = ["view-select", "-i", work / "survey.vviews", "-p", project, "--embed", "preview",
              "--preview-px", "1600"]
    for view in AUDIT_VIEWS:
        select += ["--id", view]
    cli(*select)
    cli("project-add-asset", "-p", project, "-i", HERE / "keychain.wgsl", "--type", "wgsl",
        "--asset-id", "keychain_source")

    def op(operator, inputs, output, exported=False):
        command = ["project-add-op", "-p", project, "--operator", operator, "--output-id", output]
        for item in inputs:
            command += ["--input", item if isinstance(item, str) else "json:" + json.dumps(item)]
        if not exported:
            command.append("--no-export")
        cli(*command)

    for body, part in BODIES:
        op("wgsl_script_operator", ["asset:keychain_source", {**params, "part": part}], body, exported=True)
    op("wgsl_script_operator", ["asset:keychain_source", {**params, "part": 0.0}], "tag_local", exported=True)
    frame = report["frame"]
    pose = {"rotate": {"rx_deg": 0.0, "ry_deg": 0.0, "rz_deg": frame["yaw_deg"]},
            "translate": dict(zip(["dx", "dy", "dz"], frame["origin_m"]))}
    op("pose_operator", ["asset:tag_local", pose], "tag_world", exported=True)
    (work / "project-run.json").write_text(cli("project-run", "-p", project, "--json"))

    report_fit(project, params, work)
    for body, _ in BODIES:
        for suffix in ("3mf", "stl"):
            print(cli("mesh", "-i", project, "--asset", body, "--unit", "mm", "--base-resolution", "16",
                      "--max-depth", "5", "-q", "-o", work / f"keychain_{body}.{suffix}").strip())
    if args.render:
        for view in AUDIT_VIEWS:
            print(cli("render", "-i", project, "--asset", "tag_world", "--through", view, "--overlay", "edge",
                      "--up", "0,0,1", "--width", "2400", "--height", "1600", "--resolution", "256",
                      "--grid", "0", "--no-ssao", "-o", work / f"overlay-{view}.png").strip())
        for body, _ in BODIES:
            print(cli("render", "-i", project, "--asset", body, "--views", "iso,iso-back,top,bottom",
                      "--up", "0,0,1", "--resolution", "256", "--grid", "0", "-o", work / f"{body}.png").strip())
    print(f"Built {project}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
