"""Build a tag's keychain bodies and its envelope from its parameters.

Reads <tag-dir>/work/parameters.json (the tag's measure.py) and the
measured frame in work/measurements.json, and writes work/keychain.vproj
(the bodies in the tag frame, the envelope posed into the survey's world
with a few photographs for overlay audits), work/fit.json, and the
printable work/keychain_<body>.3mf / .stl in millimetres. The bodies,
the audit views and which parameters size the fit check come from the
`build` section of <tag-dir>/tag.json.

    python3 examples/keychain_tag/build.py <tag-dir> [--render]
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


def cli(*args) -> str:
    return subprocess.check_output([str(CLI), *map(str, args)], text=True)


def sample_lines(project, asset, lines):
    command = ["sample", "-i", project, "--asset", asset, "--json"]
    for line in lines:
        command += ["-l", line]
    return json.loads(cli(*command))


def report_fit(project, params, keys, bodies, work):
    """Interference between the tag envelope and each body, sampled along x
    lines through the tag (and 1.5 mm beyond its sides): none is the fit.
    The lips' overlaps onto the face and back are the retention."""
    length, width, height = (params[keys[k]] for k in ("length", "width", "height"))
    lines = []
    for y in np.linspace(-0.5, 0.5, 15) * (width + 0.003):
        for z in np.linspace(0.0002, height - 0.0002, 12):
            half = 0.5 * length + 0.004
            lines.append(f"{-half},{y},{z} : {half},{y},{z} : 400")
    occupied = lambda rows: np.array([r["occupied"] for r in rows])
    t = occupied(sample_lines(project, "tag_local", lines)["samples"])
    fit = {"points": int(len(t)), "tag_points": int(t.sum())}
    for body, _ in bodies:
        b = occupied(sample_lines(project, body, lines)["samples"])
        fit[f"{body}_interference_points"] = int((t & b).sum())
        print(f"fit: {fit[f'{body}_interference_points']} of {fit['tag_points']} tag points inside the {body}")
    (work / "fit.json").write_text(json.dumps(fit, indent=2) + "\n")
    if any(fit[f"{body}_interference_points"] for body, _ in bodies):
        raise SystemExit("a body cuts into the tag envelope")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("tag", type=Path, help="tag directory (tag.json, keychain.wgsl, work/)")
    ap.add_argument("--render", action="store_true")
    args = ap.parse_args()
    tag = args.tag.resolve()
    work = tag / "work"
    spec = json.loads((tag / "tag.json").read_text())["build"]
    bodies = [tuple(b) for b in spec["bodies"]]
    params = json.loads((work / "parameters.json").read_text())
    frame = json.loads((work / "measurements.json").read_text())["frame"]
    for k, v in params.items():
        print(f"{k:28s} {v * 1000:8.2f} mm")

    project = work / "keychain.vproj"
    cli("project-new", "-o", project)
    select = ["view-select", "-i", work / "survey.vviews", "-p", project, "--embed", "preview",
              "--preview-px", "1600"]
    for view in spec["audit_views"]:
        select += ["--id", view]
    cli(*select)
    cli("project-add-asset", "-p", project, "-i", tag / "keychain.wgsl", "--type", "wgsl",
        "--asset-id", "keychain_source")

    def op(operator, inputs, output, exported=False):
        command = ["project-add-op", "-p", project, "--operator", operator, "--output-id", output]
        for item in inputs:
            command += ["--input", item if isinstance(item, str) else "json:" + json.dumps(item)]
        if not exported:
            command.append("--no-export")
        cli(*command)

    for body, part in bodies:
        op("wgsl_script_operator", ["asset:keychain_source", {**params, "part": part}], body, exported=True)
    op("wgsl_script_operator", ["asset:keychain_source", {**params, "part": 0.0}], "tag_local", exported=True)
    pose = {"rotate": {"rx_deg": 0.0, "ry_deg": 0.0, "rz_deg": frame["yaw_deg"]},
            "translate": {"dx": frame["centre_m"][0], "dy": frame["centre_m"][1], "dz": 0.0}}
    op("pose_operator", ["asset:tag_local", pose], "tag_world", exported=True)
    (work / "project-run.json").write_text(cli("project-run", "-p", project, "--json"))

    report_fit(project, params, spec["fit_keys"], bodies, work)
    for body, _ in bodies:
        for suffix in ("3mf", "stl"):
            print(cli("mesh", "-i", project, "--asset", body, "--unit", "mm", "--base-resolution", "16",
                      "--max-depth", "5", "-q", "-o", work / f"keychain_{body}.{suffix}").strip())
    if args.render:
        for view in spec["audit_views"]:
            print(cli("render", "-i", project, "--asset", "tag_world", "--through", view, "--overlay", "edge",
                      "--up", "0,0,1", "--width", "2400", "--height", "1600", "--resolution", "256",
                      "--grid", "0", "--no-ssao", "-o", work / f"overlay-{view}.png").strip())
        for body, _ in bodies + [("tag_local", 0.0)]:
            print(cli("render", "-i", project, "--asset", body, "--views", "iso,iso-back,top,bottom",
                      "--up", "0,0,1", "--resolution", "256", "--grid", "0", "-o", work / f"{body}.png").strip())
    print(f"Built {project}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
