#!/usr/bin/env python3
"""Build the two upper parts and photo audit from replayed measurements."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from measure import HERE, cli, vector


def add_operator(project, operator, inputs, output, exported=False):
    command = ["project-add-op", "-p", project, "--operator", operator, "--output-id", output]
    for item in inputs:
        command += ["--input", item if isinstance(item, str) else "json:" + json.dumps(item)]
    if not exported:
        command += ["--no-export"]
    cli(*command)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=HERE / "work")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    work = args.work.resolve()
    report = json.loads((work / "measurements.json").read_text())
    params = {"assumed_thickness": 0.0025, "assumed_bend_start": 0.083,
              "assumed_bend_end": 0.094, "assumed_rail_start": 0.014}
    for name, feature in report["features"].items():
        for axis, value in zip("xyz", feature["local_mm"]):
            if axis == "z" and name not in {"cross_a", "cross_b"}:
                continue  # remaining surfaces use the documented common datum
            params[f"{name}_{axis}"] = value / 1000
    cross = np.array(report["outline_local_mm"]["cross_outline"]) / 1000
    rail = np.array(report["outline_local_mm"]["rail_outline"]) / 1000
    params.update({
        "cross_xmin": float(np.mean(cross[[2, 3], 0])),
        "cross_xmax": float(np.mean(cross[[0, 1], 0])),
        "cross_ymin": float(np.mean(cross[[1, 2], 1])),
        "cross_ymax": float(np.mean(cross[[0, 3], 1])),
        "rail_end": float(np.mean(rail[[0, 3], 1])),
        "rail_front_halfwidth": float(np.mean(np.abs(rail[[1, 2], 0]))),
        "rail_back_halfwidth": float(np.mean(np.abs(rail[[0, 3], 0]))),
    })
    apertures = report["apertures"]
    for prefix, names in [("cross", ["cross_a", "cross_b"]), ("rail", ["rail_a", "rail_b"])]:
        for target, source in [("length", "long_mm"), ("width", "wide_mm")]:
            params[f"{prefix}_slot_{target}"] = float(np.mean([apertures[n][source] for n in names]) / 1000)
    params["cross_hole_diameter"] = float(np.mean([apertures[n]["wide_mm"] for n in ["cross_a_inner", "cross_b_inner"]]) / 1000)
    (work / "parameters.json").write_text(json.dumps(params, indent=2) + "\n")

    project = work / "mount.vproj"
    cli("project-new", "-o", project)
    select = ["view-select", "-i", work / "survey.vviews", "-p", project,
              "--embed", "preview", "--preview-px", "1600"]
    for view in report["fit_views"] + report["check_views"]:
        select += ["--id", view]
    cli(*select)
    cli("project-add-asset", "-p", project, "-i", HERE / "mount.wgsl", "--type", "wgsl", "--asset-id", "mount_source")

    def op(operator, inputs, output, exported=False):
        add_operator(project, operator, inputs, output, exported)

    frame = report["frame"]
    op("subspace_operator", [{"kind": "frame"}, frame["origin"], *frame["basis"][:2]], "mount_frame")
    angles = Rotation.from_matrix(np.array(frame["basis"]).T).as_euler("xyz", degrees=True)
    pose = {"rotate": dict(zip(["rx_deg", "ry_deg", "rz_deg"], angles.tolist())),
            "translate": dict(zip(["dx", "dy", "dz"], frame["origin"]))}
    for index, name in enumerate(["crossbar", "rail"]):
        op("wgsl_script_operator", ["asset:mount_source", {**params, "part": float(index)}], name + "_local")
        op("pose_operator", ["asset:" + name + "_local", pose], name, exported=True)
    run = cli("project-run", "-p", project, "--json")
    (work / "project-run.json").write_text(run)

    # The geometry rendered here is simplified. Feature crosses provide a
    # separate check of the actual measurements, independent of the mesh.
    observations = json.loads((HERE / "observations.json").read_text())
    for view in report["fit_views"] + report["check_views"]:
        coords = np.array([v[view] for v in observations["features"].values()])
        center = (coords.min(axis=0) + coords.max(axis=0)) / 2
        size = np.ceil(coords.max(axis=0) - coords.min(axis=0) + 240).astype(int)
        command = ["view-crop", "-i", work / "survey.vviews", "--view", view,
                   "--center", vector(center), "--size", f"{size[0]}x{size[1]}",
                   "--scale", "1", "--grid", "100", "-o", work / f"check-{view}.png"]
        for feature in report["features"].values():
            command += ["--mark-world", vector(feature["world"])]
        for pixel in coords:
            command += ["--mark", vector(pixel)]
        cli(*command)
        if args.render:
            print(cli("render", "-i", project, "--through", view, "--overlay", "edge",
                      "--up", "0,0,1",
                      "--width", "2400", "--height", "1600", "--resolution", "256",
                      "--grid", "0", "--no-ssao", "-o", work / f"overlay-{view}.png"))
    if args.render:
        print(cli("render", "-i", project, "--up", "0,0,1", "--views", "iso,top",
                  "--resolution", "256", "--grid", "0", "--no-ssao", "-o", work / "mount.png"))
    print(f"Built {project}")


if __name__ == "__main__":
    main()
