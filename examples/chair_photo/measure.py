#!/usr/bin/env python3
"""Replay saved photo observations through volumetric's measurement kernels.

Python orchestrates the CLI and computes Euclidean dimensions; it does not
implement camera solving, projection, distortion, or triangulation.
"""
import argparse
import copy
import json
import subprocess
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def cli(*args):
    return subprocess.check_output(
        [str(ROOT / "target/release/volumetric_cli"), *map(str, args)], text=True
    )


def vector(v):
    return ",".join(str(float(x)) for x in v)


def triangulate(survey, observations):
    args = ["view-triangulate", "-i", survey, "--json"]
    for view, pixel in observations.items():
        args += ["--ray", f"{view}:{vector(pixel)}"]
    return json.loads(cli(*args))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=HERE / "work")
    args = parser.parse_args()
    work = args.work.resolve()
    survey = work / "survey.vviews"
    observations = json.loads((HERE / "observations.json").read_text())
    results = {}
    for name, pixels in observations["features"].items():
        fitting = {v: pixels[v] for v in observations["fit_views"]}
        result = triangulate(survey, fitting)
        point = np.array(result["world"])
        # Sensitivity to one pick coordinate at a time. This is explicitly
        # not an uncertainty bound: camera and systematic errors are absent.
        shifts = []
        for view in fitting:
            for axis in range(2):
                for sign in [-1, 1]:
                    changed = copy.deepcopy(fitting)
                    changed[view][axis] += sign * observations["pixel_uncertainty"]
                    moved = triangulate(survey, changed)["world"]
                    shifts.append(float(np.linalg.norm(np.array(moved) - point) * 1000))
        result["max_single_pick_perturbation_mm"] = max(shifts)
        result["reprojections"] = {}
        for view, observed in pixels.items():
            projected = json.loads(cli("view-pick", "-i", survey, "--view", view,
                                       "--point", vector(point), "--json"))["projected"][0]["pixel"]
            result["reprojections"][view] = {
                "observed": observed, "projected": projected,
                "error_px": float(np.linalg.norm(np.array(projected) - observed)),
                "check_view": view in observations["check_views"],
            }
        results[name] = result

    points = {name: np.array(r["world"]) for name, r in results.items()}
    # Datum from the two inner crossbar holes and the midpoint of the rail
    # slots. Other features retain their measured offsets from this plane.
    origin = (points["cross_a_inner"] + points["cross_b_inner"]) / 2
    x = points["cross_a_inner"] - points["cross_b_inner"]
    x /= np.linalg.norm(x)
    rail_mid = (points["rail_a"] + points["rail_b"]) / 2
    y = rail_mid - origin
    y -= x * np.dot(x, y)
    y /= np.linalg.norm(y)
    z = np.cross(x, y)
    basis = np.array([x, y, z])
    for name, point in points.items():
        results[name]["local_mm"] = ((point - origin) @ basis.T * 1000).tolist()
    dimensions = {
        "cross_outer_centers_mm": float(np.linalg.norm(points["cross_a"] - points["cross_b"]) * 1000),
        "cross_inner_centers_mm": float(np.linalg.norm(points["cross_a_inner"] - points["cross_b_inner"]) * 1000),
        "rail_slot_centers_mm": float(np.linalg.norm(points["rail_a"] - points["rail_b"]) * 1000),
        "rail_to_cross_inner_midpoint_along_y_mm": float(np.dot(rail_mid - origin, y) * 1000),
    }
    outlines = json.loads((HERE / "outlines.json").read_text())

    def on_plane(pixels, height=0):
        call = ["view-pick", "-i", survey, "--view", outlines["view"],
                "--plane-point", vector(origin + z * height),
                "--plane-normal", vector(z), "--json"]
        for pixel in pixels:
            call += ["--pixel", vector(pixel)]
        picked = json.loads(cli(*call))["picked"]
        return [((np.array(p["world"]) - origin) @ basis.T * 1000).tolist() for p in picked]

    outline_coordinates = {"rail_outline": on_plane(outlines["rail_outline"])}
    outline_coordinates["cross_outline"] = (
        on_plane(outlines["cross_outline"][:2], results["cross_a"]["local_mm"][2] / 1000)
        + on_plane(outlines["cross_outline"][2:], results["cross_b"]["local_mm"][2] / 1000)
    )
    apertures = {}
    for name, pairs in outlines["apertures"].items():
        apertures[name] = {}
        for dimension, pixels in pairs.items():
            a, b = on_plane(pixels, results[name]["local_mm"][2] / 1000)
            apertures[name][dimension + "_mm"] = float(np.linalg.norm(np.array(a) - b))
    report = {
        "units": "world and gaps in metres; local coordinates and dimensions in mm",
        "fit_views": observations["fit_views"], "check_views": observations["check_views"],
        "check_note": "Check views are excluded from feature triangulation, but participate in the shared camera survey. Their predicted positions were inspected during picking; this is an additional-view consistency check, not blind validation.",
        "sensitivity_note": "Maximum displacement from perturbing one fit pixel coordinate by +/-3 px. Not an accuracy bound; excludes survey uncertainty and systematic rim-picking bias.",
        "frame": {"origin": origin.tolist(), "basis": basis.tolist()},
        "features": results, "dimensions": dimensions,
        "outline_local_mm": outline_coordinates, "apertures": apertures,
    }
    (work / "measurements.json").write_text(json.dumps(report, indent=2) + "\n")
    for name, r in results.items():
        checks = [v["error_px"] for v in r["reprojections"].values() if v["check_view"]]
        print(f"{name}: gap {max(r['gaps'])*1000:.2f} mm, check-view error {max(checks):.1f} px, local {np.round(r['local_mm'], 2)} mm")
    print(json.dumps(dimensions, indent=2))


if __name__ == "__main__":
    main()
