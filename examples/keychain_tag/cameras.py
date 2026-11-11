"""The survey's cameras in Python: project world points into the developed
photographs exactly as volumetric's CameraModel does (OpenCV pinhole with
radial distortion; camera_to_world rows hold the camera axes as columns and
the eye as the last column), and sample the pictures there."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CLI = ROOT / "target/release/volumetric_cli"


class View:
    def __init__(self, view: dict, camera: dict, photos: Path):
        m = np.array(view["camera_to_world"], float).reshape(3, 4)
        self.id = view["id"]
        self.r = m[:, :3].T  # world to camera
        self.eye = m[:, 3]
        self.f = np.array([camera["fx"], camera["fy"]])
        self.c = np.array([camera["cx"], camera["cy"]])
        self.k = (camera.get("k") or []) + [0.0, 0.0, 0.0]
        self.photo = photos / f"{self.id}.JPG"
        self._gray = None

    def project(self, pts: np.ndarray) -> np.ndarray:
        """World points (..., 3) to pixels (..., 2); NaN behind the camera."""
        pc = (pts - self.eye) @ self.r.T
        z = pc[..., 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            x, y = pc[..., 0] / z, pc[..., 1] / z
        r2 = x * x + y * y
        d = 1 + self.k[0] * r2 + self.k[1] * r2 * r2 + self.k[2] * r2 ** 3
        uv = np.stack([self.f[0] * x * d + self.c[0], self.f[1] * y * d + self.c[1]], -1)
        uv[z <= 0] = np.nan
        return uv

    def gray(self) -> np.ndarray:
        if self._gray is None:
            self._gray = cv2.cvtColor(cv2.imread(str(self.photo)), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        return self._gray

    def sample(self, pts: np.ndarray, image: np.ndarray | None = None) -> np.ndarray:
        """Bilinear samples of the picture (or `image`, same size) at world points."""
        shape = pts.shape[:-1]
        uv = self.project(pts).reshape(-1, 2).astype(np.float32)
        # remap wants 2-D maps under 32767 a side: rows of 4096.
        n = len(uv)
        uv = np.concatenate([uv, np.full((-n % 4096, 2), np.nan, np.float32)]).reshape(-1, 4096, 2)
        img = self.gray() if image is None else image
        out = cv2.remap(img, uv[..., 0], uv[..., 1], cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
        return out.reshape(-1)[:n].reshape(shape)


def load(work: Path) -> dict[str, View]:
    """The posed views of a tag's work directory."""
    cams = json.loads((work / "cameras.json").read_text())
    if not cams.get("cameras") or "camera_to_world" not in cams["views"][0]:
        raise SystemExit(f"{work / 'cameras.json'}: run survey.sh")
    out = {}
    for v in cams["views"]:
        if v.get("camera_to_world"):
            out[v["id"]] = View(v, cams["cameras"][v["camera"]], work / "photos")
    return out


def board(work: Path) -> dict:
    return json.loads((work / "cameras.json").read_text())["board"]


class Card:
    """The nominal ChArUco card, sampled at world points on its plane:
    1 white, 0 black. World x is the board's x, world y its -y."""

    def __init__(self, board: dict, px_per_square: int = 400):
        spec = board["spec"]
        self.spec = spec
        self.sq = px_per_square
        b = cv2.aruco.CharucoBoard(
            (spec["squares_x"], spec["squares_y"]), 1.0, spec["marker_m"] / spec["pitch_y_m"],
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11),
            np.arange(spec["first_id"], spec["first_id"] + spec["squares_x"] * spec["squares_y"] // 2))
        self.image = b.generateImage((spec["squares_x"] * self.sq, spec["squares_y"] * self.sq),
                                     marginSize=0).astype(np.float32) / 255.0

    def sample(self, pts: np.ndarray) -> np.ndarray:
        shape = pts.shape[:-1]
        x = (pts[..., 0] / self.spec["pitch_x_m"] * self.sq).reshape(-1)
        y = (-pts[..., 1] / self.spec["pitch_y_m"] * self.sq).reshape(-1)
        n = len(x)
        pad = -n % 4096
        mx = np.concatenate([x, np.zeros(pad)]).reshape(-1, 4096).astype(np.float32)
        my = np.concatenate([y, np.zeros(pad)]).reshape(-1, 4096).astype(np.float32)
        # Off the printed squares the card is its white margin.
        out = cv2.remap(self.image, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=1.0)
        return out.reshape(-1)[:n].reshape(shape)
