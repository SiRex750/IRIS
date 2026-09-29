"""MPI-Sintel ground-truth readers.

Index conventions (Sintel names are 1-based):
  frame_{k:04d}.png            image k
  flow/frame_{k:04d}.flo       forward flow  image k -> image k+1, defined on image k's pixels
  occlusions/frame_{k:04d}.png pixels of image k that are occluded in image k+1 (1 per .flo)
  invalid/frame_{k:04d}.png    invalid pixels of image k (1 per image)

In 0-based display order d (d = k-1), the flow d-1 -> d is flow/frame_{d:04d}.flo.
"""
import os

import numpy as np
from PIL import Image

TAG_FLOAT = 202021.25  # "PIEH" read as little-endian float32


def read_flo(path):
    """Read a Middlebury/Sintel .flo file -> float32 array (H, W, 2) = (u, v)."""
    with open(path, "rb") as f:
        tag = np.frombuffer(f.read(4), dtype="<f4")[0]
        if tag != TAG_FLOAT:
            raise ValueError(f"{path}: bad .flo tag {tag!r}")
        w, h = np.frombuffer(f.read(8), dtype="<i4")
        data = np.frombuffer(f.read(int(w) * int(h) * 2 * 4), dtype="<f4")
    if data.size != w * h * 2:
        raise ValueError(f"{path}: truncated ({data.size} floats, expected {w*h*2})")
    return data.reshape(int(h), int(w), 2).astype(np.float32, copy=True)


def read_mask(path):
    """Read a Sintel occlusion/invalid PNG -> bool (H, W), True = occluded/invalid.

    Files on disk are stored 0/255 (README says 0/1); anything > 0 is treated as set.
    """
    return np.asarray(Image.open(path)) > 0


def read_image(path):
    return np.asarray(Image.open(path).convert("RGB"))


class SintelSeq:
    """Accessor for one training sequence, e.g. SintelSeq(root, 'alley_1', 'final')."""

    def __init__(self, root, name, pass_="final"):
        self.root, self.name, self.pass_ = root, name, pass_
        self.img_dir = os.path.join(root, "training", pass_, name)
        self.n_frames = len([f for f in os.listdir(self.img_dir) if f.endswith(".png")])

    def image_path(self, d):
        return os.path.join(self.img_dir, f"frame_{d + 1:04d}.png")

    def image(self, d):
        return read_image(self.image_path(d))

    def flow_into(self, d):
        """GT flow from display frame d-1 to d (defined on frame d-1's pixels)."""
        return read_flo(os.path.join(self.root, "training", "flow", self.name, f"frame_{d:04d}.flo"))

    def bad_mask_into(self, d):
        """Pixels of frame d-1 to exclude for flow d-1 -> d: occluded in d, or invalid in d-1."""
        occ = read_mask(os.path.join(self.root, "training", "occlusions", self.name, f"frame_{d:04d}.png"))
        inv = read_mask(os.path.join(self.root, "training", "invalid", self.name, f"frame_{d:04d}.png"))
        return occ | inv
