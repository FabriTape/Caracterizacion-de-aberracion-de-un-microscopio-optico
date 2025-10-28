from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Literal, Tuple, Dict, Any

import numpy as np
import pandas as pd

# edge detection is injected in measure_roi_widths to avoid hard dependency during tests


Orientation = Literal["X", "Y"]


@dataclass
class ROI:
    x: int
    y: int
    width: int
    height: int

    @staticmethod
    def from_series(s: pd.Series) -> "ROI":
        return ROI(int(s["X"]), int(s["Y"]), int(s["Width"]), int(s["Height"]))

    def slice(self) -> Tuple[slice, slice]:
        return (slice(self.y, self.y + self.height), slice(self.x, self.x + self.width))


def compute_distances_from_edges(edges: np.ndarray, orient: Orientation) -> List[float]:
    """Compute distances between the first and last edge for each scanline.

    orient="X": scan along columns (for horizontal extents)
    orient="Y": scan along rows (for vertical extents)
    """
    if edges.ndim != 2:
        raise ValueError("edges must be 2D array")

    distances: List[float] = []
    h, w = edges.shape
    if orient == "X":
        for j in range(w):
            col = edges[:, j]
            idx = np.flatnonzero(col)
            if idx.size >= 2:
                distances.append(float(idx[-1] - idx[0]))
    else:  # orient == "Y"
        for i in range(h):
            row = edges[i, :]
            idx = np.flatnonzero(row)
            if idx.size >= 2:
                distances.append(float(idx[-1] - idx[0]))
    return distances


def measure_roi_widths(
    image: np.ndarray,
    rois_df: pd.DataFrame,
    orient: Orientation,
    *,
    edge_detector: Callable[[np.ndarray, Tuple[int, int], int, int], np.ndarray] | None = None,
    blur_kernel: Tuple[int, int] = (3, 3),
    canny_low: int = 100,
    canny_high: int = 131,
) -> List[Dict[str, Any]]:
    """Measure widths inside each ROI using Canny edges and return stats.

    Returns a list of dicts with keys: roi_index, mean, std, n.
    """
    if not {"X", "Y", "Width", "Height"}.issubset(rois_df.columns):
        raise ValueError("rois_df must have columns: X, Y, Width, Height")

    results: List[Dict[str, Any]] = []
    for i, row in rois_df.iterrows():
        roi = ROI.from_series(row)
        subimg = image[roi.slice()]
        if edge_detector is None:
            try:
                from .image_utils import detect_edges as _detect
            except Exception as e:
                raise ImportError(
                    "edge_detector not provided and image_utils.detect_edges not available"
                ) from e
            edges = _detect(subimg, blur_kernel=blur_kernel, canny_low=canny_low, canny_high=canny_high)
        else:
            edges = edge_detector(subimg, blur_kernel, canny_low, canny_high)
        dists = compute_distances_from_edges(edges, orient=orient)
        if len(dists) == 0:
            mean = float("nan")
            std = float("nan")
        else:
            arr = np.asarray(dists, dtype=float)
            mean = float(np.mean(arr))
            std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
        results.append({
            "roi_index": int(i),
            "mean": mean,
            "std": std,
            "n": int(len(dists)),
        })
    return results
