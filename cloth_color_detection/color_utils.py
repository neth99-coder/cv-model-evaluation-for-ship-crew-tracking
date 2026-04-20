"""
color_utils.py
──────────────
HSV-aware color clustering and base-color mapping.

Key ideas:
    - Cluster in HS space (drop V) so shadows don't drag colors to black.
    - Map clusters to broad color names.
    - Merge same-name clusters so the final dominant color is stable.
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import KDTree
from dataclasses import dataclass
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")


# ─── Base colour palette (broad classes only) ───────────────────────────────
# (name, RGB)
CLOTHING_COLORS: List[Tuple[str, Tuple[int, int, int]]] = [
    ("Black",  (10, 10, 10)),
    ("White",  (245, 245, 245)),
    ("Gray",   (128, 128, 128)),
    ("Red",    (220, 30, 30)),
    ("Orange", (245, 140, 30)),
    ("Yellow", (240, 220, 30)),
    ("Green",  (40, 170, 40)),
    ("Blue",   (40, 90, 210)),
    ("Purple", (140, 70, 180)),
    ("Pink",   (235, 130, 180)),
    ("Brown",  (130, 80, 45)),
]

# Build KD-Tree once at import time
_color_names  = [c[0] for c in CLOTHING_COLORS]
_color_rgb    = np.array([c[1] for c in CLOTHING_COLORS], dtype=np.float32)
_color_tree   = KDTree(_color_rgb)


# ─── Dataclass for one detected garment ─────────────────────────────────────

@dataclass
class GarmentColor:
    name:        str
    rgb:         Tuple[int, int, int]
    hex_code:    str
    proportion:  float          # fraction of the mask covered by this cluster
    confidence:  float          # 1 – normalised distance to nearest named colour


# ─── Public API ──────────────────────────────────────────────────────────────

def extract_cloth_colors(
    frame:       np.ndarray,
    mask:        np.ndarray,
    n_colors:    int = 3,
    min_pixels:  int = 200,
    use_hs_only: bool = True,
) -> List[GarmentColor]:
    """
    Given a BGR frame and a binary mask (uint8, 0/255), return the dominant
    clothing colours sorted by prevalence.

    Parameters
    ----------
    use_hs_only : bool
        If True, cluster in HS space (ignore the V=brightness channel).
        This prevents shadows from mis-classifying coloured garments as black.
    """
    roi_pixels = _get_roi_pixels(frame, mask)
    if roi_pixels is None or len(roi_pixels) < min_pixels:
        return []

    # Filter near-black pixels caused by occlusion / sensor noise
    roi_pixels = _filter_noise(roi_pixels)
    if len(roi_pixels) < min_pixels:
        return []

    n_colors = min(n_colors, max(1, len(roi_pixels) // 50))

    if use_hs_only:
        features = _to_hs(roi_pixels)
    else:
        features = roi_pixels.astype(np.float32)

    labels, centers_feature = _kmeans_cluster(features, n_colors)

    # Map cluster centers back to RGB for naming
    if use_hs_only:
        centers_rgb = _hs_centers_to_rgb(roi_pixels, labels, n_colors)
    else:
        centers_rgb = centers_feature.astype(int)

    results: List[GarmentColor] = []
    total   = len(labels)

    for k in range(n_colors):
        proportion = float(np.sum(labels == k)) / total
        if proportion < 0.05:          # skip tiny clusters (< 5 %)
            continue

        rgb       = tuple(int(v) for v in centers_rgb[k])
        name, conf = _name_color(rgb)
        results.append(GarmentColor(
            name       = name,
            rgb        = rgb,
            hex_code   = "#{:02X}{:02X}{:02X}".format(*rgb),
            proportion = round(proportion, 3),
            confidence = round(conf, 3),
        ))

    # Merge clusters mapped to the same base color (e.g., two blue shades).
    results = _merge_same_color_clusters(results)
    results.sort(key=lambda c: c.proportion, reverse=True)
    return results


def name_rgb(rgb: Tuple[int, int, int]) -> str:
    """Quick one-shot RGB → name lookup."""
    return _name_color(rgb)[0]


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _get_roi_pixels(frame: np.ndarray, mask: np.ndarray) -> Optional[np.ndarray]:
    if mask is None or mask.sum() == 0:
        return None
    bool_mask = mask > 127
    pixels    = frame[bool_mask]          # shape (N, 3) BGR
    rgb       = pixels[:, ::-1].astype(np.float32)   # → RGB
    return rgb


def _filter_noise(pixels: np.ndarray, brightness_floor: int = 15) -> np.ndarray:
    """Remove near-black pixels that are almost certainly shadow / occlusion."""
    bright = np.max(pixels, axis=1) > brightness_floor
    return pixels[bright]


def _to_hs(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB (float32) → H, S (ignore V) for shadow-invariant clustering."""
    bgr   = rgb[:, ::-1].astype(np.uint8)
    hsv   = cv2.cvtColor(bgr[np.newaxis], cv2.COLOR_BGR2HSV)[0]
    hs    = hsv[:, :2].astype(np.float32)
    # Circular H: encode as sin/cos so 0° and 179° aren't far apart
    h_rad = hs[:, 0] * (np.pi / 90.0)    # 0–180 → 0–2π
    hs_enc = np.stack([
        np.sin(h_rad) * hs[:, 1],
        np.cos(h_rad) * hs[:, 1],
        hs[:, 1],
    ], axis=1)
    return hs_enc


def _kmeans_cluster(features: np.ndarray, n: int):
    km     = KMeans(n_clusters=n, n_init=5, max_iter=100, random_state=42)
    labels = km.fit_predict(features)
    return labels, km.cluster_centers_


def _hs_centers_to_rgb(
    rgb_pixels: np.ndarray,
    labels:     np.ndarray,
    n:          int,
) -> np.ndarray:
    """Compute the mean RGB per cluster (ignores brightness floor already done)."""
    centers = np.zeros((n, 3), dtype=np.float32)
    for k in range(n):
        mask = labels == k
        if mask.sum() > 0:
            centers[k] = rgb_pixels[mask].mean(axis=0)
    return centers.astype(int)


def _name_color(rgb: Tuple[int, int, int]) -> Tuple[str, float]:
    # Classify into broad buckets using HSV rules.
    hsv = cv2.cvtColor(np.uint8([[list(rgb[::-1])]]), cv2.COLOR_BGR2HSV)[0, 0]
    h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])

    # Neutral handling first.
    if v < 35 and s < 90:
        name = "Black"
    elif s < 22 and v > 210:
        name = "White"
    elif s < 35:
        name = "Gray"
    # Brown is low-value orange in clothing context.
    elif 8 <= h <= 25 and 40 <= v <= 190 and s >= 35:
        name = "Brown"
    elif h < 10 or h >= 170:
        name = "Red"
    elif h < 22:
        name = "Orange"
    elif h < 36:
        name = "Yellow"
    elif h < 86:
        name = "Green"
    elif h < 132:
        name = "Blue"
    elif h < 160:
        name = "Purple"
    else:
        name = "Pink"

    # Confidence from perceptual-ish distance to chosen base color.
    idx = _color_names.index(name)
    dist = np.linalg.norm(np.array(rgb, dtype=np.float32) - _color_rgb[idx])
    max_dist = np.sqrt(3) * 255
    confidence = 1.0 - float(dist) / max_dist

    # Slight penalty when chroma is very low but we still mapped to a chromatic class.
    if name not in {"Black", "White", "Gray"} and s < 45:
        confidence *= 0.85

    return name, max(0.0, min(1.0, confidence))


def _merge_same_color_clusters(colors: List[GarmentColor]) -> List[GarmentColor]:
    if not colors:
        return []

    merged: dict[str, dict] = {}
    for c in colors:
        if c.name not in merged:
            merged[c.name] = {
                "prop": 0.0,
                "rgb_sum": np.zeros(3, dtype=np.float64),
                "conf_sum": 0.0,
            }

        w = float(c.proportion)
        merged[c.name]["prop"] += w
        merged[c.name]["rgb_sum"] += np.array(c.rgb, dtype=np.float64) * w
        merged[c.name]["conf_sum"] += float(c.confidence) * w

    out: List[GarmentColor] = []
    for name, item in merged.items():
        prop = item["prop"]
        if prop <= 0:
            continue
        rgb = np.clip(item["rgb_sum"] / prop, 0, 255).astype(int)
        conf = item["conf_sum"] / prop
        rgb_t = (int(rgb[0]), int(rgb[1]), int(rgb[2]))
        out.append(
            GarmentColor(
                name=name,
                rgb=rgb_t,
                hex_code="#{:02X}{:02X}{:02X}".format(*rgb_t),
                proportion=round(float(prop), 3),
                confidence=round(float(conf), 3),
            )
        )
    return out
