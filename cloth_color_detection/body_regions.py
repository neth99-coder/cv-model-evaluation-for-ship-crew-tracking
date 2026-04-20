"""
body_regions.py
───────────────
Split a person mask into upper-body (top/jacket) and lower-body (trousers/skirt)
so we can report separate colours for shirt vs pants.

Strategy: simple vertical split at a configurable fraction of the bbox height.
This is deliberately lightweight — the segmenter already did the hard work.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class BodyRegions:
    upper: np.ndarray   # uint8 mask H×W
    lower: np.ndarray   # uint8 mask H×W
    head:  Optional[np.ndarray] = None   # optional head exclusion mask


def split_body(
    mask: np.ndarray,
    bbox: Tuple[int, int, int, int],
    upper_fraction:  float = 0.50,   # top % of person bbox = "upper body"
    head_fraction:   float = 0.18,   # top % excluded as head
    exclude_head:    bool  = True,
) -> BodyRegions:
    """
    Split mask into upper / lower body regions.

    Parameters
    ----------
    upper_fraction : fraction of person height that counts as "upper body"
                     (below head exclusion)
    head_fraction  : fraction of person height at the top treated as head/hair
    """
    H, W = mask.shape[:2]
    x1, y1, x2, y2 = bbox
    person_h = y2 - y1
    if person_h <= 0:
        empty = np.zeros((H, W), dtype=np.uint8)
        return BodyRegions(upper=empty, lower=empty)

    head_cut  = y1 + int(person_h * head_fraction)
    upper_cut = y1 + int(person_h * upper_fraction)

    # Build region masks
    region_upper = np.zeros((H, W), dtype=np.uint8)
    region_lower = np.zeros((H, W), dtype=np.uint8)
    region_head  = np.zeros((H, W), dtype=np.uint8)

    region_head [y1:head_cut,  x1:x2] = 255
    region_upper[head_cut:upper_cut, x1:x2] = 255
    region_lower[upper_cut:y2, x1:x2] = 255

    # Intersect with actual segmentation mask
    upper_mask = cv2.bitwise_and(mask, region_upper)
    lower_mask = cv2.bitwise_and(mask, region_lower)
    head_mask  = cv2.bitwise_and(mask, region_head)

    if exclude_head:
        # Remove head pixels from upper mask just in case bbox is tight
        upper_mask = cv2.bitwise_and(upper_mask, cv2.bitwise_not(head_mask))

    return BodyRegions(upper=upper_mask, lower=lower_mask, head=head_mask)


def morphological_refine(mask: np.ndarray, ksize: int = 5) -> np.ndarray:
    """Light morphological cleanup on a binary mask."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
    return mask
