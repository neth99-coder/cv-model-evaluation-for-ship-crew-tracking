"""
Shared appearance embedding and stable identity helpers for trackers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys

import numpy as np


TORCHREID_MODEL_MAP = {
    "osnet_x0_25": "osnet_x0_25",
    "osnet_x1_0": "osnet_x1_0",
    "resnet50": "resnet50",
    "mlfn": "mlfn",
}

BOXMOT_WEIGHT_FILENAMES = {
    "osnet_x0_25": "osnet_x0_25_msmt17.pt",
    "osnet_x1_0": "osnet_x1_0_msmt17.pt",
    "resnet50": "resnet50_msmt17.pt",
    "mlfn": "mlfn_msmt17.pt",
}

STABLE_ID_MAX_AGE = 90
STABLE_ID_MAX_RELINK_AGE = 20
MIN_BOX_WIDTH_FOR_RELINK = 18.0
MIN_BOX_HEIGHT_FOR_RELINK = 36.0
MIN_BOX_AREA_FOR_RELINK = 900.0
MIN_IOU_FOR_RELINK = 0.10
MAX_CENTER_DIST_FOR_RELINK = 0.55
MAX_PREDICTED_CENTER_DIST = 0.42
OCCLUSION_RECOVERY_MAX_AGE = 10
OCCLUSION_RECOVERY_MAX_PREDICTED_DIST = 0.22
OCCLUSION_RECOVERY_MIN_IOU = 0.02
OCCLUSION_RECOVERY_MIN_HITS = 4
AMBIGUITY_SCORE_GAP = 0.06
AMBIGUITY_PREDICTED_DIST_GAP = 0.05


def install_reid_packages(prefix: str) -> None:
    print(f"[{prefix}] Installing missing Re-ID packages.")
    subprocess.check_call([
        sys.executable,
        "-m",
        "pip",
        "install",
        "torchreid",
        "gdown",
        "tensorboard",
    ])


def normalize_embeddings(embeddings: np.ndarray | None) -> np.ndarray | None:
    if embeddings is None:
        return None
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    return embeddings / norms


class AppearanceEncoder:
    """
    Lazy torchreid feature extractor wrapper with light weight-path resolution.
    """

    def __init__(self, model_name: str, log_prefix: str):
        if model_name not in TORCHREID_MODEL_MAP:
            supported = ", ".join(sorted(TORCHREID_MODEL_MAP))
            raise ValueError(f"Unsupported reid_model '{model_name}'. Supported: {supported}")
        self.model_name = model_name
        self.log_prefix = log_prefix
        self._extractor = None
        self._install_attempted = False

    def extract(self, frame: np.ndarray, boxes: np.ndarray) -> np.ndarray | None:
        if frame is None or boxes is None or len(boxes) == 0:
            return None
        crops = _extract_crops(frame, boxes)
        if not crops:
            return None
        extractor = self._ensure_extractor()
        embeddings = extractor(crops)
        if hasattr(embeddings, "cpu"):
            embeddings = embeddings.cpu().numpy()
        return normalize_embeddings(np.asarray(embeddings, dtype=np.float32))

    def _ensure_extractor(self):
        if self._extractor is not None:
            return self._extractor
        try:
            from torchreid.utils import FeatureExtractor
            device = "cuda" if _cuda_available() else "cpu"
            self._extractor = FeatureExtractor(
                model_name=TORCHREID_MODEL_MAP[self.model_name],
                model_path=self._resolve_model_path(),
                device=device,
            )
            return self._extractor
        except Exception as exc:
            if not self._install_attempted and _is_missing_reid_dependency(exc):
                self._install_attempted = True
                install_reid_packages(self.log_prefix)
                return self._ensure_extractor()
            raise RuntimeError(f"{self.log_prefix} Re-ID encoder init failed: {exc}") from exc

    def _resolve_model_path(self) -> str:
        filename = BOXMOT_WEIGHT_FILENAMES.get(self.model_name)
        if filename:
            try:
                from boxmot.utils import WEIGHTS
                candidate = Path(WEIGHTS) / filename
                if candidate.exists():
                    return str(candidate)
            except Exception:
                pass
        return ""


@dataclass
class StableTrackState:
    bbox: np.ndarray
    prev_bbox: np.ndarray | None
    embedding: np.ndarray | None
    last_seen: int
    hit_count: int


class StableIdentityMemory:
    def __init__(self, max_age: int = STABLE_ID_MAX_AGE):
        self.max_age = max_age
        self.frame_idx = 0
        self.next_stable_id = 1
        self.stable_id_by_internal: dict[int, int] = {}
        self.memory: dict[int, StableTrackState] = {}

    def step(self) -> None:
        self.frame_idx += 1

    def clear_active_mapping(self) -> None:
        self.stable_id_by_internal = {}

    def reassign(
        self,
        tracks: np.ndarray,
        boxes: np.ndarray,
        internal_ids: list[int],
        embeddings: np.ndarray | None,
        stringify_ids: bool = False,
    ) -> np.ndarray:
        self._prune()
        assigned_stable_ids = set()
        current_mapping: dict[int, int] = {}
        stabilized = np.array(tracks, dtype=object, copy=True)

        for idx, internal_id in enumerate(internal_ids):
            stable_id = self.stable_id_by_internal.get(internal_id)
            if stable_id in assigned_stable_ids:
                stable_id = None
            if stable_id is None:
                embedding = embeddings[idx] if embeddings is not None else None
                stable_id = self._match(boxes[idx], embedding, assigned_stable_ids)
            if stable_id is None:
                stable_id = self._preferred_new_stable_id(internal_id, assigned_stable_ids)

            assigned_stable_ids.add(stable_id)
            current_mapping[internal_id] = stable_id
            stabilized[idx, 4] = str(stable_id) if stringify_ids else stable_id
            prev_state = self.memory.get(stable_id)
            self.memory[stable_id] = StableTrackState(
                bbox=np.asarray(boxes[idx], dtype=np.float32),
                prev_bbox=None if prev_state is None else prev_state.bbox.copy(),
                embedding=self._blend_embedding(stable_id, embeddings[idx] if embeddings is not None else None),
                last_seen=self.frame_idx,
                hit_count=1 if prev_state is None else prev_state.hit_count + 1,
            )

        self.stable_id_by_internal = current_mapping
        return stabilized

    def _preferred_new_stable_id(self, internal_id: int, assigned_stable_ids: set[int]) -> int:
        """
        Keep the native tracker id unless we already need that stable id for
        something else. This makes manual re-id a conservative stitching layer
        rather than a second source of ids.
        """
        if internal_id > 0 and internal_id not in assigned_stable_ids:
            self.next_stable_id = max(self.next_stable_id, internal_id + 1)
            return internal_id

        while self.next_stable_id in assigned_stable_ids or self.next_stable_id <= 0:
            self.next_stable_id += 1
        stable_id = self.next_stable_id
        self.next_stable_id += 1
        return stable_id

    def _match(
        self,
        box: np.ndarray,
        embedding: np.ndarray | None,
        assigned_stable_ids: set[int],
    ) -> int | None:
        best_id = None
        best_score = -1.0
        second_best_score = -1.0
        best_predicted_center_dist = None
        second_best_predicted_center_dist = None

        for stable_id, state in self.memory.items():
            if stable_id in assigned_stable_ids:
                continue

            age = self.frame_idx - int(state.last_seen)
            if age <= 0 or age > self.max_age:
                continue
            if age > STABLE_ID_MAX_RELINK_AGE:
                continue

            iou = _bbox_iou(box, state.bbox)
            center_dist = _center_distance_ratio(box, state.bbox)
            predicted_center_dist = _predicted_center_distance_ratio(
                box,
                state.prev_bbox,
                state.bbox,
                age,
            )
            area_sim = _area_similarity(box, state.bbox)
            aspect_sim = _aspect_ratio_similarity(box, state.bbox)
            appearance = _cosine_similarity(embedding, state.embedding)
            box_large_enough = _box_large_enough_for_relink(box)
            appearance_reliable = (
                appearance is not None
                and box_large_enough
                and state.hit_count >= 3
                and age <= 8
            )

            if not box_large_enough and age > 6:
                continue

            if state.hit_count < 2 and age > 3:
                continue

            if predicted_center_dist > MAX_PREDICTED_CENTER_DIST and iou < MIN_IOU_FOR_RELINK:
                continue

            if center_dist > MAX_CENTER_DIST_FOR_RELINK and iou < MIN_IOU_FOR_RELINK:
                continue

            if appearance_reliable and appearance < 0.82:
                continue

            if not appearance_reliable and iou < 0.18 and predicted_center_dist > 0.28:
                continue

            motion_score = max(
                iou,
                1.0 - min(1.0, center_dist),
                1.0 - min(1.0, predicted_center_dist),
            )
            recency = 1.0 - min(1.0, age / max(float(self.max_age), 1.0))
            appearance_score = appearance if appearance_reliable else 0.0
            combined = (
                0.55 * motion_score
                + 0.15 * area_sim
                + 0.15 * aspect_sim
                + 0.10 * recency
                + 0.05 * appearance_score
            )

            if self._is_confident_occlusion_recovery(
                age=age,
                state=state,
                iou=iou,
                area_sim=area_sim,
                aspect_sim=aspect_sim,
                predicted_center_dist=predicted_center_dist,
                center_dist=center_dist,
            ):
                combined = max(combined, 0.93)

            threshold = 0.82 if appearance_reliable else 0.88
            if combined >= threshold and combined > best_score:
                second_best_score = best_score
                second_best_predicted_center_dist = best_predicted_center_dist
                best_score = combined
                best_id = stable_id
                best_predicted_center_dist = predicted_center_dist
            elif combined > second_best_score:
                second_best_score = combined
                second_best_predicted_center_dist = predicted_center_dist

        if best_id is None:
            return None
        if second_best_score >= 0 and (best_score - second_best_score) < AMBIGUITY_SCORE_GAP:
            return None
        if (
            second_best_score >= 0.88
            and best_predicted_center_dist is not None
            and second_best_predicted_center_dist is not None
            and abs(best_predicted_center_dist - second_best_predicted_center_dist) < AMBIGUITY_PREDICTED_DIST_GAP
        ):
            return None
        return best_id

    def _is_confident_occlusion_recovery(
        self,
        age: int,
        state: StableTrackState,
        iou: float,
        area_sim: float,
        aspect_sim: float,
        predicted_center_dist: float,
        center_dist: float,
    ) -> bool:
        """
        Short full occlusions often produce a brand new native id. Allow a
        motion-first relink when the previous track was mature and the new box
        reappears exactly where that person should emerge.
        """
        return bool(
            state.hit_count >= OCCLUSION_RECOVERY_MIN_HITS
            and age <= OCCLUSION_RECOVERY_MAX_AGE
            and predicted_center_dist <= OCCLUSION_RECOVERY_MAX_PREDICTED_DIST
            and center_dist <= 0.35
            and iou >= OCCLUSION_RECOVERY_MIN_IOU
            and area_sim >= 0.55
            and aspect_sim >= 0.6
        )

    def _blend_embedding(self, stable_id: int, embedding: np.ndarray | None) -> np.ndarray | None:
        if embedding is None:
            prev = self.memory.get(stable_id)
            return None if prev is None else prev.embedding
        prev = self.memory.get(stable_id)
        if prev is None or prev.embedding is None:
            return np.asarray(embedding, dtype=np.float32)
        blended = (0.75 * prev.embedding) + (0.25 * embedding)
        return normalize_embeddings(blended)[0]

    def _prune(self) -> None:
        expired = [
            stable_id
            for stable_id, state in self.memory.items()
            if self.frame_idx - int(state.last_seen) > self.max_age
        ]
        for stable_id in expired:
            self.memory.pop(stable_id, None)


def _extract_crops(frame: np.ndarray, boxes: np.ndarray) -> list[np.ndarray]:
    height, width = frame.shape[:2]
    crops: list[np.ndarray] = []
    for box in np.asarray(boxes, dtype=np.float32):
        x1, y1, x2, y2 = box[:4]
        x1 = int(np.clip(np.floor(x1), 0, max(width - 1, 0)))
        y1 = int(np.clip(np.floor(y1), 0, max(height - 1, 0)))
        x2 = int(np.clip(np.ceil(x2), x1 + 1, width))
        y2 = int(np.clip(np.ceil(y2), y1 + 1, height))
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            crop = np.zeros((64, 32, 3), dtype=np.uint8)
        crops.append(crop)
    return crops


def _cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _is_missing_reid_dependency(err: Exception) -> bool:
    msg = str(err).lower()
    if isinstance(err, ModuleNotFoundError):
        missing = (getattr(err, "name", "") or "").lower()
        if any(token in missing for token in ("torchreid", "gdown", "tensorboard")):
            return True
    return any(token in msg for token in ("torchreid", "gdown", "tensorboard"))


def _cosine_similarity(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    if a is None or b is None:
        return None
    return float(np.dot(a, b))


def _area_similarity(box_a: np.ndarray, box_b: np.ndarray) -> float:
    area_a = max(float(box_a[2] - box_a[0]), 1.0) * max(float(box_a[3] - box_a[1]), 1.0)
    area_b = max(float(box_b[2] - box_b[0]), 1.0) * max(float(box_b[3] - box_b[1]), 1.0)
    return min(area_a, area_b) / max(area_a, area_b)


def _aspect_ratio_similarity(box_a: np.ndarray, box_b: np.ndarray) -> float:
    wa = max(float(box_a[2] - box_a[0]), 1.0)
    ha = max(float(box_a[3] - box_a[1]), 1.0)
    wb = max(float(box_b[2] - box_b[0]), 1.0)
    hb = max(float(box_b[3] - box_b[1]), 1.0)
    ra = wa / ha
    rb = wb / hb
    return min(ra, rb) / max(ra, rb)


def _bbox_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area_a = max(0.0, float(box_a[2] - box_a[0])) * max(0.0, float(box_a[3] - box_a[1]))
    area_b = max(0.0, float(box_b[2] - box_b[0])) * max(0.0, float(box_b[3] - box_b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _center_distance_ratio(box_a: np.ndarray, box_b: np.ndarray) -> float:
    ax = float(box_a[0] + box_a[2]) / 2.0
    ay = float(box_a[1] + box_a[3]) / 2.0
    bx = float(box_b[0] + box_b[2]) / 2.0
    by = float(box_b[1] + box_b[3]) / 2.0
    dist = float(np.hypot(ax - bx, ay - by))

    aw = max(float(box_a[2] - box_a[0]), 1.0)
    ah = max(float(box_a[3] - box_a[1]), 1.0)
    bw = max(float(box_b[2] - box_b[0]), 1.0)
    bh = max(float(box_b[3] - box_b[1]), 1.0)
    scale = max(np.hypot(aw, ah), np.hypot(bw, bh), 1.0)
    return dist / scale


def _predicted_center_distance_ratio(
    box: np.ndarray,
    prev_box: np.ndarray | None,
    current_box: np.ndarray,
    age: int,
) -> float:
    if prev_box is None:
        return _center_distance_ratio(box, current_box)

    prev_center = _box_center(prev_box)
    current_center = _box_center(current_box)
    velocity = current_center - prev_center
    horizon = float(min(max(age, 1), 3))
    predicted_center = current_center + velocity * horizon
    candidate_center = _box_center(box)
    dist = float(np.hypot(*(candidate_center - predicted_center)))

    cw = max(float(current_box[2] - current_box[0]), 1.0)
    ch = max(float(current_box[3] - current_box[1]), 1.0)
    bw = max(float(box[2] - box[0]), 1.0)
    bh = max(float(box[3] - box[1]), 1.0)
    scale = max(np.hypot(cw, ch), np.hypot(bw, bh), 1.0)
    return dist / scale


def _box_center(box: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            (float(box[0]) + float(box[2])) / 2.0,
            (float(box[1]) + float(box[3])) / 2.0,
        ],
        dtype=np.float32,
    )


def _box_large_enough_for_relink(box: np.ndarray) -> bool:
    width = max(float(box[2] - box[0]), 0.0)
    height = max(float(box[3] - box[1]), 0.0)
    area = width * height
    return (
        width >= MIN_BOX_WIDTH_FOR_RELINK
        and height >= MIN_BOX_HEIGHT_FOR_RELINK
        and area >= MIN_BOX_AREA_FOR_RELINK
    )
