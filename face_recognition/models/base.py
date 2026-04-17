import time
from abc import ABC, abstractmethod
from typing import Any

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Normalised cosine similarity in [-1, 1]."""
    a_n = a / (np.linalg.norm(a) + 1e-8)
    b_n = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a_n, b_n))


class BaseFaceRecognizer(ABC):
    """
    Abstract base class for all face recognition models.

    Subclasses must implement :meth:`get_embedding`.
    For multi-face images, override :meth:`get_all_embeddings`.
    Override :meth:`enroll` / :meth:`recognize_all` for REST-based models (e.g. CompreFace).
    """

    model_name: str = "Base"
    threshold: float = 0.35  # cosine similarity cut-off

    def __init__(self) -> None:
        self.database: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------
    @abstractmethod
    def get_embedding(self, image_path: str) -> np.ndarray | None:
        """Return a (normalised) embedding for the *largest* face, or None."""

    def get_all_embeddings(self, image_path: str) -> list[np.ndarray]:
        """
        Return a list of (normalised) embeddings – one per detected face.

        Default: delegates to :meth:`get_embedding` (single face).
        InsightFace subclasses override this to return all detected faces.
        """
        emb = self.get_embedding(image_path)
        return [emb] if emb is not None else []

    def detect_faces(self, image_path: str) -> list[dict[str, Any]]:
        """
        Return detected faces with embeddings and optional bounding boxes.

        Each item is ``{"embedding": np.ndarray, "bbox": (x1, y1, x2, y2) | None}``.
        Default implementation uses embeddings only (no bounding boxes).
        """
        return [{"embedding": emb, "bbox": None} for emb in self.get_all_embeddings(image_path)]

    def detect_bboxes(self, image_path: str) -> list[tuple[int, int, int, int]]:
        """Return face bounding boxes for annotation-only fallbacks."""
        boxes: list[tuple[int, int, int, int]] = []
        for face in self.detect_faces(image_path):
            bbox = face.get("bbox")
            if bbox is None:
                continue
            x1, y1, x2, y2 = [int(v) for v in bbox]
            boxes.append((x1, y1, x2, y2))
        return boxes

    # ------------------------------------------------------------------
    # High-level API
    # ------------------------------------------------------------------
    def enroll(self, person_name: str, image_path: str) -> bool:
        """Store the embedding for *person_name* extracted from *image_path*."""
        embedding = self.get_embedding(image_path)
        if embedding is not None:
            self.database[person_name] = embedding
            return True
        print(f"  [WARN] No face detected in enrollment image: {image_path}")
        return False

    def _match_embedding(self, embedding: np.ndarray) -> tuple[str, float]:
        """Match a single embedding against the database. Returns (name, score)."""
        scores = {
            name: cosine_similarity(embedding, db_emb)
            for name, db_emb in self.database.items()
        }
        best_name = max(scores, key=scores.__getitem__)
        best_score = scores[best_name]
        if best_score < self.threshold:
            return "unknown", best_score
        return best_name, best_score

    def recognize_faces(self, image_path: str) -> tuple[list[dict[str, Any]], float]:
        """
        Identify all detected faces and keep optional bounding boxes.

        Returns
        -------
        (detections, inference_time_ms)
        detections is a list of dicts with ``name``, ``confidence``, and ``bbox``.
        """
        t0 = time.perf_counter()
        faces = self.detect_faces(image_path)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if not faces or not self.database:
            return [], elapsed_ms

        detections: list[dict[str, Any]] = []
        known_name_to_index: dict[str, int] = {}

        for face in faces:
            emb = face.get("embedding")
            if emb is None:
                continue

            name, score = self._match_embedding(emb)
            bbox = face.get("bbox")

            if name == "unknown":
                detections.append({
                    "name": "unknown",
                    "confidence": float(score),
                    "bbox": bbox,
                })
                continue

            if name in known_name_to_index:
                idx = known_name_to_index[name]
                if score > detections[idx]["confidence"]:
                    detections[idx] = {
                        "name": name,
                        "confidence": float(score),
                        "bbox": bbox,
                    }
            else:
                known_name_to_index[name] = len(detections)
                detections.append({
                    "name": name,
                    "confidence": float(score),
                    "bbox": bbox,
                })

        return detections, elapsed_ms

    def recognize_all(self, image_path: str) -> tuple[list[tuple[str, float]], float]:
        """
        Identify **all** people visible in *image_path*.

        Returns
        -------
        (matches, inference_time_ms)
        matches is a list of (name, confidence) for every detected face.
        Faces below the threshold are labelled ``"unknown"``.
        Each enrolled person is only returned once (highest-confidence match
        wins when the same person appears multiple times).
        """
        detections, elapsed_ms = self.recognize_faces(image_path)
        matches = [
            (d["name"], float(d["confidence"]))
            for d in detections
        ]
        return matches, elapsed_ms

    def clear_database(self) -> None:
        self.database.clear()
