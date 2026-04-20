import time
from abc import ABC, abstractmethod
from typing import Any

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_n = a / (np.linalg.norm(a) + 1e-8)
    b_n = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a_n, b_n))


class BaseFaceRecognizer(ABC):
    model_name: str = "Base"
    threshold: float = 0.35

    def __init__(self) -> None:
        self.database: dict[str, np.ndarray] = {}

    @abstractmethod
    def get_embedding(self, image_path: str) -> np.ndarray | None:
        """Return a normalized embedding for the largest face, or None."""

    def get_all_embeddings(self, image_path: str) -> list[np.ndarray]:
        emb = self.get_embedding(image_path)
        return [emb] if emb is not None else []

    def detect_faces(self, image_path: str) -> list[dict[str, Any]]:
        return [{"embedding": emb, "bbox": None} for emb in self.get_all_embeddings(image_path)]

    def detect_bboxes(self, image_path: str) -> list[tuple[int, int, int, int]]:
        boxes: list[tuple[int, int, int, int]] = []
        for face in self.detect_faces(image_path):
            bbox = face.get("bbox")
            if bbox is None:
                continue
            x1, y1, x2, y2 = [int(v) for v in bbox]
            boxes.append((x1, y1, x2, y2))
        return boxes

    def enroll(self, person_name: str, image_path: str) -> bool:
        embedding = self.get_embedding(image_path)
        if embedding is not None:
            self.database[person_name] = embedding
            return True
        return False

    def _match_embedding(self, embedding: np.ndarray) -> tuple[str, float]:
        if not self.database:
            return "unknown", 0.0
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
        t0 = time.perf_counter()
        faces = self.detect_faces(image_path)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if not faces:
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
