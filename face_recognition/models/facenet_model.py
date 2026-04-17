"""
FaceNet (Sandberg) via DeepFace.

Uses DeepFace's ``Facenet512`` model – the 512-dimensional variant of David
Sandberg's original FaceNet, offering better accuracy than the 128-d version.

Install:
    pip install deepface tf-keras
"""

import numpy as np
from typing import Any
import cv2

from .base import BaseFaceRecognizer


class FaceNetRecognizer(BaseFaceRecognizer):
    model_name = "FaceNet_Sandberg"
    # FaceNet embeddings are L2-normalised; cosine threshold ~0.40 works well
    threshold = 0.40

    def __init__(self, use_512: bool = True, min_face_size: int = 80) -> None:
        super().__init__()
        try:
            from deepface import DeepFace
            self._deepface = DeepFace
        except ImportError as exc:
            raise ImportError(
                "deepface is required.  Install with:\n"
                "  pip install deepface tf-keras"
            ) from exc

        self._variant = "Facenet512" if use_512 else "Facenet"
        self._min_face_size = min_face_size
        # Pre-warm: avoids cold-start latency during the timed evaluation
        print(f"  Pre-loading {self._variant} weights (one-time download if absent)...")
        self._deepface.build_model(self._variant)

    def get_embedding(self, image_path: str) -> np.ndarray | None:
        """Return embedding for the first (largest) detected face (used for enrollment)."""
        embeddings = self.get_all_embeddings(image_path)
        return embeddings[0] if embeddings else None

    def get_all_embeddings(self, image_path: str) -> list[np.ndarray]:
        """Return embeddings for **every** face detected in the image."""
        return [d["embedding"] for d in self.detect_faces(image_path)]

    def detect_faces(self, image_path: str) -> list[dict[str, Any]]:
        """Return embedding + bounding box for each detected face."""
        try:
            result = self._deepface.represent(
                img_path=str(image_path),
                model_name=self._variant,
                enforce_detection=True,
                detector_backend="opencv",
                align=True,
                normalization="Facenet2018",
            )
        except Exception:
            return []

        detections: list[dict[str, Any]] = []
        for item in result:
            embedding = np.array(item["embedding"], dtype=np.float32)
            area = item.get("facial_area") or {}
            x = int(area.get("x", 0))
            y = int(area.get("y", 0))
            w = int(area.get("w", 0))
            h = int(area.get("h", 0))

            # Skip tiny detections which are often unstable in crowded scenes.
            if w < self._min_face_size or h < self._min_face_size:
                continue

            bbox = (x, y, x + w, y + h) if w > 0 and h > 0 else None
            detections.append({"embedding": embedding, "bbox": bbox})

        return detections

    def detect_bboxes(self, image_path: str) -> list[tuple[int, int, int, int]]:
        """Fallback face boxes for annotation when embedding detection fails."""
        boxes = super().detect_bboxes(image_path)
        if boxes:
            return boxes

        img = cv2.imread(str(image_path))
        if img is None:
            return []
        img_h, img_w = img.shape[:2]
        img_area = img_h * img_w

        try:
            faces = self._deepface.extract_faces(
                img_path=str(image_path),
                detector_backend="retinaface",
                enforce_detection=False,
                align=True,
            )
        except Exception:
            return []

        fallback_boxes: list[tuple[int, int, int, int]] = []
        for face in faces:
            area = face.get("facial_area") or {}
            x = int(area.get("x", 0))
            y = int(area.get("y", 0))
            w = int(area.get("w", 0))
            h = int(area.get("h", 0))
            if w <= 0 or h <= 0:
                continue

            # Ignore degenerate detections that span almost the whole frame.
            if (w * h) / max(img_area, 1) > 0.95:
                continue

            x1 = max(0, min(x, img_w - 1))
            y1 = max(0, min(y, img_h - 1))
            x2 = max(0, min(x + w, img_w - 1))
            y2 = max(0, min(y + h, img_h - 1))
            if x2 <= x1 or y2 <= y1:
                continue

            fallback_boxes.append((x1, y1, x2, y2))

        return fallback_boxes
