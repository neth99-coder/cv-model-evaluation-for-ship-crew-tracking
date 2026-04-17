"""
AuraFace (ResNet100) via InsightFace.

Uses the ``antelopev2`` model pack which ships ``glintr100`` – a ResNet100
backbone trained on Glint360K with ArcFace loss (the same architecture class
as AuraFace).  The pack is auto-downloaded on first run.

Install:
    pip install insightface onnxruntime
"""

import numpy as np
import cv2
from typing import Any
from pathlib import Path

from .base import BaseFaceRecognizer


class AuraFaceRecognizer(BaseFaceRecognizer):
    model_name = "AuraFace_ResNet100"
    threshold = 0.35

    def __init__(self) -> None:
        super().__init__()
        try:
            from insightface.app import FaceAnalysis
        except ImportError as exc:
            raise ImportError(
                "insightface is required.  Install with:\n"
                "  pip install insightface onnxruntime"
            ) from exc

        # Some antelopev2 archives unpack as ~/.insightface/models/antelopev2/antelopev2/*.onnx.
        # InsightFace expects ONNX files directly under ~/.insightface/models/antelopev2.
        self._normalize_antelopev2_layout()

        self._app = FaceAnalysis(
            name="antelopev2",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self._app.prepare(ctx_id=0, det_size=(640, 640))

    @staticmethod
    def _normalize_antelopev2_layout() -> None:
        model_root = Path.home() / ".insightface" / "models" / "antelopev2"
        nested = model_root / "antelopev2"
        if not nested.exists() or not nested.is_dir():
            return

        nested_onnx = list(nested.glob("*.onnx"))
        if not nested_onnx:
            return

        # Only move files if root doesn't already contain model files.
        root_has_onnx = any(model_root.glob("*.onnx"))
        if root_has_onnx:
            return

        for src in nested_onnx:
            src.rename(model_root / src.name)

    def get_embedding(self, image_path: str) -> np.ndarray | None:
        """Return embedding for the largest detected face (used for enrollment)."""
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"  [WARN] Cannot read image: {image_path}")
            return None
        faces = self._app.get(img)
        if not faces:
            return None
        # largest detected face
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        return face.normed_embedding

    def get_all_embeddings(self, image_path: str) -> list[np.ndarray]:
        """Return embeddings for **every** face detected in the image."""
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"  [WARN] Cannot read image: {image_path}")
            return []
        faces = self._app.get(img)
        return [f.normed_embedding for f in faces]

    def detect_faces(self, image_path: str) -> list[dict[str, Any]]:
        """Return embedding + bounding box for each detected face."""
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"  [WARN] Cannot read image: {image_path}")
            return []

        faces = self._app.get(img)
        detections: list[dict[str, Any]] = []
        for face in faces:
            x1, y1, x2, y2 = [int(v) for v in face.bbox]
            detections.append({
                "embedding": face.normed_embedding,
                "bbox": (x1, y1, x2, y2),
            })
        return detections
