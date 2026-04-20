import cv2
import numpy as np
from typing import Any

from .base import BaseFaceRecognizer


class MobileFaceNetRecognizer(BaseFaceRecognizer):
    model_name = "MobileFaceNet"
    threshold = 0.35

    def __init__(self) -> None:
        super().__init__()
        try:
            from insightface.app import FaceAnalysis
        except ImportError as exc:
            raise ImportError(
                "insightface is required. Install with: pip install insightface onnxruntime"
            ) from exc

        self._app = FaceAnalysis(
            name="buffalo_s",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self._app.prepare(ctx_id=0, det_size=(640, 640))

    def get_embedding(self, image_path: str) -> np.ndarray | None:
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        faces = self._app.get(img)
        if not faces:
            return None
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        return face.normed_embedding

    def get_all_embeddings(self, image_path: str) -> list[np.ndarray]:
        img = cv2.imread(str(image_path))
        if img is None:
            return []
        faces = self._app.get(img)
        return [f.normed_embedding for f in faces]

    def detect_faces(self, image_path: str) -> list[dict[str, Any]]:
        img = cv2.imread(str(image_path))
        if img is None:
            return []
        faces = self._app.get(img)
        detections: list[dict[str, Any]] = []
        for face in faces:
            x1, y1, x2, y2 = [int(v) for v in face.bbox]
            detections.append({"embedding": face.normed_embedding, "bbox": (x1, y1, x2, y2)})
        return detections
