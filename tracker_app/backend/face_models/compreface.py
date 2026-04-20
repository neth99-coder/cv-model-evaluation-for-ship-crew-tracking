import os
import time
import uuid
from typing import Any

from .base import BaseFaceRecognizer


class CompreFaceRecognizer(BaseFaceRecognizer):
    model_name = "CompreFace"
    threshold = 0.80

    def __init__(
        self,
        host: str = "http://localhost",
        port: int = 8000,
        api_key: str = "",
    ) -> None:
        super().__init__()
        try:
            import requests
            self._requests = requests
        except ImportError as exc:
            raise ImportError("requests is required. Install with: pip install requests") from exc

        api_key = (api_key or os.getenv("COMPREFACE_API_KEY", "")).strip()
        if not api_key:
            raise ValueError("COMPREFACE_API_KEY is required for CompreFace.")
        try:
            uuid.UUID(api_key)
        except ValueError as exc:
            raise ValueError("CompreFace API key must be a valid UUID.") from exc

        self._base_url = f"{host}:{port}"
        self._api_key = api_key

    def _headers(self) -> dict:
        return {"x-api-key": self._api_key}

    def get_embedding(self, image_path: str):  # type: ignore[override]
        return None

    def enroll(self, person_name: str, image_path: str) -> bool:
        url = f"{self._base_url}/api/v1/recognition/faces"
        try:
            with open(image_path, "rb") as fh:
                resp = self._requests.post(
                    url,
                    headers=self._headers(),
                    data={"subject": person_name},
                    files={"file": ("image.jpg", fh, "image/jpeg")},
                    timeout=30,
                )
            return resp.status_code in (200, 201)
        except Exception:
            return False

    def recognize_faces(self, image_path: str) -> tuple[list[dict[str, Any]], float]:
        url = f"{self._base_url}/api/v1/recognition/recognize"
        t0 = time.perf_counter()
        try:
            with open(image_path, "rb") as fh:
                resp = self._requests.post(
                    url,
                    headers=self._headers(),
                    files={"file": ("image.jpg", fh, "image/jpeg")},
                    params={"limit": 1, "det_prob_threshold": 0.8},
                    timeout=30,
                )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            if resp.status_code != 200:
                return [], elapsed_ms

            detections: list[dict[str, Any]] = []
            known_name_to_index: dict[str, int] = {}
            for face_result in resp.json().get("result", []):
                box = face_result.get("box") or {}
                x1 = int(box.get("x_min", 0))
                y1 = int(box.get("y_min", 0))
                x2 = int(box.get("x_max", 0))
                y2 = int(box.get("y_max", 0))
                bbox = (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None

                subjects = face_result.get("subjects", [])
                if not subjects:
                    detections.append({"name": "unknown", "confidence": 0.0, "bbox": bbox})
                    continue

                best = subjects[0]
                raw_name = best.get("subject", "unknown")
                similarity = float(best.get("similarity", 0.0))
                name = raw_name if similarity >= self.threshold else "unknown"

                if name == "unknown":
                    detections.append({"name": "unknown", "confidence": similarity, "bbox": bbox})
                    continue

                if name in known_name_to_index:
                    idx = known_name_to_index[name]
                    if similarity > detections[idx]["confidence"]:
                        detections[idx] = {"name": name, "confidence": similarity, "bbox": bbox}
                else:
                    known_name_to_index[name] = len(detections)
                    detections.append({"name": name, "confidence": similarity, "bbox": bbox})
            return detections, elapsed_ms
        except Exception:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return [], elapsed_ms
