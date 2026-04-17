"""
CompreFace (REST service) face recognition.

CompreFace runs as a Docker container and exposes a REST API for face
enrollment and recognition.  This class wraps that API.

Setup:
    docker run -p 8000:8000 exadel/compreface

Then create a Recognition service in the CompreFace UI, copy the API key,
and pass it via --compreface-key when running main.py.

Install:
    pip install requests
"""

import os
import time
import uuid
from typing import Any

from .base import BaseFaceRecognizer


class CompreFaceRecognizer(BaseFaceRecognizer):
    model_name = "CompreFace"
    # CompreFace similarity is already 0–1; skip embedding-level threshold
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
            raise ImportError(
                "requests is required.  Install with:\n"
                "  pip install requests"
            ) from exc

        api_key = (api_key or os.getenv("COMPREFACE_API_KEY", "")).strip()
        if not api_key:
            raise ValueError(
                "CompreFace API key is required. "
                "Pass --compreface-key or set COMPREFACE_API_KEY."
            )
        try:
            uuid.UUID(api_key)
        except ValueError as exc:
            raise ValueError(
                "CompreFace API key must be a valid UUID from a Recognition service."
            ) from exc

        self._base_url = f"{host}:{port}"
        self._api_key = api_key
        self._enrolled: set[str] = set()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _headers(self) -> dict:
        return {"x-api-key": self._api_key}

    # ------------------------------------------------------------------
    # Overrides – CompreFace manages embeddings server-side
    # ------------------------------------------------------------------
    def get_embedding(self, image_path: str):  # type: ignore[override]
        """Not used; CompreFace handles embeddings internally."""
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
            if resp.status_code in (200, 201):
                self._enrolled.add(person_name)
                return True
            print(
                f"  [WARN] CompreFace enroll failed "
                f"({resp.status_code}): {resp.text[:200]}"
            )
            return False
        except Exception as exc:
            print(f"  [WARN] CompreFace enroll error: {exc}")
            return False

    def recognize(self, image_path: str) -> tuple[str, float, float]:
        matches, elapsed_ms = self.recognize_all(image_path)
        if not matches:
            return "unknown", 0.0, elapsed_ms
        # Return the highest-confidence known match
        known = [(n, s) for n, s in matches if n != "unknown"]
        if not known:
            return "unknown", 0.0, elapsed_ms
        best_name, best_score = max(known, key=lambda x: x[1])
        return best_name, best_score, elapsed_ms

    def recognize_all(self, image_path: str) -> tuple[list[tuple[str, float]], float]:
        """
        Identify **all** faces in *image_path* via the CompreFace REST API.

        Returns
        -------
        (matches, inference_time_ms)
        matches is a list of (name, confidence) for every detected face.
        """
        detections, elapsed_ms = self.recognize_faces(image_path)
        matches = [
            (d["name"], float(d["confidence"]))
            for d in detections
        ]
        return matches, elapsed_ms

    def recognize_faces(self, image_path: str) -> tuple[list[dict[str, Any]], float]:
        """Return face-level matches including bounding boxes."""
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
                print(f"  [WARN] CompreFace recognize {resp.status_code}: {resp.text[:200]}")
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

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            print(f"  [WARN] CompreFace recognize_all error: {exc}")
            return [], elapsed_ms
