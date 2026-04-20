"""
In-memory session & job state manager.
"""
import threading
from typing import Any, Optional


class SessionManager:
    def __init__(self):
        self._jobs: dict[str, dict] = {}
        self._ws_active: dict[str, bool] = {}
        self._lock = threading.Lock()

    def create_job(self, job_id: str, data: dict):
        with self._lock:
            self._jobs[job_id] = dict(data)

    def update_job(self, job_id: str, updates: dict):
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].update(updates)

    def get_job(self, job_id: str) -> Optional[dict]:
        with self._lock:
            return dict(self._jobs.get(job_id, {})) or None

    def list_jobs(self) -> list[dict]:
        with self._lock:
            return [
                {"job_id": job_id, **dict(data)}
                for job_id, data in self._jobs.items()
            ]

    def set_ws_active(self, session_id: str, active: bool):
        with self._lock:
            self._ws_active[session_id] = active

    def is_ws_active(self, session_id: str) -> bool:
        with self._lock:
            return self._ws_active.get(session_id, False)
