"""Tracker factories for person tracking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from boxmot.trackers.tracker_zoo import TRACKER_MAPPING

from .botsort import BoTSORTTracker
from .boxmot_client import BoxMOTTrackerClient, DISPLAY_NAMES
from .bytetrack import ByteTrackTracker
from .deepsort import DeepSORTTracker

CUSTOM_TRACKERS = ("botsort", "bytetrack", "deepsort")
BOXMOT_TRACKERS = tuple(sorted(TRACKER_MAPPING))
DEFAULT_TRACKERS = (
	"botsort",
	"bytetrack",
	"deepsort",
	"boxmot:strongsort",
	"boxmot:deepocsort",
)


@dataclass(frozen=True)
class TrackerSpec:
	raw: str
	kind: str
	backend: str
	label: str


def parse_tracker_spec(spec: str) -> TrackerSpec:
	normalized = spec.strip().lower()
	if normalized.startswith("boxmot:"):
		backend = normalized.split(":", 1)[1]
		if backend not in BOXMOT_TRACKERS:
			raise ValueError(
				f"Unsupported BoxMOT tracker '{backend}'. Supported values: {', '.join(BOXMOT_TRACKERS)}"
			)
		return TrackerSpec(raw=spec, kind="boxmot", backend=backend, label=DISPLAY_NAMES.get(backend, backend.title()))

	if normalized in CUSTOM_TRACKERS:
		return TrackerSpec(raw=spec, kind="custom", backend=normalized, label=DISPLAY_NAMES.get(normalized, normalized.title()))

	if normalized in BOXMOT_TRACKERS:
		return TrackerSpec(raw=spec, kind="boxmot", backend=normalized, label=DISPLAY_NAMES.get(normalized, normalized.title()))

	raise ValueError(
		"Unsupported tracker "
		f"'{spec}'. Supported values: {', '.join(CUSTOM_TRACKERS)} and boxmot:<tracker> for {', '.join(BOXMOT_TRACKERS)}"
	)


def expand_tracker_specs(specs: list[str]) -> list[TrackerSpec]:
	expanded: list[TrackerSpec] = []
	for spec in specs:
		if spec.lower() == "all":
			expanded.extend(parse_tracker_spec(value) for value in DEFAULT_TRACKERS)
			continue
		expanded.append(parse_tracker_spec(spec))

	deduped: list[TrackerSpec] = []
	seen: set[str] = set()
	for spec in expanded:
		key = f"{spec.kind}:{spec.backend}"
		if key in seen:
			continue
		seen.add(key)
		deduped.append(spec)
	return deduped


def build_tracker(
	spec: TrackerSpec | str,
	*,
	reid_model: str | None = None,
	device: str = "cpu",
	half: bool = False,
	deepsort_embedder: str = "mobilenet",
	deepsort_embedder_model: str | None = None,
	deepsort_embedder_weights: str | None = None,
) -> Any:
	tracker_spec = spec if isinstance(spec, TrackerSpec) else parse_tracker_spec(spec)

	if tracker_spec.kind == "custom":
		if tracker_spec.backend == "botsort":
			return BoTSORTTracker(reid_model=reid_model, device=device, half=half)
		if tracker_spec.backend == "bytetrack":
			return ByteTrackTracker(device=device, half=half)
		if tracker_spec.backend == "deepsort":
			return DeepSORTTracker(
				embedder=deepsort_embedder,
				embedder_model_name=deepsort_embedder_model,
				embedder_weights=deepsort_embedder_weights,
				half=half,
			)

	return BoxMOTTrackerClient(
		tracker_backend=tracker_spec.backend,
		reid_model=reid_model,
		device=device,
		half=half,
		backend_label="boxmot",
	)


__all__ = (
	"BOXMOT_TRACKERS",
	"CUSTOM_TRACKERS",
	"DEFAULT_TRACKERS",
	"TrackerSpec",
	"build_tracker",
	"expand_tracker_specs",
	"parse_tracker_spec",
)
