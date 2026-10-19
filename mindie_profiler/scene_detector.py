"""Explicit scene annotation and measured-step steady-state detection.

[Role]
Reads measured `StepSample` windows plus caller metadata. Workload mode
(inference/training/RL) comes only from an explicit annotation; framework or
kernel names are never used to infer it. Steady state is a suffix of measured
steps whose selected timing metric stays inside a declared tolerance band.

[Invariants]
- No explicit scene annotation, unknown annotations and insufficient samples
  return `status="uncertain"` with a reason, never a guessed mode or window.
- Detection consumes measured samples; it never fabricates timing values or
  claims a synthetic run is a real device result.
- A steady window has at least `min_steady_steps` consecutive measured steps;
  duplicate or non-finite step identities invalidate the window.
- Annotation hints (`warmup_steps`, `steady_start_step`) only filter explicit
  measured steps; they never create a step or relax the stability band.

[Timeouts & Locks]
- Pure computation. No locks, sleeps, I/O, subprocesses or global caches.

[Side-Effect Policy]
- Read-only over caller-owned sequences; returns fresh dataclasses/dicts only.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

SCENE_MODES = ("inference", "training", "rl")
SCENE_ALIASES = {
    "infer": "inference",
    "inference": "inference",
    "train": "training",
    "training": "training",
    "rl": "rl",
    "reinforcement_learning": "rl",
    "reinforcement": "rl",
}
PHASE_ALIASES = {
    "prefill": "prefill",
    "prompt": "prefill",
    "decode": "decode",
    "decoding": "decode",
    "mixed": "mixed",
    "train_forward": "train_forward",
    "train_backward": "train_backward",
    "rl_rollout": "rl_rollout",
    "rl_train": "rl_train",
}
SUPPORTED_METRICS = ("wall_ms", "busy_union_ms", "kernel_sum_ms", "total_cost_ms")
STEADY = "steady"
UNCERTAIN = "uncertain"


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _pairs(value: Any) -> tuple[tuple[str, str], ...]:
    if value is None:
        return ()
    if isinstance(value, Mapping):
        return tuple(sorted((str(k), str(v)) for k, v in value.items()))
    return tuple((str(k), str(v)) for k, v in value)


def _annotation(metadata: "SceneMetadata", name: str) -> str | None:
    target = name.strip().lower()
    for key, value in metadata.annotations:
        if key.strip().lower() == target:
            text = str(value).strip()
            return text or None
    return None


@dataclass(frozen=True)
class SceneMetadata:
    """Explicit caller metadata; no field is inferred from names."""

    mode: str | None = None
    phase: str | None = None
    framework: str | None = None
    tp: int | None = None
    dp: int | None = None
    graph_mode: str | None = None
    clock_domain: str | None = None
    annotation_source: str | None = None
    annotations: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "annotations", _pairs(self.annotations))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SceneMetadata":
        data = dict(value)
        if data.get("mode") is None and data.get("scene") is not None:
            data["mode"] = data["scene"]
        allowed = {
            "mode", "phase", "framework", "tp", "dp", "graph_mode",
            "clock_domain", "annotation_source", "annotations",
        }
        return cls(**{key: data[key] for key in allowed if key in data})

    def annotation(self, name: str) -> str | None:
        return _annotation(self, name)


def _coerce_metadata(value: SceneMetadata | Mapping[str, Any] | str | None) -> SceneMetadata | None:
    if value is None:
        return None
    if isinstance(value, SceneMetadata):
        return value
    if isinstance(value, str):
        return SceneMetadata(mode=value)
    if isinstance(value, Mapping):
        return SceneMetadata.from_mapping(value)
    return None


@dataclass(frozen=True)
class SceneDetection:
    status: str
    mode: str | None
    phase: str | None
    metadata: SceneMetadata | None
    annotation: str | None
    annotation_source: str | None
    uncertainty_reason: str | None
    limitations: tuple[str, ...] = ()

    def as_c7_scene(self) -> dict[str, Any]:
        metadata = self.metadata or SceneMetadata()
        return {
            "framework": metadata.framework,
            "phase": self.phase,
            "tp": metadata.tp,
            "dp": metadata.dp,
            "graph_mode": metadata.graph_mode,
            "clock_domain": metadata.clock_domain,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "mode": self.mode,
            "phase": self.phase,
            "annotation": self.annotation,
            "annotation_source": self.annotation_source,
            "uncertainty_reason": self.uncertainty_reason,
            "limitations": list(self.limitations),
            "scene": self.as_c7_scene(),
        }


def detect_scene(metadata: SceneMetadata | Mapping[str, Any] | str | None) -> SceneDetection:
    """Classify workload mode only from an explicit annotation."""
    meta = _coerce_metadata(metadata)
    if meta is None:
        return SceneDetection(
            status=UNCERTAIN,
            mode=None,
            phase=None,
            metadata=None,
            annotation=None,
            annotation_source=None,
            uncertainty_reason="missing_scene_annotation",
        )
    raw = meta.mode
    source = "metadata.mode" if raw is not None else None
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        for key in ("mode", "scene", "workload"):
            raw = meta.annotation(key)
            if raw is not None:
                source = f"annotations.{key}"
                break
    if raw is None:
        return SceneDetection(
            status=UNCERTAIN,
            mode=None,
            phase=None,
            metadata=meta,
            annotation=None,
            annotation_source=None,
            uncertainty_reason="missing_explicit_scene_annotation",
        )
    normalized = str(raw).strip().lower().replace("-", "_").replace(" ", "_")
    mode = SCENE_ALIASES.get(normalized)
    if mode is None:
        return SceneDetection(
            status=UNCERTAIN,
            mode=None,
            phase=None,
            metadata=meta,
            annotation=str(raw),
            annotation_source=source,
            uncertainty_reason=f"unsupported_scene_annotation:{normalized}",
        )
    limitations: list[str] = []
    raw_phase = meta.phase if meta.phase is not None else meta.annotation("phase")
    phase: str | None = None
    if raw_phase is not None:
        phase_key = str(raw_phase).strip().lower().replace("-", "_").replace(" ", "_")
        phase = PHASE_ALIASES.get(phase_key)
        if phase is None:
            limitations.append(f"unsupported_phase_annotation:{phase_key}")
    return SceneDetection(
        status="detected",
        mode=mode,
        phase=phase,
        metadata=meta,
        annotation=str(raw),
        annotation_source=source,
        uncertainty_reason=None,
        limitations=tuple(limitations),
    )


@dataclass(frozen=True)
class StepSample:
    """One measured step. `step_id` is accepted as an alias for `step`."""

    step: int | None = None
    wall_ms: float | None = None
    busy_union_ms: float | None = None
    kernel_sum_ms: float | None = None
    exposed_comm_ms: float | None = None
    host_gap_ms: float | None = None
    total_cost_ms: float | None = None
    rank: int | None = None
    annotations: tuple[tuple[str, str], ...] = ()
    step_id: int | None = None

    def __post_init__(self) -> None:
        if self.step is None and self.step_id is not None:
            object.__setattr__(self, "step", self.step_id)
        elif self.step is not None and self.step_id is None:
            object.__setattr__(self, "step_id", self.step)
        object.__setattr__(self, "annotations", _pairs(self.annotations))

    @property
    def resolved_step(self) -> int | None:
        return self.step if self.step is not None else self.step_id


def coerce_step_sample(value: StepSample | Mapping[str, Any]) -> StepSample | None:
    if isinstance(value, StepSample):
        return value
    if not isinstance(value, Mapping):
        return None
    step = value.get("step", value.get("step_id"))
    return StepSample(
        step=step,
        wall_ms=value.get("wall_ms"),
        busy_union_ms=value.get("busy_union_ms"),
        kernel_sum_ms=value.get("kernel_sum_ms"),
        exposed_comm_ms=value.get("exposed_comm_ms"),
        host_gap_ms=value.get("host_gap_ms"),
        total_cost_ms=value.get("total_cost_ms"),
        rank=value.get("rank"),
        annotations=value.get("annotations", ()),
        step_id=value.get("step_id"),
    )


@dataclass(frozen=True)
class SteadyStateConfig:
    min_steady_steps: int = 3
    relative_tolerance: float = 0.10
    absolute_tolerance_ms: float = 0.0
    metric: str = "wall_ms"
    baseline_count: int | None = None

    def validation_errors(self) -> tuple[str, ...]:
        errors: list[str] = []
        if not isinstance(self.min_steady_steps, int) or isinstance(self.min_steady_steps, bool) or self.min_steady_steps < 3:
            errors.append("min_steady_steps_must_be_at_least_3")
        if not _finite(self.relative_tolerance) or float(self.relative_tolerance) < 0:
            errors.append("relative_tolerance_invalid")
        if not _finite(self.absolute_tolerance_ms) or float(self.absolute_tolerance_ms) < 0:
            errors.append("absolute_tolerance_invalid")
        if self.metric not in SUPPORTED_METRICS:
            errors.append(f"unsupported_metric:{self.metric}")
        if self.baseline_count is not None and (
            not isinstance(self.baseline_count, int)
            or isinstance(self.baseline_count, bool)
            or self.baseline_count < 1
        ):
            errors.append("baseline_count_invalid")
        return tuple(errors)


@dataclass(frozen=True)
class SteadyWindow:
    status: str
    steady_steps: tuple[int, ...]
    window_start_step: int | None
    window_end_step: int | None
    metric: str
    baseline_ms: float | None
    tolerance_ms: float | None
    sample_count: int
    excluded_steps: tuple[int, ...]
    warmup_steps: tuple[int, ...]
    uncertainty_reason: str | None
    limitations: tuple[str, ...] = ()
    metadata_used: tuple[str, ...] = ()

    def as_c7_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "steady_steps": list(self.steady_steps),
            "window_start_step": self.window_start_step,
            "window_end_step": self.window_end_step,
            "metric": self.metric,
            "baseline_ms": self.baseline_ms,
            "tolerance_ms": self.tolerance_ms,
            "sample_count": self.sample_count,
            "excluded_steps": list(self.excluded_steps),
            "warmup_steps": list(self.warmup_steps),
            "uncertainty_reason": self.uncertainty_reason,
            "limitations": list(self.limitations),
            "metadata_used": list(self.metadata_used),
        }

    as_dict = as_c7_dict


def _parse_ints(value: str) -> tuple[int, ...] | None:
    parts = [part for part in value.replace(";", ",").replace(" ", ",").split(",") if part]
    try:
        return tuple(sorted({int(part) for part in parts}))
    except ValueError:
        return None


def _metric_value(sample: StepSample, metric: str) -> Any:
    return getattr(sample, metric, None)


def detect_steady_window(
    samples: Sequence[StepSample | Mapping[str, Any]] | None,
    metadata: SceneMetadata | Mapping[str, Any] | None = None,
    config: SteadyStateConfig | Mapping[str, Any] | None = None,
    *,
    min_steady_steps: int | None = None,
    relative_tolerance: float | None = None,
    absolute_tolerance_ms: float | None = None,
) -> SteadyWindow:
    """Find a measured stable suffix or return an explicit uncertainty."""
    if isinstance(config, Mapping):
        try:
            config = SteadyStateConfig(**dict(config))
        except (TypeError, ValueError) as exc:
            return SteadyWindow(
                status=UNCERTAIN, steady_steps=(), window_start_step=None, window_end_step=None,
                metric="wall_ms", baseline_ms=None, tolerance_ms=None, sample_count=0,
                excluded_steps=(), warmup_steps=(), uncertainty_reason=f"invalid_config:{type(exc).__name__}",
            )
    if config is None:
        config = SteadyStateConfig()
    if min_steady_steps is not None or relative_tolerance is not None or absolute_tolerance_ms is not None:
        config = SteadyStateConfig(
            min_steady_steps=config.min_steady_steps if min_steady_steps is None else min_steady_steps,
            relative_tolerance=config.relative_tolerance if relative_tolerance is None else relative_tolerance,
            absolute_tolerance_ms=config.absolute_tolerance_ms if absolute_tolerance_ms is None else absolute_tolerance_ms,
            metric=config.metric,
            baseline_count=config.baseline_count,
        )
    errors = config.validation_errors()
    if errors:
        return SteadyWindow(
            status=UNCERTAIN, steady_steps=(), window_start_step=None, window_end_step=None,
            metric=config.metric, baseline_ms=None, tolerance_ms=None, sample_count=0,
            excluded_steps=(), warmup_steps=(), uncertainty_reason="invalid_config:" + ",".join(errors),
        )
    raw = list(samples or ())
    if not raw:
        return SteadyWindow(
            status=UNCERTAIN, steady_steps=(), window_start_step=None, window_end_step=None,
            metric=config.metric, baseline_ms=None, tolerance_ms=None, sample_count=0,
            excluded_steps=(), warmup_steps=(), uncertainty_reason="insufficient_step_samples",
        )
    coerced: list[StepSample] = []
    for index, item in enumerate(raw):
        sample = coerce_step_sample(item)
        if sample is None:
            return SteadyWindow(
                status=UNCERTAIN, steady_steps=(), window_start_step=None, window_end_step=None,
                metric=config.metric, baseline_ms=None, tolerance_ms=None, sample_count=len(raw),
                excluded_steps=(), warmup_steps=(), uncertainty_reason=f"invalid_step_sample:{index}",
            )
        coerced.append(sample)
    conflicting = [
        sample for sample in coerced
        if sample.step is not None and sample.step_id is not None and sample.step != sample.step_id
    ]
    if conflicting:
        return SteadyWindow(
            status=UNCERTAIN, steady_steps=(), window_start_step=None, window_end_step=None,
            metric=config.metric, baseline_ms=None, tolerance_ms=None, sample_count=len(coerced),
            excluded_steps=(), warmup_steps=(), uncertainty_reason="conflicting_step_identity",
        )
    steps = [sample.resolved_step for sample in coerced]
    if any(step is None for step in steps):
        return SteadyWindow(
            status=UNCERTAIN, steady_steps=(), window_start_step=None, window_end_step=None,
            metric=config.metric, baseline_ms=None, tolerance_ms=None, sample_count=len(coerced),
            excluded_steps=(), warmup_steps=(), uncertainty_reason="missing_step_identity",
        )
    if len(set(steps)) != len(steps):
        return SteadyWindow(
            status=UNCERTAIN, steady_steps=(), window_start_step=None, window_end_step=None,
            metric=config.metric, baseline_ms=None, tolerance_ms=None, sample_count=len(coerced),
            excluded_steps=(), warmup_steps=(), uncertainty_reason="duplicate_step_identity",
        )
    meta = _coerce_metadata(metadata)
    limitations: list[str] = []
    metadata_used: list[str] = []
    if meta is not None:
        warmup_annotation = meta.annotation("warmup_steps")
        if warmup_annotation is not None:
            warmup = _parse_ints(warmup_annotation)
            if warmup is None:
                limitations.append("invalid_warmup_steps_annotation")
            else:
                warmup_set = set(warmup)
                filtered = [sample for sample in coerced if sample.resolved_step not in warmup_set]
                metadata_used.append("warmup_steps")
                coerced = filtered
        start_annotation = meta.annotation("steady_start_step")
        if start_annotation is not None:
            try:
                start_step = int(start_annotation)
            except ValueError:
                limitations.append("invalid_steady_start_step_annotation")
            else:
                filtered = [sample for sample in coerced if sample.resolved_step >= start_step]
                metadata_used.append("steady_start_step")
                coerced = filtered
    ordered = sorted(coerced, key=lambda sample: sample.resolved_step)
    values = [_metric_value(sample, config.metric) for sample in ordered]
    missing = [sample.resolved_step for sample, value in zip(ordered, values) if value is None]
    non_finite = [
        sample.resolved_step for sample, value in zip(ordered, values)
        if value is not None and not _finite(value)
    ]
    if missing or non_finite:
        reason = (
            "missing_metric:" + config.metric
            if missing
            else "non_finite_metric:" + config.metric
        )
        return SteadyWindow(
            status=UNCERTAIN, steady_steps=(), window_start_step=None, window_end_step=None,
            metric=config.metric, baseline_ms=None, tolerance_ms=None, sample_count=len(ordered),
            excluded_steps=tuple(sample.resolved_step for sample in ordered),
            warmup_steps=tuple(sample.resolved_step for sample in ordered),
            uncertainty_reason=reason, limitations=tuple(limitations), metadata_used=tuple(metadata_used),
        )
    numeric = [float(value) for value in values]
    if config.metric == "wall_ms" and any(value <= 0 for value in numeric):
        return SteadyWindow(
            status=UNCERTAIN, steady_steps=(), window_start_step=None, window_end_step=None,
            metric=config.metric, baseline_ms=None, tolerance_ms=None, sample_count=len(ordered),
            excluded_steps=tuple(sample.resolved_step for sample in ordered),
            warmup_steps=tuple(sample.resolved_step for sample in ordered),
            uncertainty_reason="nonpositive_wall_ms", limitations=tuple(limitations),
            metadata_used=tuple(metadata_used),
        )
    if len(ordered) < config.min_steady_steps:
        return SteadyWindow(
            status=UNCERTAIN, steady_steps=(), window_start_step=None, window_end_step=None,
            metric=config.metric, baseline_ms=None, tolerance_ms=None, sample_count=len(ordered),
            excluded_steps=tuple(sample.resolved_step for sample in ordered),
            warmup_steps=tuple(sample.resolved_step for sample in ordered),
            uncertainty_reason="insufficient_step_samples", limitations=tuple(limitations),
            metadata_used=tuple(metadata_used),
        )
    baseline_count = config.baseline_count or max(config.min_steady_steps, min(len(numeric), max(config.min_steady_steps, len(numeric) // 3)))
    baseline_count = min(len(numeric), max(1, baseline_count))
    baseline = float(statistics.median(numeric[-baseline_count:]))
    band = max(float(config.absolute_tolerance_ms), float(config.relative_tolerance) * abs(baseline))
    start = len(numeric)
    while start > 0 and abs(numeric[start - 1] - baseline) <= band + 1e-12:
        start -= 1
    stable = ordered[start:]
    if len(stable) < config.min_steady_steps:
        return SteadyWindow(
            status=UNCERTAIN, steady_steps=(), window_start_step=None, window_end_step=None,
            metric=config.metric, baseline_ms=baseline, tolerance_ms=band, sample_count=len(ordered),
            excluded_steps=tuple(sample.resolved_step for sample in ordered),
            warmup_steps=tuple(sample.resolved_step for sample in ordered),
            uncertainty_reason="no_steady_suffix_within_tolerance", limitations=tuple(limitations),
            metadata_used=tuple(metadata_used),
        )
    excluded = ordered[:start]
    return SteadyWindow(
        status=STEADY,
        steady_steps=tuple(sample.resolved_step for sample in stable),
        window_start_step=stable[0].resolved_step,
        window_end_step=stable[-1].resolved_step,
        metric=config.metric,
        baseline_ms=baseline,
        tolerance_ms=band,
        sample_count=len(ordered),
        excluded_steps=tuple(sample.resolved_step for sample in excluded),
        warmup_steps=tuple(sample.resolved_step for sample in excluded),
        uncertainty_reason=None,
        limitations=tuple(limitations),
        metadata_used=tuple(metadata_used),
    )


@dataclass(frozen=True)
class SceneAndSteady:
    scene: SceneDetection
    steady: SteadyWindow

    def as_c7_dict(self) -> dict[str, Any]:
        return {"scene": self.scene.as_c7_scene(), "steady_window": self.steady.as_c7_dict()}


def detect_scene_and_steady(
    metadata: SceneMetadata | Mapping[str, Any] | str | None,
    samples: Sequence[StepSample | Mapping[str, Any]] | None,
    config: SteadyStateConfig | Mapping[str, Any] | None = None,
) -> SceneAndSteady:
    return SceneAndSteady(scene=detect_scene(metadata), steady=detect_steady_window(samples, metadata, config))


__all__ = [
    "SCENE_MODES",
    "SUPPORTED_METRICS",
    "STEADY",
    "UNCERTAIN",
    "SceneMetadata",
    "SceneDetection",
    "StepSample",
    "SteadyStateConfig",
    "SteadyWindow",
    "SceneAndSteady",
    "coerce_step_sample",
    "detect_scene",
    "detect_steady_window",
    "detect_scene_and_steady",
]
