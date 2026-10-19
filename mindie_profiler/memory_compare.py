"""Peak comparison gates for measured/sampled memory observations.

[Role]
  Compare peak byte values only when phase, device scope, measurement kind,
  and sampled/measured peak scope agree.

[Invariants]
  * A non-peak observation never compares as a peak.
  * Missing bytes/phase/scope are explicit reasons, not zero fills.

[Timeouts & Locks]
  Pure mapping/arithmetic.

[Side-Effect Policy]
  No I/O or mutation.
"""

from __future__ import annotations

from typing import Any, Mapping

from .memory_units import _is_number, to_bytes


def _peak_bytes(value: Any) -> float | None:
    if not isinstance(value, Mapping):
        return None
    for key in ("bytes", "value_bytes", "byte_value"):
        if _is_number(value.get(key)):
            return float(value[key])
    if _is_number(value.get("value")):
        try:
            return to_bytes(value["value"], str(value.get("unit") or "B"))
        except ValueError:
            return None
    return None

def compare_peaks(baseline: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Compare two peaks only when phase and scope match."""
    reasons: list[dict[str, str]] = []
    b_bytes, c_bytes = _peak_bytes(baseline), _peak_bytes(candidate)
    b_phase = baseline.get("phase") if isinstance(baseline, Mapping) else None
    c_phase = candidate.get("phase") if isinstance(candidate, Mapping) else None
    b_scope = baseline.get("scope") if isinstance(baseline, Mapping) else None
    c_scope = candidate.get("scope") if isinstance(candidate, Mapping) else None
    if b_bytes is None or c_bytes is None:
        reasons.append({"code": "peak_bytes_missing", "message": "both peaks need explicit byte values"})
    if not b_phase or not c_phase:
        reasons.append({"code": "peak_phase_missing", "message": "both peaks need an explicit phase"})
    elif b_phase != c_phase:
        reasons.append({"code": "peak_phase_mismatch", "message": f"phase differs ({b_phase!r} vs {c_phase!r})"})
    if not b_scope or not c_scope:
        reasons.append({"code": "peak_scope_missing", "message": "both peaks need an explicit scope"})
    elif b_scope != c_scope:
        reasons.append({"code": "peak_scope_mismatch", "message": f"scope differs ({b_scope!r} vs {c_scope!r})"})
    b_kind = baseline.get("measurement_kind") if isinstance(baseline, Mapping) else None
    c_kind = candidate.get("measurement_kind") if isinstance(candidate, Mapping) else None
    if b_kind and c_kind and b_kind != c_kind:
        reasons.append({"code": "peak_measurement_kind_mismatch", "message": f"measurement kind differs ({b_kind!r} vs {c_kind!r})"})
    b_peak_scope = baseline.get("peak_scope") if isinstance(baseline, Mapping) else None
    c_peak_scope = candidate.get("peak_scope") if isinstance(candidate, Mapping) else None
    if b_peak_scope and c_peak_scope and b_peak_scope != c_peak_scope:
        reasons.append({"code": "peak_scope_mismatch", "message": f"peak scope differs ({b_peak_scope!r} vs {c_peak_scope!r})"})
    for label, item in (("baseline", baseline), ("candidate", candidate)):
        if isinstance(item, Mapping) and item.get("is_peak") is False:
            reasons.append({"code": "not_a_peak", "message": f"{label} observation is not marked as a peak"})
    b_at = baseline.get("observed_at") if isinstance(baseline, Mapping) else None
    c_at = candidate.get("observed_at") if isinstance(candidate, Mapping) else None
    exact = str(b_at) == str(c_at) if b_at is not None and c_at is not None else None
    if reasons:
        return {"comparable": False, "reasons": reasons, "baseline_bytes": b_bytes,
                "candidate_bytes": c_bytes, "delta_bytes": None, "ratio": None, "exact_moment": exact}
    assert b_bytes is not None and c_bytes is not None
    return {"comparable": True, "reasons": [], "baseline_bytes": b_bytes, "candidate_bytes": c_bytes,
            "delta_bytes": c_bytes - b_bytes, "ratio": (c_bytes / b_bytes) if b_bytes else None,
            "exact_moment": exact}

__all__ = ["compare_peaks"]
