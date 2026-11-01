"""Numeric metric kernels for lockstep differential comparison.

[Role]
  Compute max-abs, relative-L2, ULP, nonfinite mismatch, jitter and SJR for one
  captured tensor path.  Kept in a small module because the differ itself owns
  tracing/healing rather than metric arithmetic.

[Invariants]
  * Integer magnitudes are always computed as non-negative Python ints before
    any float conversion; signed int64 overflow can never produce a negative
    ``max_abs_err`` or ``sjr``.
  * Float64 ULP distances subtract Python 64-bit integers, preserving all 53+
    mantissa bits that a float64 subtraction would discard.

[Timeouts & Locks]
  Pure synchronous tensor math.  No sleeps, subprocesses, network or locks.

[Side-Effect Policy]
  Allocates temporary tensors only; never mutates its inputs.
"""
from __future__ import annotations
import math
import struct
from .storage_graph import _require_torch
_INF = float("inf")
def _missing_metric(path, status, tensor):
    return {"tree_path": path, "status": status, "max_abs_err": _INF,
            "rel_l2_err": _INF, "ulp_max": 0,
            "nonfinite_mismatch_count": int(tensor.numel()),
            "sample_count": int(tensor.numel()), "jitter_p50": 0.0,
            "jitter_max": 0.0, "sjr": _INF, "ulp_floor": 0.0}
def _ordered_bits(tensor):
    torch = _require_torch()
    if tensor.dtype == torch.float64:
        result = []
        for value in tensor.reshape(-1).tolist():
            bits = struct.unpack(">q", struct.pack(">d", float(value)))[0]
            ordered = (-0x8000000000000000 - bits) if bits < 0 else bits
            result.append(ordered)
        return result
    width = int(tensor.element_size()) * 8
    if width not in (8, 16, 32):
        return None
    itype = {8: torch.int8, 16: torch.int16, 32: torch.int32}[width]
    bits = tensor.contiguous().reshape(-1).view(itype).to(torch.int64) & ((1 << width) - 1)
    sign = (bits >> (width - 1)) & 1
    half = 1 << (width - 1)
    return torch.where(sign.bool(), half - (bits & (half - 1)), half + bits)
def _ulp_max(a, b):
    torch = _require_torch()
    if a.dtype != b.dtype or not a.dtype.is_floating_point or int(a.numel()) == 0:
        return 0
    ordered_a = _ordered_bits(a)
    ordered_b = _ordered_bits(b)
    if ordered_a is None or ordered_b is None:
        return 0
    both_nan = torch.isnan(a.reshape(-1)) & torch.isnan(b.reshape(-1))
    both_same_inf = (torch.isinf(a.reshape(-1)) & torch.isinf(b.reshape(-1))
                     & (torch.sign(a.reshape(-1)) == torch.sign(b.reshape(-1))))
    ignored = (both_nan | both_same_inf).tolist()
    if isinstance(ordered_a, list):
        best = 0
        for left, right, skip in zip(ordered_a, ordered_b, ignored):
            if skip:
                continue
            best = max(best, abs(int(left) - int(right)))
        return float(best)
    distance = (ordered_a - ordered_b).abs().to(torch.float64)
    ignored_tensor = torch.tensor(ignored, dtype=torch.bool, device=distance.device)
    distance = torch.where(ignored_tensor, torch.zeros_like(distance), distance)
    return float(distance.max().item()) if distance.numel() else 0.0
def _ulp_floor(tensor, override=None):
    if override is not None:
        return float(override)
    torch = _require_torch()
    if not tensor.dtype.is_floating_point:
        return 1.0
    finite = torch.nan_to_num(tensor.to(torch.float64), nan=0.0, posinf=0.0, neginf=0.0)
    scale = max(float(finite.abs().max().item()) if finite.numel() else 0.0, 1.0)
    return float(torch.finfo(tensor.dtype).eps) * scale
def _python_ints(tensor):
    return [int(value) for value in tensor.reshape(-1).tolist()]
def _jitter_delta(baseline, candidate):
    torch = _require_torch()
    if tuple(candidate.shape) != tuple(baseline.shape) or candidate.dtype != baseline.dtype:
        return None
    complex_input = baseline.is_complex()
    int_input = not baseline.dtype.is_floating_point and not complex_input
    if int_input:
        delta = torch.tensor(
            [abs(left - right) for left, right in zip(
                _python_ints(candidate.to(torch.int64)),
                _python_ints(baseline.to(torch.int64)))],
            dtype=torch.float64, device=baseline.device)
    elif complex_input:
        c_val = candidate.to(torch.complex128)
        b_val = baseline.to(torch.complex128)
        delta = (c_val - b_val).abs()
        same_nan = torch.isnan(c_val) & torch.isnan(b_val)
        same_inf = torch.isinf(c_val) & torch.isinf(b_val) & (c_val == b_val)
        delta = torch.where(same_nan | same_inf, torch.zeros_like(delta), delta)
    else:
        c_val = candidate.to(torch.float64)
        b_val = baseline.to(torch.float64)
        delta = (c_val - b_val).abs()
        same_nan = torch.isnan(c_val) & torch.isnan(b_val)
        same_inf = (torch.isinf(c_val) & torch.isinf(b_val)
                    & (torch.sign(c_val) == torch.sign(b_val)))
        delta = torch.where(same_nan | same_inf, torch.zeros_like(delta), delta)
    return torch.where(torch.isnan(delta), torch.full_like(delta, _INF), delta).reshape(-1)
def _tensor_metrics(path, baseline, target, jitter_tensors, jitter_floor_override,
                    jitter_summary=None):
    torch = _require_torch()
    if tuple(baseline.shape) != tuple(target.shape):
        return _missing_metric(path, "shape_mismatch", baseline)
    if baseline.dtype != target.dtype:
        return _missing_metric(path, "dtype_mismatch", baseline)
    with torch.no_grad():
        complex_input = baseline.is_complex()
        int_input = not baseline.dtype.is_floating_point and not complex_input
        int_diff = None
        if int_input:
            a_int = baseline.to(torch.int64)
            b_int = target.to(torch.int64)
            a = a_int.to(torch.float64)
            int_diff = [abs(left - right)
                        for left, right in zip(_python_ints(a_int), _python_ints(b_int))]
            diff = torch.tensor(int_diff, dtype=torch.float64, device=a.device)
            nan_a = nan_b = inf_a = inf_b = torch.zeros_like(a, dtype=torch.bool)
            same_nan = same_inf = torch.zeros_like(a, dtype=torch.bool)
            mismatch = torch.zeros_like(a, dtype=torch.bool)
        elif complex_input:
            a = baseline.to(torch.complex128)
            b = target.to(torch.complex128)
            nan_a, nan_b = torch.isnan(a), torch.isnan(b)
            inf_a, inf_b = torch.isinf(a), torch.isinf(b)
            same_nan = nan_a & nan_b
            same_inf = inf_a & inf_b & (a == b)
            diff = (a - b).abs()
            diff = torch.where(same_nan | same_inf, torch.zeros_like(diff), diff)
            diff = torch.where(torch.isnan(diff), torch.full_like(diff, _INF), diff)
            mismatch = ((nan_a != nan_b) | (inf_a != inf_b)
                        | (inf_a & inf_b & ~same_inf))
            a = a.abs()
            b = b.abs()
        else:
            a = baseline.to(torch.float64)
            b = target.to(torch.float64)
            nan_a, nan_b = torch.isnan(a), torch.isnan(b)
            inf_a, inf_b = torch.isinf(a), torch.isinf(b)
            same_nan = nan_a & nan_b
            same_inf = inf_a & inf_b & (torch.sign(a) == torch.sign(b))
            diff = (a - b).abs()
            diff = torch.where(same_nan | same_inf, torch.zeros_like(diff), diff)
            diff = torch.where(torch.isnan(diff), torch.full_like(diff, _INF), diff)
            mismatch = ((nan_a != nan_b) | (inf_a != inf_b)
                        | (inf_a & inf_b & (torch.sign(a) != torch.sign(b))))
        if int_input:
            max_abs = float(max(int_diff, default=0))
        else:
            max_abs = float(diff.max().item()) if diff.numel() else 0.0
        norm_base = float(torch.linalg.vector_norm(a).item()) if a.numel() else 0.0
        norm_diff = float(torch.linalg.vector_norm(diff).item()) if diff.numel() else 0.0
        nonfinite = int(mismatch.sum().item())
        if nonfinite == 0 and norm_diff == 0.0:
            rel_l2 = 0.0
        elif not math.isfinite(norm_diff):
            rel_l2 = _INF
        elif not math.isfinite(norm_base):
            rel_l2 = _INF if norm_diff > 0.0 else 0.0
        elif norm_base > 0.0:
            rel_l2 = norm_diff / norm_base
        else:
            rel_l2 = 0.0 if norm_diff == 0.0 else _INF
        ulp = _ulp_max(baseline, target)
        jitter_max = 0.0
        jitter_p50 = 0.0
        if jitter_summary is not None:
            jitter_max, jitter_p50 = float(jitter_summary[0]), float(jitter_summary[1])
        else:
            deviations = [
                delta for candidate in jitter_tensors
                if (delta := _jitter_delta(baseline, candidate)) is not None and delta.numel()
            ]
            if deviations:
                elementwise = torch.stack(deviations).max(dim=0).values
                if elementwise.numel():
                    jitter_max = float(elementwise.max().item())
                    jitter_p50 = float(elementwise.median().item())
        floor = _ulp_floor(baseline, jitter_floor_override)
        denominator = max(jitter_max, floor)
        sjr = (max_abs / denominator) if denominator > 0 else (0.0 if max_abs == 0 else _INF)
        return {"tree_path": path, "status": "ok", "max_abs_err": max_abs, "rel_l2_err": rel_l2,
                "ulp_max": ulp, "nonfinite_mismatch_count": nonfinite,
                "sample_count": int(baseline.numel()), "jitter_p50": jitter_p50,
                "jitter_max": jitter_max, "sjr": sjr, "ulp_floor": floor}
def _aggregate_metrics(metrics):
    return {"max_abs_err": max((m["max_abs_err"] for m in metrics), default=0.0),
            "rel_l2_err": max((m["rel_l2_err"] for m in metrics), default=0.0),
            "ulp_max": max((m["ulp_max"] for m in metrics), default=0),
            "nonfinite_mismatch_count": sum(m.get("nonfinite_mismatch_count", 0) for m in metrics),
            "sample_count": sum(m.get("sample_count", 0) for m in metrics),
            "jitter_p50": max((m["jitter_p50"] for m in metrics), default=0.0),
            "jitter_max": max((m["jitter_max"] for m in metrics), default=0.0),
            "sjr": max((m["sjr"] for m in metrics), default=0.0), "tensor_paths": len(metrics)}
