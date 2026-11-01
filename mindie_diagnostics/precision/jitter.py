"""Streaming repeat-jitter accumulator for tensor and scalar outputs.

[Role]
  Consume one trial's output-value dict at a time and produce min/max/std range
  statistics without retaining all N trial tensors.

[Invariants]
  * At most one min/max/sum accumulator per element is kept, independent of the
    repeat count.
  * Integer ranges are computed from Python ints, so ``2**60 + i`` yields an
    exact range and signed int64 overflow cannot invert the result.
  * Complex tensors accumulate real and imaginary parts separately; identical
    ``inf``/``-inf``/``nan`` values produce a zero range and zero std, never
    ``NaN``.
  * Shape/dtype/structure changes surface as explicit mismatch statuses.

[Timeouts & Locks]
  Pure synchronous tensor math.  No sleeps, subprocesses, network or locks.

[Side-Effect Policy]
  Allocates temporary accumulators only; never mutates caller tensors.
"""
from __future__ import annotations
import math
_INF = float("inf")
def _torch():
    from .storage_graph import _require_torch
    return _require_torch()
def _tensor_bytes(values):
    torch = _torch()
    total = 0
    for value in values.values():
        if isinstance(value, torch.Tensor):
            total += int(value.numel()) * int(value.element_size())
    return total
def _scalar_equal(left, right):
    if isinstance(left, float) and isinstance(right, float):
        if math.isnan(left) and math.isnan(right):
            return True
    try:
        return bool(left == right)
    except Exception:
        return left is right
def _scalar_numeric(value):
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value
    if isinstance(value, complex):
        return abs(value)
    return None
class _TensorState:
    __slots__ = ("dtype", "shape", "numel", "complex", "int", "nan", "posinf",
                 "neginf", "finite", "minimum", "maximum", "mean", "m2",
                 "int_sum", "int_sumsq", "min_re", "max_re", "mean_re", "m2_re",
                 "min_im", "max_im", "mean_im", "m2_im", "count", "status")

    def __init__(self, value):
        self.dtype = value.dtype
        self.shape = tuple(int(item) for item in value.shape)
        self.numel = int(value.numel())
        self.complex = bool(value.is_complex())
        self.int = (not value.is_floating_point() and not self.complex
                    and not value.dtype.is_complex)
        self.nan = self.posinf = self.neginf = self.finite = None
        self.minimum = self.maximum = self.mean = self.m2 = None
        self.int_sum = self.int_sumsq = None
        self.min_re = self.max_re = self.mean_re = self.m2_re = None
        self.min_im = self.max_im = self.mean_im = self.m2_im = None
        self.count = 0
        self.status = "ok"
        self._initialize(value)

    def _initialize(self, value):
        torch = _torch()
        device = value.device
        if self.int:
            info = torch.iinfo(torch.int64)
            self.minimum = torch.full(
                (self.numel,), info.max, dtype=torch.int64, device=device)
            self.maximum = torch.full(
                (self.numel,), info.min, dtype=torch.int64, device=device)
            self.int_sum = [0] * self.numel
            self.int_sumsq = [0] * self.numel
            return
        self.minimum = torch.full(
            (self.numel,), _INF, dtype=torch.float64, device=device)
        self.maximum = torch.full(
            (self.numel,), -_INF, dtype=torch.float64, device=device)
        self.nan = torch.zeros((self.numel,), dtype=torch.int64, device=device)
        self.posinf = torch.zeros((self.numel,), dtype=torch.int64, device=device)
        self.neginf = torch.zeros((self.numel,), dtype=torch.int64, device=device)
        self.finite = torch.zeros((self.numel,), dtype=torch.int64, device=device)
        self.mean = torch.zeros((self.numel,), dtype=torch.float64, device=device)
        self.m2 = torch.zeros((self.numel,), dtype=torch.float64, device=device)
        if self.complex:
            for name in ("min_re", "max_re", "min_im", "max_im"):
                setattr(self, name, torch.full(
                    (self.numel,), _INF if name.startswith("min") else -_INF,
                    dtype=torch.float64, device=device))
            for name in ("mean_re", "m2_re", "mean_im", "m2_im"):
                setattr(self, name, torch.zeros(
                    (self.numel,), dtype=torch.float64, device=device))

    @staticmethod
    def _update_moments(series, finite, count, mean, m2):
        torch = _torch()
        delta = torch.where(finite, series - mean, 0.0)
        mean = mean + delta / count
        delta2 = torch.where(finite, series - mean, 0.0)
        return mean, m2 + delta * delta2

    def update(self, value):
        torch = _torch()
        if self.status != "ok":
            return
        if not isinstance(value, torch.Tensor):
            self.status = "shape_mismatch"
            return
        if value.dtype != self.dtype or tuple(int(item) for item in value.shape) != self.shape:
            self.status = "shape_mismatch"
            return
        self.count += 1
        if self.int:
            current = value.detach().to(torch.int64).reshape(-1)
            self.minimum = torch.minimum(self.minimum, current)
            self.maximum = torch.maximum(self.maximum, current)
            for index, item in enumerate(current.tolist()):
                number = int(item)
                self.int_sum[index] += number
                self.int_sumsq[index] += number * number
            return
        if self.complex:
            current = value.detach().to(torch.complex128).reshape(-1)
            real, imag = current.real, current.imag
            nan = torch.isnan(real) | torch.isnan(imag)
            posinf = torch.isposinf(real) | torch.isposinf(imag)
            neginf = torch.isneginf(real) | torch.isneginf(imag)
            finite = ~(nan | posinf | neginf)
        else:
            current = value.detach().to(torch.float64).reshape(-1)
            real = current
            nan = torch.isnan(current)
            posinf = torch.isposinf(current)
            neginf = torch.isneginf(current)
            finite = ~(nan | posinf | neginf)
        self.nan += nan.to(torch.int64)
        self.posinf += posinf.to(torch.int64)
        self.neginf += neginf.to(torch.int64)
        self.finite += finite.to(torch.int64)
        count = self.finite.clamp_min(1).to(torch.float64)
        if self.complex:
            for series, minimum, maximum, mean, m2 in (
                    (real, "min_re", "max_re", "mean_re", "m2_re"),
                    (imag, "min_im", "max_im", "mean_im", "m2_im")):
                setattr(self, minimum, torch.minimum(
                    getattr(self, minimum),
                    torch.where(finite, series, torch.full_like(series, _INF))))
                setattr(self, maximum, torch.maximum(
                    getattr(self, maximum),
                    torch.where(finite, series, torch.full_like(series, -_INF))))
                moments = self._update_moments(
                    series, finite, count, getattr(self, mean), getattr(self, m2))
                setattr(self, mean, moments[0])
                setattr(self, m2, moments[1])
            return
        self.minimum = torch.minimum(
            self.minimum, torch.where(finite, current, torch.full_like(current, _INF)))
        self.maximum = torch.maximum(
            self.maximum, torch.where(finite, current, torch.full_like(current, -_INF)))
        self.mean, self.m2 = self._update_moments(
            current, finite, count, self.mean, self.m2)

    def _float_vectors(self):
        torch = _torch()
        total = self.nan + self.posinf + self.neginf + self.finite
        nonfinite = self.nan + self.posinf + self.neginf
        all_nan = (self.nan == total) & (total > 0)
        only_pos = (self.posinf == total) & (total > 0)
        only_neg = (self.neginf == total) & (total > 0)
        homogeneous = all_nan | only_pos | only_neg
        safe_count = self.finite.clamp_min(1).to(torch.float64)
        if self.complex:
            finite_range = torch.hypot(
                (self.max_re - self.min_re).clamp_min(0),
                (self.max_im - self.min_im).clamp_min(0))
            variance = ((self.m2_re + self.m2_im) / safe_count).clamp_min(0)
        else:
            finite_range = (self.maximum - self.minimum).clamp_min(0)
            variance = (self.m2 / safe_count).clamp_min(0)
        range_values = torch.where(
            homogeneous, torch.zeros_like(finite_range),
            torch.where(nonfinite == 0, finite_range,
                        torch.full_like(finite_range, _INF)))
        std_values = torch.where(
            self.finite > 0, variance.sqrt(),
            torch.zeros_like(variance))
        std_values = torch.where(
            homogeneous | (nonfinite == 0), std_values,
            torch.full_like(std_values, _INF))
        return range_values, std_values

    def report(self, path):
        torch = _torch()
        if self.status != "ok":
            return {"tree_path": path, "status": self.status,
                    "max_range": _INF, "p50_range": _INF, "max_std": _INF,
                    "sample_count": self.numel}
        if self.int:
            minimum = [int(item) for item in self.minimum.tolist()]
            maximum = [int(item) for item in self.maximum.tolist()]
            ranges = [max(0, high - low) for low, high in zip(minimum, maximum)]
            range_tensor = torch.tensor(ranges, dtype=torch.float64)
            count = max(self.count, 1)
            stds = []
            for total, total_sq in zip(self.int_sum, self.int_sumsq):
                var_num = count * total_sq - total * total
                stds.append(math.sqrt(var_num) / count if var_num > 0 else 0.0)
            std_tensor = torch.tensor(stds, dtype=torch.float64)
        else:
            range_tensor, std_tensor = self._float_vectors()
        if int(range_tensor.numel()) == 0:
            return {"tree_path": path, "status": "ok", "max_range": 0.0,
                    "p50_range": 0.0, "max_std": 0.0, "sample_count": 0}
        return {"tree_path": path, "status": "ok",
                "max_range": float(range_tensor.max().item()),
                "p50_range": float(range_tensor.median().item()),
                "max_std": float(std_tensor.max().item()),
                "sample_count": self.numel}
class _ScalarState:
    __slots__ = ("values", "status")

    def __init__(self):
        self.values = []
        self.status = "ok"

    def update(self, value):
        if self.status != "ok":
            return
        if isinstance(value, _torch().Tensor):
            self.status = "shape_mismatch"
            return
        self.values.append(value)

    def report(self, path):
        if self.status != "ok":
            return {"tree_path": path, "status": self.status,
                    "max_range": _INF, "p50_range": _INF, "max_std": _INF,
                    "sample_count": 1}
        numeric = [_scalar_numeric(value) for value in self.values]
        if any(value is None for value in numeric):
            equal = all(_scalar_equal(self.values[0], value) for value in self.values[1:])
            return {"tree_path": path, "status": "ok",
                    "max_range": 0.0 if equal else _INF,
                    "p50_range": 0.0 if equal else _INF,
                    "max_std": 0.0 if equal else _INF, "sample_count": 1}
        equal = all(_scalar_equal(self.values[0], value) for value in self.values[1:])
        if equal:
            return {"tree_path": path, "status": "ok", "max_range": 0.0,
                    "p50_range": 0.0, "max_std": 0.0, "sample_count": 1}
        if all(isinstance(value, int) for value in numeric):
            low, high = min(numeric), max(numeric)
            count = len(numeric)
            variance_numerator = (count * sum(value * value for value in numeric)
                                  - sum(numeric) ** 2)
            std = (math.sqrt(variance_numerator) / count
                   if variance_numerator > 0 else 0.0)
            return {"tree_path": path, "status": "ok",
                    "max_range": float(high - low),
                    "p50_range": float(high - low),
                    "max_std": std, "sample_count": 1}
        try:
            finite = [float(value) for value in numeric]
        except (TypeError, ValueError):
            finite = []
        if len(finite) != len(numeric) or any(not math.isfinite(value) for value in finite):
            return {"tree_path": path, "status": "ok", "max_range": _INF,
                    "p50_range": _INF, "max_std": _INF, "sample_count": 1}
        low, high = min(finite), max(finite)
        mean = sum(finite) / len(finite)
        std = math.sqrt(max(0.0, sum((value - mean) ** 2 for value in finite)
                           / len(finite)))
        return {"tree_path": path, "status": "ok", "max_range": high - low,
                "p50_range": high - low, "max_std": std, "sample_count": 1}
class JitterAccumulator:
    """One-pass range/std accumulator over output-value dicts."""

    def __init__(self):
        self._states = {}
        self._expected_paths = None
        self.structure_mismatch = False
        self.updates = 0

    def update(self, values):
        self.updates += 1
        paths = set(values)
        if self._expected_paths is None:
            self._expected_paths = paths
        elif paths != self._expected_paths:
            self.structure_mismatch = True
        for path, value in values.items():
            state = self._states.get(path)
            if state is None:
                if isinstance(value, _torch().Tensor):
                    state = _TensorState(value)
                else:
                    state = _ScalarState()
                self._states[path] = state
            state.update(value)

    def report(self):
        if self.updates < 2:
            return {"status": "insufficient_repeats", "reason": "repeat_lt_2",
                    "repeats": self.updates, "max_range": 0.0, "p50_range": 0.0,
                    "max_std": 0.0, "per_path": []}
        per_path = [self._states[path].report(path)
                    for path in sorted(self._states)]
        shape_mismatch = any(item.get("status") != "ok" for item in per_path)
        if shape_mismatch:
            status = "shape_mismatch"
        elif self.structure_mismatch:
            status = "output_structure_mismatch"
        else:
            status = "ok"
        valid = [item for item in per_path if item.get("status") == "ok"]
        if status != "ok":
            max_range = p50_range = max_std = _INF
        else:
            max_range = max((item["max_range"] for item in valid), default=0.0)
            p50_range = max((item["p50_range"] for item in valid), default=0.0)
            max_std = max((item["max_std"] for item in valid), default=0.0)
        return {"status": status, "repeats": self.updates,
                "max_range": max_range, "p50_range": p50_range,
                "max_std": max_std, "per_path": per_path}
