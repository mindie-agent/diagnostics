"""C2 ``npu-top`` telemetry-window review for throttling and measurement limits.

[Role]
Read-only consumer of the ``npu-top`` telemetry-window contract.  It turns
``coverage_ratio``, ``throttling_detected`` (bool or null), per-device
statistics, and ``measurement_limitations`` into a conservative thermal guard
receipt.  It also provides an explicit, bounded HTTP GET client for one
``(host, devices, start_ts, end_ts)`` window.

[Invariants]
- A temperature at or above 85 C is only an observation, never proof of
  throttling.  Only the device/telemetry throttling signal can prove it.
- A negative verdict requires complete coverage, a known clock/frequency, the
  requested devices, no blocking measurement limitation, and an explicit
  ``throttling_detected is False`` signal.
- Missing coverage, unknown clocks, absent low-level signals, and null throttle
  bits remain ``inconclusive``; they are never converted to "healthy".
- A positive throttle bit remains positive evidence even when coverage is
  partial, but the limitation is still reported.
- Missing values remain ``None``; no zero/false fallback is fabricated.

[Timeouts & Locks]
The HTTP client uses a bounded timeout (default 5 s, maximum 30 s), a bounded
response body, one GET attempt, and no automatic retry.  Review itself is pure
CPU with no locks or sleeps.

[Side-Effect Policy]
No NPU scheduling, no host mutation, no writes, and no background work.  The
only optional external effect is an explicit caller-requested read-only GET.
"""

from __future__ import annotations

import json
import math
import urllib.parse
import urllib.request
from typing import Any, Callable, Mapping, Sequence

TELEMETRY_REVIEW_SCHEMA = "mindie-thermal-guard/1"
DEFAULT_COVERAGE_THRESHOLD = 1.0
DEFAULT_TIMEOUT_S = 5.0
MAX_TIMEOUT_S = 30.0
DEFAULT_MAX_RESPONSE_BYTES = 2_000_000
TEMPERATURE_WARNING_C = 85.0
TELEMETRY_ENDPOINT = "/api/agent/telemetry-window"


class TelemetryFetchError(RuntimeError):
    """Raised when one explicit telemetry-window GET cannot be completed."""


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _number(value: Any) -> float | None:
    return float(value) if _is_number(value) else None


def _integer(value: Any) -> int | None:
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else None


def _limitation(
    code: str,
    message: str,
    *,
    device_id: int | None = None,
    details: Any = None,
) -> dict[str, Any]:
    item: dict[str, Any] = {"code": code, "message": message}
    if device_id is not None:
        item["device_id"] = device_id
    if details is not None:
        item["details"] = details
    return item


def _dedupe(items: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    result: list[dict[str, Any]] = []
    for item in items:
        key = (item.get("code"), item.get("device_id"))
        if key in seen:
            continue
        seen.add(key)
        result.append(dict(item))
    return result


def _payload_window(window: Mapping[str, Any]) -> Mapping[str, Any]:
    value = window.get("window")
    return value if isinstance(value, Mapping) else {}


def _expected_devices(
    window: Mapping[str, Any],
    explicit: Sequence[int] | None,
) -> list[int] | None:
    if explicit is not None:
        devices: list[int] = []
        for item in explicit:
            if not isinstance(item, int) or isinstance(item, bool):
                raise ValueError("expected_devices must contain integers")
            devices.append(item)
        return sorted(set(devices))
    payload_window = _payload_window(window)
    for source in (payload_window, window):
        value = source.get("devices")
        if isinstance(value, list) and all(
            isinstance(item, int) and not isinstance(item, bool) for item in value
        ):
            return sorted(set(value))
    value = payload_window.get("requested_devices")
    if isinstance(value, list) and all(
        isinstance(item, int) and not isinstance(item, bool) for item in value
    ):
        return sorted(set(value))
    return None


def _device_statistics(window: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw: Any = None
    value = window.get("devices")
    if isinstance(value, list) and any(isinstance(item, Mapping) for item in value):
        raw = value
    elif isinstance(value, Mapping):
        raw = value
    if raw is None:
        return []
    if isinstance(raw, Mapping):
        result: list[Mapping[str, Any]] = []
        for key, value in raw.items():
            if isinstance(value, Mapping):
                item = dict(value)
                if not any(name in item for name in ("device_id", "device", "id", "npu_id")):
                    device_id = _integer(key)
                    if device_id is None:
                        device_id = int(key) if isinstance(key, str) and key.isdigit() else None
                    if device_id is not None:
                        item["device_id"] = device_id
                result.append(item)
        return result
    return [item for item in raw if isinstance(item, Mapping)]


def _aggregate_device_statistics(window: Mapping[str, Any], expected: list[int] | None) -> list[Mapping[str, Any]]:
    """Use top-level C2 aggregates only when they provably describe one device."""
    if expected is None or len(expected) != 1:
        return []
    fields = ("throttling_detected", "freq_mhz_min", "power_max_w", "temp_max_c",
              "aicore_p50", "aicore_p95", "hbm_peak_mb", "coverage_ratio")
    if all(window.get(field) is None for field in fields):
        return []
    return [{
        "device_id": expected[0],
        **{field: window.get(field) for field in fields},
        "measurement_limitations": window.get("measurement_limitations", []),
    }]


def _device_id(item: Mapping[str, Any]) -> int | None:
    for key in ("npu_id", "device_id"):
        value = _integer(item.get(key))
        if value is not None:
            return value
    return None


def _coverage_ratio(window: Mapping[str, Any]) -> Any:
    if "coverage_ratio" in window:
        return window["coverage_ratio"]
    coverage = window.get("coverage")
    if isinstance(coverage, Mapping):
        return coverage.get("ratio", coverage.get("coverage_ratio"))
    return None


def _window_meta(window: Mapping[str, Any]) -> dict[str, Any]:
    payload_window = _payload_window(window)
    server = window.get("server") if isinstance(window.get("server"), Mapping) else {}
    meta: dict[str, Any] = {}
    for key in ("host", "start_ts", "end_ts", "devices"):
        for source in (window, payload_window, server):
            if key in source:
                meta[key] = source[key]
                break
    if "requested_devices" in payload_window:
        meta["devices"] = payload_window["requested_devices"]
    if server:
        for key, source_key in (("server_id", "id"), ("server_name", "name"), ("server_host", "host")):
            if source_key in server:
                meta[key] = server[source_key]
    return meta


def _review_device(
    device_id: int,
    item: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if item is None:
        return {
            "device_id": device_id,
            "status": "missing",
            "throttling_detected": None,
            "temp_max_c": None,
            "freq_mhz_min": None,
            "power_max_w": None,
            "aicore_p50": None,
            "aicore_p95": None,
            "hbm_peak_mb": None,
            "coverage_ratio": None,
            "sample_count": None,
            "clock_known": None,
            "measurement_limitations": [],
        }
    declared_clock = item.get("clock_known")
    freq = _number(item.get("freq_mhz_min"))
    if isinstance(declared_clock, bool):
        clock_known: bool | None = declared_clock
    else:
        clock_known = True if freq is not None and freq > 0 else None
    return {
        "device_id": device_id,
        "status": "observed",
        "throttling_detected": item.get("throttling_detected")
        if isinstance(item.get("throttling_detected"), bool)
        else None,
        "temp_max_c": _number(item.get("temp_max_c")),
        "freq_mhz_min": freq,
        "power_max_w": _number(item.get("power_max_w")),
        "aicore_p50": _number(item.get("aicore_p50")),
        "aicore_p95": _number(item.get("aicore_p95")),
        "hbm_peak_mb": _number(item.get("hbm_peak_mb")),
        "coverage_ratio": _number(item.get("coverage_ratio")),
        "sample_count": _integer(item.get("sample_count")),
        "clock_known": clock_known,
        "measurement_limitations": [entry for entry in item.get("measurement_limitations", [])
                                    if isinstance(entry, str)]
        if isinstance(item.get("measurement_limitations"), list) else [],
    }


def _no_data(reason: str) -> dict[str, Any]:
    limitation = _limitation("telemetry_window_unavailable", reason)
    return {
        "schema_version": TELEMETRY_REVIEW_SCHEMA,
        "status": "no_data",
        "conclusive": False,
        "confidence": "none",
        "summary": reason,
        "coverage": {"ratio": None, "threshold": DEFAULT_COVERAGE_THRESHOLD, "complete": None},
        "throttling": {"detected": None, "source": None, "proven": False},
        "clock": {"known": None, "devices_with_frequency": [], "missing_devices": []},
        "temperature": {
            "max_c": None,
            "at_or_above_85c": None,
            "used_as_throttling_proof": False,
        },
        "window": {},
        "devices": [],
        "measurement_limitations": [],
        "limitations": [limitation],
        "findings": [{
            "finding_type": "measurement_limitation",
            "severity": "medium",
            "confidence": "high",
            "summary": reason,
        }],
    }


def review_telemetry_window(
    window: Any,
    *,
    expected_host: str | None = None,
    expected_devices: Sequence[int] | None = None,
    expected_start_ts: int | None = None,
    expected_end_ts: int | None = None,
    coverage_threshold: float = DEFAULT_COVERAGE_THRESHOLD,
) -> dict[str, Any]:
    """Review one C2 telemetry-window payload without inventing observations."""
    if not _is_number(coverage_threshold) or not 0.0 <= float(coverage_threshold) <= 1.0:
        raise ValueError("coverage_threshold must be a number between 0 and 1")
    if not isinstance(window, Mapping):
        return _no_data(
            f"telemetry window is {type(window).__name__}, expected an object"
        )

    limitations: list[dict[str, Any]] = []
    blocking: list[dict[str, Any]] = []

    raw_coverage = _coverage_ratio(window)
    coverage = _number(raw_coverage)
    coverage_complete: bool | None = None
    if raw_coverage is None:
        blocking.append(_limitation(
            "coverage_ratio_missing",
            "coverage_ratio is missing; complete coverage cannot be established",
        ))
    elif coverage is None or coverage < 0.0 or coverage > 1.0:
        blocking.append(_limitation(
            "coverage_ratio_invalid",
            "coverage_ratio must be a number between 0 and 1",
            details={"value": raw_coverage},
        ))
    else:
        coverage_complete = coverage >= float(coverage_threshold) - 1e-12
        if not coverage_complete:
            blocking.append(_limitation(
                "coverage_incomplete",
                f"coverage_ratio {coverage:.6f} is below the required {coverage_threshold:.6f}",
                details={"coverage_ratio": coverage, "threshold": coverage_threshold},
            ))

    measurement_limitations: list[str] = []
    raw_limits = window.get("measurement_limitations")
    if raw_limits is None:
        blocking.append(_limitation(
            "measurement_limitations_missing",
            "measurement_limitations is missing; unsupported low-level signals are unknown",
        ))
    elif not isinstance(raw_limits, list):
        blocking.append(_limitation(
            "measurement_limitations_invalid",
            "measurement_limitations must be a list of strings",
        ))
    else:
        invalid = [item for item in raw_limits if not isinstance(item, str)]
        measurement_limitations = [item for item in raw_limits if isinstance(item, str)]
        if invalid:
            blocking.append(_limitation(
                "measurement_limitation_invalid_entry",
                "measurement_limitations contains a non-string entry",
            ))
        if measurement_limitations:
            limitations.append(_limitation(
                "measurement_limitations_reported",
                "the telemetry source reports measurement limitations",
                details={"measurement_limitations": list(measurement_limitations)},
            ))
            blocking_markers = (
                "no raw device samples",
                "no raw samples were observed",
                "throttling flag",
                "throttling_detected",
                "frequency register",
                "freq_mhz",
                "power register",
                "power_max_w",
                "coverage is below 100%",
            )
            severe = [
                text for text in measurement_limitations
                if any(marker in text.lower() for marker in blocking_markers)
            ]
            if severe:
                blocking.append(_limitation(
                    "measurement_limitation_blocks_throttle_verdict",
                    "a low-level throttle/clock/power signal is reported as unavailable",
                    details={"measurement_limitations": severe},
                ))

    raw_throttle = window.get("throttling_detected")
    window_throttle: bool | None
    if raw_throttle is None:
        window_throttle = None
        blocking.append(_limitation(
            "throttling_signal_missing",
            "throttling_detected is absent or null; throttle state is unknown",
        ))
    elif isinstance(raw_throttle, bool):
        window_throttle = raw_throttle
    else:
        window_throttle = None
        blocking.append(_limitation(
            "throttling_signal_invalid",
            "throttling_detected must be a boolean or null",
            details={"value": raw_throttle},
        ))

    requested_meta = _window_meta(window)
    expected = _expected_devices(window, expected_devices)
    stats = _device_statistics(window)
    if not stats:
        stats = _aggregate_device_statistics(window, expected)
    if expected is None:
        expected = sorted({
            device_id for device_id in (_device_id(item) for item in stats)
            if device_id is not None
        })
    expected_set = set(expected or [])
    stats_by_id: dict[int, Mapping[str, Any]] = {}
    unassigned_stats: list[Mapping[str, Any]] = []
    for item in stats:
        device_id = _device_id(item)
        if device_id is None:
            unassigned_stats.append(item)
            continue
        stats_by_id.setdefault(device_id, item)
    if unassigned_stats:
        limitations.append(_limitation(
            "device_statistics_without_id",
            "some device statistics have no usable device id",
            details={"count": len(unassigned_stats)},
        ))
    if expected is not None:
        missing = sorted(expected_set - set(stats_by_id))
        for device_id in missing:
            blocking.append(_limitation(
                "device_statistics_missing",
                f"no telemetry statistics for requested device {device_id}",
                device_id=device_id,
            ))
        extra = sorted(set(stats_by_id) - expected_set)
        if extra:
            limitations.append(_limitation(
                "unexpected_device_statistics",
                "telemetry contains statistics for devices outside the requested set",
                details={"devices": extra},
            ))

    reviewed_devices = [
        _review_device(device_id, stats_by_id.get(device_id))
        for device_id in (expected if expected is not None else sorted(stats_by_id))
    ]
    observed = [item for item in reviewed_devices if item["status"] == "observed"]

    device_throttle = [item["throttling_detected"] for item in observed]
    if any(value is True for value in device_throttle):
        throttling_detected: bool | None = True
        throttle_source = "device_signal"
    elif window_throttle is True:
        throttling_detected = True
        throttle_source = "window_signal"
    elif window_throttle is False and all(value is not True for value in device_throttle):
        throttling_detected = False
        throttle_source = "window_signal"
    elif window_throttle is None:
        throttling_detected = None
        throttle_source = None
    else:
        throttling_detected = None
        throttle_source = None
    if throttling_detected is True and any(value is False for value in device_throttle):
        limitations.append(_limitation(
            "conflicting_throttling_signals",
            "window and device throttle signals disagree; the positive signal is retained",
        ))

    declared_clock = window.get("clock_known")
    if declared_clock is not None and not isinstance(declared_clock, bool):
        limitations.append(_limitation(
            "clock_signal_invalid",
            "clock_known is present but is not a boolean",
            details={"value": declared_clock},
        ))
        declared_clock = None
    if isinstance(declared_clock, bool) and declared_clock is False:
        clock_known: bool | None = False
    elif isinstance(declared_clock, bool) and declared_clock is True:
        clock_known = True
    elif observed and all(item["clock_known"] is True for item in observed):
        clock_known = True
    elif observed and any(item["clock_known"] is False for item in observed):
        clock_known = False
    else:
        clock_known = None
    if clock_known is not True:
        blocking.append(_limitation(
            "clock_unknown",
            "frequency/clock evidence is missing or unknown; a negative throttle verdict is not safe",
        ))
    missing_power = [
        item["device_id"] for item in observed if item["power_max_w"] is None
    ]
    if missing_power:
        blocking.append(_limitation(
            "power_signal_missing",
            "per-device power evidence is missing; a negative throttle verdict is not safe",
            details={"devices": missing_power},
        ))

    if expected_host is not None:
        advertised = {
            value for value in (
                requested_meta.get("host"),
                requested_meta.get("server_host"),
                requested_meta.get("server_id"),
                requested_meta.get("server_name"),
            ) if isinstance(value, str) and value
        }
        if advertised and expected_host not in advertised:
            blocking.append(_limitation(
                "window_host_mismatch",
                "telemetry window host does not match the requested host",
                details={"expected": expected_host, "actual": sorted(advertised)},
            ))
    for key, expected_value in (
        ("start_ts", expected_start_ts),
        ("end_ts", expected_end_ts),
    ):
        if expected_value is None:
            continue
        actual = requested_meta.get(key)
        if actual is None:
            limitations.append(_limitation(
                "window_bounds_unverified",
                f"telemetry response does not echo {key}; window coverage is unverified",
            ))
        elif actual != expected_value:
            blocking.append(_limitation(
                f"window_{key}_mismatch",
                f"telemetry window {key} does not match the request",
                details={"expected": expected_value, "actual": actual},
            ))
    if expected_host is not None and "host" not in requested_meta:
        limitations.append(_limitation(
            "window_host_unverified",
            "telemetry response does not echo the host; host provenance is unverified",
        ))

    temperatures = [
        item["temp_max_c"] for item in observed if item["temp_max_c"] is not None
    ]
    max_temp = max(temperatures) if temperatures else None
    temp_high = max_temp is not None and max_temp >= TEMPERATURE_WARNING_C
    frequencies = [
        item["device_id"] for item in observed
        if item["freq_mhz_min"] is not None and item["freq_mhz_min"] > 0
    ]
    missing_frequency = [
        item["device_id"] for item in observed
        if item["freq_mhz_min"] is None or item["freq_mhz_min"] <= 0
    ]

    findings: list[dict[str, Any]] = []
    if temp_high:
        findings.append({
            "finding_type": "measurement_limitation",
            "severity": "low",
            "confidence": "high",
            "summary": (
                f"max temperature {max_temp:.2f} C is at or above {TEMPERATURE_WARNING_C:.0f} C; "
                "temperature alone is not proof of throttling"
            ),
        })

    if throttling_detected is True:
        status = "throttled_observed"
        conclusive = True
        confidence = "high" if coverage_complete and not blocking else "limited"
        findings.insert(0, {
            "finding_type": "thermal_or_power_throttling",
            "severity": "high",
            "confidence": confidence,
            "summary": "npu-top telemetry reports throttling detected for the requested window",
        })
        summary = "throttling was observed; partial coverage or missing signals do not erase a positive throttle bit"
    elif throttling_detected is False and not blocking:
        status = "no_throttling_observed"
        conclusive = True
        confidence = "high"
        summary = "complete coverage, known clock, and an explicit false throttle signal"
    else:
        status = "inconclusive"
        conclusive = False
        confidence = "none"
        findings.insert(0, {
            "finding_type": "measurement_limitation",
            "severity": "medium",
            "confidence": "high",
            "summary": "throttling state cannot be concluded from the available telemetry window",
        })
        summary = "inconclusive: " + "; ".join(item["code"] for item in blocking)

    all_limitations = _dedupe([*limitations, *blocking])
    return {
        "schema_version": TELEMETRY_REVIEW_SCHEMA,
        "status": status,
        "conclusive": conclusive,
        "confidence": confidence,
        "summary": summary,
        "coverage": {
            "ratio": coverage,
            "threshold": float(coverage_threshold),
            "complete": coverage_complete,
        },
        "throttling": {
            "detected": throttling_detected,
            "source": throttle_source,
            "proven": throttling_detected is True,
        },
        "clock": {
            "known": clock_known,
            "devices_with_frequency": frequencies,
            "missing_devices": missing_frequency,
        },
        "temperature": {
            "max_c": max_temp,
            "at_or_above_85c": temp_high,
            "used_as_throttling_proof": False,
        },
        "window": {
            "host": requested_meta.get("host"),
            "requested_host": expected_host,
            "start_ts": requested_meta.get("start_ts"),
            "end_ts": requested_meta.get("end_ts"),
            "devices": expected,
            "device_statistics_count": len(observed),
        },
        "devices": reviewed_devices,
        "measurement_limitations": measurement_limitations,
        "limitations": all_limitations,
        "findings": findings,
    }


class TelemetryClient:
    """Explicit read-only HTTP GET client for one telemetry-window query."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = DEFAULT_TIMEOUT_S,
        max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
        opener: Callable[..., Any] | None = None,
    ) -> None:
        if not isinstance(base_url, str) or not base_url:
            raise ValueError("base_url must be a non-empty HTTP(S) URL")
        parsed = urllib.parse.urlsplit(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("base_url must use http or https and include a host")
        if parsed.fragment or parsed.query:
            raise ValueError("base_url must not contain a fragment or query")
        if not _is_number(timeout) or not 0.0 < float(timeout) <= MAX_TIMEOUT_S:
            raise ValueError(f"timeout must be within (0, {MAX_TIMEOUT_S:g}] seconds")
        if not isinstance(max_response_bytes, int) or isinstance(max_response_bytes, bool):
            raise ValueError("max_response_bytes must be a positive integer")
        if max_response_bytes <= 0:
            raise ValueError("max_response_bytes must be a positive integer")
        self.base_url = base_url
        self.timeout = float(timeout)
        self.max_response_bytes = max_response_bytes
        self._opener = opener or urllib.request.urlopen

    def _url(self, host: str, devices: Sequence[int], start_ts: int, end_ts: int) -> str:
        parsed = urllib.parse.urlsplit(self.base_url)
        path = parsed.path.rstrip("/") + TELEMETRY_ENDPOINT
        query = urllib.parse.urlencode({
            "host": host,
            "devices": ",".join(str(item) for item in devices),
            "start_ts": start_ts,
            "end_ts": end_ts,
        })
        return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, path, query, ""))

    def fetch_window(
        self,
        host: str,
        devices: Sequence[int],
        start_ts: int,
        end_ts: int,
    ) -> dict[str, Any]:
        """Perform exactly one read-only GET; never retry or mutate the host."""
        if not isinstance(host, str) or not host:
            raise ValueError("host must be a non-empty string")
        if isinstance(devices, (str, bytes)) or not isinstance(devices, Sequence):
            raise ValueError("devices must be a sequence of integers")
        normalized: list[int] = []
        for item in devices:
            if not isinstance(item, int) or isinstance(item, bool):
                raise ValueError("devices must contain integers")
            normalized.append(item)
        if not normalized:
            raise ValueError("devices must not be empty")
        if not isinstance(start_ts, int) or isinstance(start_ts, bool):
            raise ValueError("start_ts must be an integer")
        if not isinstance(end_ts, int) or isinstance(end_ts, bool):
            raise ValueError("end_ts must be an integer")
        if start_ts >= end_ts:
            raise ValueError("start_ts must be before end_ts")

        url = self._url(host, sorted(set(normalized)), start_ts, end_ts)
        request = urllib.request.Request(
            url,
            headers={"Accept": "application/json"},
            method="GET",
        )
        try:
            with self._opener(request, timeout=self.timeout) as response:
                status = getattr(response, "status", None)
                if status is None and hasattr(response, "getcode"):
                    status = response.getcode()
                if status is not None and int(status) != 200:
                    raise TelemetryFetchError(
                        f"telemetry request failed with HTTP status {status}"
                    )
                body = response.read(self.max_response_bytes + 1)
        except TelemetryFetchError:
            raise
        except Exception as exc:  # one attempt only; surface the concrete failure
            raise TelemetryFetchError(f"telemetry request failed: {exc}") from exc
        if not isinstance(body, (bytes, bytearray)):
            raise TelemetryFetchError("telemetry response body is not bytes")
        if len(body) > self.max_response_bytes:
            raise TelemetryFetchError(
                f"telemetry response exceeds {self.max_response_bytes} bytes"
            )
        try:
            payload = json.loads(bytes(body).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise TelemetryFetchError(f"telemetry response is not valid UTF-8 JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise TelemetryFetchError(
                f"telemetry response is {type(payload).__name__}, expected an object"
            )
        return payload

    def fetch_and_review(
        self,
        host: str,
        devices: Sequence[int],
        start_ts: int,
        end_ts: int,
        **review_options: Any,
    ) -> dict[str, Any]:
        window = self.fetch_window(host, devices, start_ts, end_ts)
        return review_telemetry_window(
            window,
            expected_host=host,
            expected_devices=list(devices),
            expected_start_ts=start_ts,
            expected_end_ts=end_ts,
            **review_options,
        )


def fetch_telemetry_window(
    base_url: str,
    host: str,
    devices: Sequence[int],
    start_ts: int,
    end_ts: int,
    *,
    timeout: float = DEFAULT_TIMEOUT_S,
    max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
    opener: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Fetch one telemetry window with an explicit bounded read-only client."""
    client = TelemetryClient(
        base_url,
        timeout=timeout,
        max_response_bytes=max_response_bytes,
        opener=opener,
    )
    return client.fetch_window(host, devices, start_ts, end_ts)


__all__ = [
    "DEFAULT_COVERAGE_THRESHOLD",
    "DEFAULT_TIMEOUT_S",
    "MAX_TIMEOUT_S",
    "TELEMETRY_REVIEW_SCHEMA",
    "TEMPERATURE_WARNING_C",
    "TelemetryClient",
    "TelemetryFetchError",
    "fetch_telemetry_window",
    "review_telemetry_window",
]
