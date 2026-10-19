import json
import urllib.parse
from pathlib import Path

import pytest

from mindie_profiler.thermal_guard import (
    TELEMETRY_REVIEW_SCHEMA,
    TelemetryClient,
    TelemetryFetchError,
    fetch_telemetry_window,
    review_telemetry_window,
)


def _device(device_id=0, **overrides):
    item = {
        "device_id": device_id,
        "temp_max_c": 70.0,
        "freq_mhz_min": 1800.0,
        "power_max_w": 300.0,
        "aicore_p50": 0.70,
        "aicore_p95": 0.90,
        "hbm_peak_mb": 1024.0,
        "throttling_detected": None,
    }
    item.update(overrides)
    return item


def _window(**overrides):
    window = {
        "host": "a3-165",
        "start_ts": 1_000,
        "end_ts": 2_000,
        "devices": [_device(0)],
        "coverage_ratio": 1.0,
        "throttling_detected": False,
        "measurement_limitations": [],
    }
    window.update(overrides)
    return window


def _review(window, **kwargs):
    kwargs.setdefault("expected_host", "a3-165")
    kwargs.setdefault("expected_devices", [0])
    kwargs.setdefault("expected_start_ts", 1_000)
    kwargs.setdefault("expected_end_ts", 2_000)
    return review_telemetry_window(window, **kwargs)


def _codes(result):
    return {item["code"] for item in result["limitations"]}


def test_complete_window_with_false_signal_is_negative():
    result = _review(_window())

    assert result["schema_version"] == TELEMETRY_REVIEW_SCHEMA
    assert result["status"] == "no_throttling_observed"
    assert result["conclusive"] is True
    assert result["coverage"]["ratio"] == 1.0
    assert result["coverage"]["complete"] is True
    assert result["throttling"] == {"detected": False, "source": "window_signal", "proven": False}
    assert result["clock"]["known"] is True
    assert result["devices"][0]["status"] == "observed"
    assert result["devices"][0]["freq_mhz_min"] == 1800.0


def test_85c_alone_is_not_throttling_proof():
    result = _review(_window(devices=[_device(0, temp_max_c=85.0)]))

    assert result["temperature"]["max_c"] == 85.0
    assert result["temperature"]["at_or_above_85c"] is True
    assert result["temperature"]["used_as_throttling_proof"] is False
    assert result["throttling"]["proven"] is False
    assert result["status"] != "throttled_observed"


def test_null_throttle_signal_remains_inconclusive():
    result = _review(_window(throttling_detected=None))

    assert result["status"] == "inconclusive"
    assert result["conclusive"] is False
    assert result["throttling"]["detected"] is None
    assert "throttling_signal_missing" in _codes(result)


def test_incomplete_coverage_remains_inconclusive():
    result = _review(_window(coverage_ratio=0.42))

    assert result["status"] == "inconclusive"
    assert result["coverage"]["complete"] is False
    assert "coverage_incomplete" in _codes(result)


def test_unknown_clock_remains_inconclusive():
    device = _device(0)
    device.pop("freq_mhz_min")
    result = _review(_window(devices=[device]))

    assert result["status"] == "inconclusive"
    assert result["clock"]["known"] is None
    assert "clock_unknown" in _codes(result)


def test_measurement_limitations_block_negative_verdict():
    result = _review(_window(measurement_limitations=["frequency register unavailable"]))

    assert result["status"] == "inconclusive"
    assert result["measurement_limitations"] == ["frequency register unavailable"]
    assert "measurement_limitations_reported" in _codes(result)


def test_missing_low_level_signal_blocks_negative_verdict():
    result = _review(_window(measurement_limitations=None))

    assert result["status"] == "inconclusive"
    assert "measurement_limitations_missing" in _codes(result)


def test_positive_signal_survives_partial_coverage_with_limitation():
    result = _review(_window(throttling_detected=True, coverage_ratio=0.50))

    assert result["status"] == "throttled_observed"
    assert result["conclusive"] is True
    assert result["confidence"] == "limited"
    assert result["throttling"]["proven"] is True
    assert "coverage_incomplete" in _codes(result)


def test_device_level_positive_signal_is_used():
    result = _review(_window(
        devices=[_device(0, throttling_detected=True)],
        throttling_detected=False,
    ))

    assert result["status"] == "throttled_observed"
    assert result["throttling"]["source"] == "device_signal"


def test_conflicting_device_signals_are_reported_but_positive_is_retained():
    result = _review(_window(
        devices=[
            _device(0, throttling_detected=True),
            _device(1, throttling_detected=False),
        ],
        throttling_detected=False,
    ), expected_devices=[0, 1])

    assert result["status"] == "throttled_observed"
    assert "conflicting_throttling_signals" in _codes(result)


def test_missing_requested_device_statistics_are_null_not_zero():
    result = _review(_window(devices=[_device(0)]), expected_devices=[0, 1])

    assert result["status"] == "inconclusive"
    assert "device_statistics_missing" in _codes(result)
    devices = {item["device_id"]: item for item in result["devices"]}
    assert devices[1]["status"] == "missing"
    assert devices[1]["temp_max_c"] is None
    assert devices[1]["freq_mhz_min"] is None


def test_single_device_aggregate_without_clock_or_power_is_inconclusive():
    result = review_telemetry_window(
        {
            "host": "a3-165",
            "devices": [0],
            "start_ts": 1_000,
            "end_ts": 2_000,
            "coverage_ratio": 1.0,
            "throttling_detected": False,
            "measurement_limitations": [],
        },
        expected_host="a3-165",
    )

    assert result["status"] == "inconclusive"
    assert "clock_unknown" in _codes(result)
    assert "power_signal_missing" in _codes(result)


def test_host_and_window_mismatch_block_negative_verdict():
    host = _review(_window(), expected_host="a3-156")
    assert host["status"] == "inconclusive"
    assert "window_host_mismatch" in _codes(host)

    bounds = _review(_window(), expected_end_ts=2_500)
    assert bounds["status"] == "inconclusive"
    assert "window_end_ts_mismatch" in _codes(bounds)


def test_missing_window_echo_is_a_nonblocking_limitation():
    window = _window()
    del window["start_ts"]
    del window["end_ts"]
    result = _review(window)

    assert result["status"] == "no_throttling_observed"
    assert "window_bounds_unverified" in _codes(result)


def test_non_mapping_window_returns_no_data():
    result = review_telemetry_window(None)
    assert result["status"] == "no_data"
    assert result["conclusive"] is False


class _FakeResponse:
    def __init__(self, body, status=200):
        self.body = body if isinstance(body, bytes) else body.encode("utf-8")
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self, size):
        return self.body[:size]


def test_client_issues_one_bounded_read_only_get():
    calls = []
    payload = _window()

    def opener(request, timeout):
        calls.append((request, timeout))
        return _FakeResponse(json.dumps(payload))

    result = fetch_telemetry_window(
        "http://127.0.0.1:8899/base",
        "a3-165",
        [1, 0, 1],
        1_000,
        2_000,
        timeout=2.5,
        opener=opener,
    )

    assert result == payload
    assert len(calls) == 1
    request, timeout = calls[0]
    assert timeout == 2.5
    assert request.get_method() == "GET"
    parsed = urllib.parse.urlsplit(request.full_url)
    assert parsed.path == "/base/api/agent/telemetry-window"
    query = urllib.parse.parse_qs(parsed.query)
    assert query["host"] == ["a3-165"]
    assert query["devices"] == ["0,1"]
    assert query["start_ts"] == ["1000"]
    assert query["end_ts"] == ["2000"]


def test_client_does_not_retry():
    calls = []

    def opener(request, timeout):
        calls.append(request)
        raise OSError("connection refused")

    client = TelemetryClient("http://127.0.0.1:8899", timeout=1.0, opener=opener)
    with pytest.raises(TelemetryFetchError):
        client.fetch_window("a3-165", [0], 1_000, 2_000)
    assert len(calls) == 1


def test_client_rejects_unbounded_timeout():
    with pytest.raises(ValueError):
        TelemetryClient("http://127.0.0.1:8899", timeout=31.0)
    with pytest.raises(ValueError):
        TelemetryClient("http://127.0.0.1:8899", timeout=0)


def test_client_bounds_response_size():
    client = TelemetryClient(
        "http://127.0.0.1:8899",
        timeout=1.0,
        max_response_bytes=8,
        opener=lambda request, timeout: _FakeResponse(b"{" + b"x" * 20 + b"}"),
    )
    with pytest.raises(TelemetryFetchError):
        client.fetch_window("a3-165", [0], 1_000, 2_000)


def test_client_rejects_http_error_and_invalid_json():
    failing = TelemetryClient(
        "http://127.0.0.1:8899",
        timeout=1.0,
        opener=lambda request, timeout: _FakeResponse(b"{}", status=503),
    )
    with pytest.raises(TelemetryFetchError):
        failing.fetch_window("a3-165", [0], 1_000, 2_000)

    invalid = TelemetryClient(
        "http://127.0.0.1:8899",
        timeout=1.0,
        opener=lambda request, timeout: _FakeResponse(b"not json"),
    )
    with pytest.raises(TelemetryFetchError):
        invalid.fetch_window("a3-165", [0], 1_000, 2_000)


def test_fetch_and_review_uses_explicit_window_identity():
    payload = _window()

    def opener(request, timeout):
        return _FakeResponse(json.dumps(payload))

    client = TelemetryClient("http://127.0.0.1:8899", timeout=1.0, opener=opener)
    result = client.fetch_and_review("a3-165", [0], 1_000, 2_000)

    assert result["status"] == "no_throttling_observed"
    assert result["window"]["host"] == "a3-165"
    assert result["window"]["devices"] == [0]


def test_invalid_throttle_type_is_inconclusive():
    result = _review(_window(throttling_detected="false"))

    assert result["status"] == "inconclusive"
    assert result["throttling"]["detected"] is None
    assert "throttling_signal_invalid" in _codes(result)


def test_invalid_measurement_limitations_field_is_inconclusive():
    result = _review(_window(measurement_limitations="none"))

    assert result["status"] == "inconclusive"
    assert "measurement_limitations_invalid" in _codes(result)


def test_client_rejects_base_url_query():
    with pytest.raises(ValueError):
        TelemetryClient("http://127.0.0.1:8899/base?token=secret", timeout=1.0)


def test_missing_power_signal_blocks_negative_verdict():
    device = _device(0)
    device.pop("power_max_w")
    result = _review(_window(devices=[device]))

    assert result["status"] == "inconclusive"
    assert "power_signal_missing" in _codes(result)


def test_real_npu_top_payload_shape_is_consumed():
    payload = {
        "source": "sqlite",
        "observation": {"notice": "historical raw telemetry"},
        "server": {"id": "a3-165", "name": "a3-165", "host": "192.0.2.10", "port": 22},
        "window": {
            "start_ts": 1_000,
            "end_ts": 2_000,
            "requested_devices": [0],
            "raw_sample_count": 2,
            "rollup_sample_count": 0,
        },
        "coverage_ratio": 1.0,
        "throttling_detected": False,
        "resource_interference_detected": None,
        "measurement_limitations": [
            "Resource interference was not measured explicitly; "
            "resource_interference_detected is null, not false."
        ],
        "devices": [
            {
                "npu_id": 0,
                "sample_count": 2,
                "expected_sample_count": 2,
                "coverage_ratio": 1.0,
                "throttling_detected": False,
                "aicore_p50": 0.7,
                "aicore_p95": 0.9,
                "hbm_peak_mb": 1024.0,
                "temp_max_c": 70.0,
                "power_max_w": 300.0,
                "freq_mhz_min": 1800.0,
                "measurement_limitations": [],
            }
        ],
    }

    result = review_telemetry_window(payload)

    assert result["status"] == "no_throttling_observed"
    assert result["window"]["host"] == "192.0.2.10"
    assert result["window"]["start_ts"] == 1_000
    assert result["window"]["end_ts"] == 2_000
    assert result["window"]["devices"] == [0]
    assert result["devices"][0]["device_id"] == 0
    assert result["devices"][0]["coverage_ratio"] == 1.0
    assert "measurement_limitations_reported" in _codes(result)
    assert "measurement_limitation_blocks_throttle_verdict" not in _codes(result)


C4_SOURCE_TELEMETRY = Path(__file__).resolve().parent / "fixtures" / "profiler_core" / "c2_a2" / "source.telemetry.json"


def test_real_c4_c2_aggregate_single_card_telemetry_is_consumed():
    payload = json.loads(C4_SOURCE_TELEMETRY.read_text(encoding="utf-8"))

    result = review_telemetry_window(payload)

    assert result["status"] == "no_throttling_observed"
    assert result["coverage"]["complete"] is True
    assert result["clock"]["known"] is True
    assert result["devices"][0]["device_id"] == 0
    assert result["devices"][0]["freq_mhz_min"] == payload["freq_mhz_min"]
    assert result["devices"][0]["power_max_w"] == payload["power_max_w"]
    assert result["temperature"]["used_as_throttling_proof"] is False
