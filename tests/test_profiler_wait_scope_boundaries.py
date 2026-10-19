"""Pure interval regressions for observation and physical-device boundaries."""
from dataclasses import replace

import pytest

from mindie_profiler.anomaly_discovery import KernelEvent, assess_wait_anchor


def _pair():
    candidate = KernelEvent(
        name="anchor", step=1, start_us=1000, duration_us=1, wait_us=1000,
        task_type="AI_CORE", stream_id=7, rank=0, event_id="anchor",
        observation_id="first", window_start_ns=0, window_end_ns=2_000_000,
        device_id="0",
    )
    work = replace(candidate, name="work", start_us=0, duration_us=1000,
                   wait_us=0, event_id="work")
    return candidate, work


@pytest.mark.parametrize("relation", [False, True])
def test_other_device_cannot_supply_wait_overlap(relation):
    candidate, work = _pair()
    if relation:
        candidate = replace(candidate, related_event_ids=("work",))
    work = replace(work, device_id="1")
    report = assess_wait_anchor([candidate, work])
    assert report.findings[0]["confirmed_event_ids"] == []
    assert report.findings[0]["overlap_pairs"] == []


def test_related_event_cannot_cross_observation_or_window():
    candidate, work = _pair()
    candidate = replace(candidate, related_event_ids=("work",))
    work = replace(work, observation_id="second", window_start_ns=2_000_000,
                   window_end_ns=4_000_000)
    report = assess_wait_anchor([candidate, work])
    assert report.findings[0]["confirmed_event_ids"] == []
    assert report.findings[0]["overlap_pairs"] == []


def test_repeated_event_in_second_window_keeps_first_related_overlap():
    candidate, work = _pair()
    candidate = replace(candidate, start_us=2000, window_end_ns=3_000_000,
                        related_event_ids=("work",))
    work = replace(work, start_us=1000, stream_id=8, window_end_ns=3_000_000)
    duplicate = replace(work, observation_id="second", window_start_ns=500_000,
                        window_end_ns=3_500_000, start_us=500)
    first = assess_wait_anchor([candidate, work]).findings[0]
    assert first["confirmed_event_ids"] == ["anchor"]
    repeated = assess_wait_anchor([candidate, work, duplicate]).findings[0]
    assert repeated["confirmed_event_ids"] == first["confirmed_event_ids"]
    assert repeated["overlap_pairs"] == first["overlap_pairs"]
