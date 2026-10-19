"""A steady writer checks one marker, while changes still apply immediately."""
import json
from pathlib import Path

from mindie_diagnostics import configure
from mindie_diagnostics import reader_registry as registry


def test_marker_changes_are_immediate_without_revalidating_every_ancestor(tmp_path, monkeypatch):
    calls = []
    original = registry._real_path
    def counted(path):
        calls.append(path)
        return original(path)
    monkeypatch.setattr(registry, '_real_path', counted)
    reader = registry.PressureReader(tmp_path)
    for _ in range(100):
        assert not reader()
    assert len(calls) == 1
    marker = tmp_path / 'retention-state.json'
    marker.write_text(json.dumps({'schema': 1, 'blocked': True, 'limited': True}))
    marker.chmod(0o600)
    assert reader()
    marker.write_text('invalid')
    assert not reader()
    marker.unlink()
    assert not reader()
    assert len(calls) == 4
    assert not reader(force=True)
    assert len(calls) == 5


def test_existing_writer_stops_growth_on_new_pressure_and_resumes(tmp_path):
    recorder = configure('pressure-cost', root=tmp_path)
    try:
        # Cold registry/segment setup may consume the nonblocking writer budget.
        # Warm it once before measuring an existing writer's pressure response.
        recorder.event('INFO', 'warmup')
        warmup_drops = recorder.dropped_records
        if warmup_drops:
            assert recorder.logging_failed
        recorder.event('INFO', 'first')
        assert recorder.dropped_records == warmup_drops, (
            f"existing writer did not admit the first measured event: "
            f"logging_failed={recorder.logging_failed}, "
            f"dropped_records={recorder.dropped_records}, warmup_drops={warmup_drops}"
        )
        log = Path(recorder.record_ref)
        before = log.stat().st_size
        marker = tmp_path / 'retention-state.json'
        marker.write_text(json.dumps({'schema': 1, 'blocked': True, 'limited': True}))
        marker.chmod(0o600)
        recorder.event('INFO', 'blocked')
        assert log.stat().st_size == before
        marker.unlink()
        recorder.event('INFO', 'resumed')
        assert log.stat().st_size > before
    finally:
        recorder.close()
