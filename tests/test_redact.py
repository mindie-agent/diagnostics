import json

import pytest

from mindie_diagnostics.redact import Allowlist, REDACTION_PROFILE, RULE_IDS, redact_text, scan_text, scan_tree


@pytest.mark.parametrize(("value", "rule"), [
    ("10." + "17.23.41", "ipv4-address"),
    ("node-" + "01", "hostname-numbered"),
    ("worker-pool-size-32", None),
    ("host: " + "npu-rack2", "hostname-assignment"),
    ("C:\\Users\\" + "exampleuser" + "\\code", "user-path-windows"),
    ("/home/" + "exampleuser" + "/project", "user-path"),
    ("ghp_" + "A" * 36, "credential-known-format"),
    ("Authorization: Bearer " + "abcdEFGH1234", "credential-bearer"),
    ("token " + "Zq9" * 12, "credential-high-entropy"),
    ("container: " + "mindie-a3-01", "container-name-assignment"),
    ("sha256:" + "ab" * 32, None),
    ("torch_npu 2.10.0; NPU index 4; 8192x8192 dense matmul", None),
    ("https://example.com and /home/<user>/work", None),
])
def test_migrated_r2_golden(value, rule):
    found = scan_text(value)
    if rule is None:
        assert found == []
        assert redact_text(value) == value
    else:
        assert rule in {item.rule for item in found}
        sanitized = redact_text(value)
        assert all(item.value not in sanitized for item in found)
        assert scan_text(sanitized) == []


def test_rules_unique_and_allowlist_core_compatible(tmp_path):
    assert REDACTION_PROFILE == "r2" and len(set(RULE_IDS)) == len(RULE_IDS)
    literal = "node-" + "01"
    path = tmp_path / "allow.json"
    path.write_text(json.dumps([literal]))
    assert not scan_text(literal, Allowlist.from_args(files=[str(path)]))
    assert not scan_text(literal, Allowlist(terms=[literal]))
    assert scan_tree({"location": [literal]})[0].path == "location[0]"
    with pytest.raises(ValueError):
        Allowlist(patterns=["("])
