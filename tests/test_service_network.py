import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from mindie_diagnostics import service_environment as network


@pytest.fixture(autouse=True)
def clean_network(monkeypatch):
    for key in network.ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_service_network_survives_process_boundary_without_copying_api_tokens(tmp_path, monkeypatch):
    selected = {
        "HTTPS_PROXY": "http://fixture:encoded%40password@proxy.invalid:8080",
        "NO_PROXY": "localhost,127.0.0.1",
        "SSL_CERT_FILE": str(tmp_path / 'CA files' / 'company.pem'),
        "GH_CONFIG_DIR": str(tmp_path / "GitHub profile"),
    }
    for key, value in selected.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("GH_TOKEN", "must-not-persist")
    path, changed = network.prepare(tmp_path)
    assert changed
    assert "must-not-persist" not in path.read_text()
    env = {k: v for k, v in os.environ.items() if k not in network.ENV_KEYS}
    env["PYTHONPATH"] = str(Path(network.__file__).resolve().parents[1])
    result = subprocess.run(
        [sys.executable, "-c",
         "import json,os,sys,urllib.request;"
         "from mindie_diagnostics.service_environment import read;"
         "os.environ.update(read(sys.argv[1]));"
         "print(json.dumps({'proxy':urllib.request.getproxies()['https'],"
         "'ca':os.environ['SSL_CERT_FILE'],'gh':os.environ['GH_CONFIG_DIR'],"
         "'token':os.environ.get('GH_TOKEN')}))", str(path)],
        env=env, capture_output=True, text=True, timeout=5, check=True,
    )
    assert json.loads(result.stdout) == {
        "proxy": selected["HTTPS_PROXY"], "ca": selected["SSL_CERT_FILE"],
        "gh": selected["GH_CONFIG_DIR"], "token": None,
    }
    assert network.prepare(tmp_path) == (path, False)
    monkeypatch.setenv("HTTPS_PROXY", "http://new-proxy.invalid:8080")
    assert network.prepare(tmp_path) == (path, True)
    assert network.read(path)["HTTPS_PROXY"] == "http://new-proxy.invalid:8080"


def test_explicit_environment_remains_authoritative(tmp_path, monkeypatch):
    explicit = tmp_path / "operator.env"
    explicit.write_text("HTTPS_PROXY=http://selected.invalid\nGH_TOKEN=fixture\n")
    monkeypatch.setenv("HTTPS_PROXY", "http://ambient.invalid")
    assert network.prepare(tmp_path, explicit) == (explicit, False)
    assert network.read(explicit)["HTTPS_PROXY"] == "http://selected.invalid"
    assert not (tmp_path / "mindie-network.env").exists()
