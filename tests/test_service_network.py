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
    for key in list(os.environ):
        if key.startswith("GIT_CONFIG_"):
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
         "from mindie_diagnostics.service_environment import read,apply;"
         "apply(read(sys.argv[1]));"
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


def test_saved_network_removes_stale_proxy_and_credentials_in_real_child(tmp_path, monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://previous-network.invalid:8080")
    path, _ = network.prepare(tmp_path)
    monkeypatch.delenv("HTTPS_PROXY")
    path, changed = network.prepare(tmp_path)
    assert changed and network.read(path) == {}
    env = dict(os.environ, HTTPS_PROXY="http://stale-scheduled-task.invalid:8080",
               GH_TOKEN="stale-task-token",
               PYTHONPATH=str(Path(network.__file__).resolve().parents[1]))
    result = subprocess.run(
        [sys.executable, "-c",
         "import json,os,sys;"
         "from mindie_diagnostics.service_environment import apply,read;"
         "apply(read(sys.argv[1]));"
         "print(json.dumps({k:os.environ.get(k) for k in ('HTTPS_PROXY','GH_TOKEN')}))",
         str(path)],
        env=env, text=True, capture_output=True, timeout=5, check=True,
        **({"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}),
    )
    assert json.loads(result.stdout) == {"HTTPS_PROXY": None, "GH_TOKEN": None}


def test_git_network_configuration_crosses_process_without_helpers_or_auth_headers(tmp_path, monkeypatch):
    import shutil

    selected = [("http.proxy", "http://fixture:encoded%40password@proxy.invalid:8080"),
                ("http.sslCAInfo", str(tmp_path / "Company CA.pem")),
                ("http.schannelCheckRevoke", "false"),
                ("credential.helper", "!must-not-execute"),
                ("http.https://github.com/.extraheader", "Authorization: must-not-persist")]
    monkeypatch.setenv("GIT_CONFIG_COUNT", str(len(selected)))
    for index, (key, value) in enumerate(selected):
        monkeypatch.setenv(f"GIT_CONFIG_KEY_{index}", key)
        monkeypatch.setenv(f"GIT_CONFIG_VALUE_{index}", value)
    path, _ = network.prepare(tmp_path)
    assert "must-not-" not in path.read_text()
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_CONFIG_")}
    env["PYTHONPATH"] = str(Path(network.__file__).resolve().parents[1])
    result = subprocess.run(
        [sys.executable, "-c",
         "import json,os,subprocess,sys;"
         "from mindie_diagnostics.service_environment import apply,read;"
         "apply(read(sys.argv[1]));"
         "print(json.dumps(subprocess.run([sys.argv[2],'config','--get','http.proxy'],"
         "env=os.environ,text=True,capture_output=True,timeout=3).stdout.strip()))",
         str(path), shutil.which("git")],
        env=env, text=True, capture_output=True, timeout=5, check=True,
        **({"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}),
    )
    assert json.loads(result.stdout) == selected[0][1]
