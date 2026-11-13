"""[Role] Carry caller network settings into native services and scheduled updates.
[Invariants] Explicit environment files take precedence; generated files contain
only transport settings and gh configuration paths, never API tokens.
[Timeouts & Locks] Local bounded file I/O under the caller's service lock.
[Side-Effect Policy] Atomically replaces the service's private environment file;
does not change user/global configuration or publication authorization.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import tempfile

NETWORK_ENV = (
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
    "http_proxy", "https_proxy", "all_proxy", "no_proxy",
    "SSL_CERT_FILE", "SSL_CERT_DIR", "SSL_CLIENT_CERT",
    "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE", "NODE_EXTRA_CA_CERTS",
    "CODEX_CA_CERTIFICATE", "GIT_SSL_CAINFO", "GIT_SSL_CAPATH",
    "UV_SYSTEM_CERTS", "UV_NATIVE_TLS", "UV_INDEX", "UV_DEFAULT_INDEX",
    "UV_INDEX_URL", "UV_EXTRA_INDEX_URL", "UV_CONFIG_FILE",
    "UV_PYTHON_INSTALL_MIRROR", "UV_PYPY_INSTALL_MIRROR",
    "PIP_INDEX_URL", "PIP_EXTRA_INDEX_URL", "PIP_CERT", "PIP_CONFIG_FILE",
    "GH_CONFIG_DIR", "GH_HOST", "MINDIE_GIT_BIN", "GCM_INTERACTIVE",
)
ENV_KEYS = {*NETWORK_ENV, "GH_TOKEN", "GITHUB_TOKEN"}
LIMIT = 32768
_GIT_SLOT = re.compile(r"GIT_CONFIG_(?:COUNT|(?:KEY|VALUE)_[0-9]+)\Z")
_GIT_TRANSPORT = re.compile(
    r"http(?:\.[^\r\n]+)?\.(?:proxy|proxyauthmethod|sslbackend|sslcainfo|sslcapath|"
    r"sslverify|sslversion|schannelcheckrevoke|schannelusesslcainfo)\Z", re.IGNORECASE)


def git_transport_environment(environment):
    """Project transport options only; credential helpers and headers stay out."""
    if "GIT_CONFIG_COUNT" not in environment:
        return {}
    try:
        count = int(environment["GIT_CONFIG_COUNT"])
    except (TypeError, ValueError):
        raise ValueError("invalid_service_git_configuration") from None
    if not 0 <= count <= 256:
        raise ValueError("invalid_service_git_configuration")
    values, selected = {}, 0
    for index in range(count):
        try:
            key = environment[f"GIT_CONFIG_KEY_{index}"]
            value = environment[f"GIT_CONFIG_VALUE_{index}"]
        except KeyError:
            raise ValueError("invalid_service_git_configuration") from None
        if _GIT_TRANSPORT.fullmatch(key):
            values[f"GIT_CONFIG_KEY_{selected}"] = key
            values[f"GIT_CONFIG_VALUE_{selected}"] = value
            selected += 1
    values["GIT_CONFIG_COUNT"] = str(selected)
    return values


def validate_git_environment(values):
    supplied = {key: value for key, value in values.items() if _GIT_SLOT.fullmatch(key)}
    if supplied != git_transport_environment(supplied):
        raise ValueError("invalid_service_git_configuration")


def prepare(directory, explicit=None):
    """Return the selected file and whether its effective settings changed."""
    path = Path(directory) / "mindie-network.env"
    if explicit is not None and Path(explicit).absolute() != path.absolute():
        return Path(explicit), False
    values = {key: os.environ[key] for key in NETWORK_ENV if key in os.environ}
    values.update(git_transport_environment(os.environ))
    if any(any(ord(char) < 32 for char in value) for value in values.values()):
        raise ValueError("invalid_service_network_environment")
    content = "".join(f"{key}={json.dumps(value, ensure_ascii=False)}\n"
                      for key, value in sorted(values.items()))
    encoded = content.encode("utf-8")
    if len(encoded) > LIMIT:
        raise ValueError("service_network_environment_too_large")
    if not values and not path.exists():
        return None, False
    if path.exists() and path.read_bytes() == encoded:
        return path, False
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary = tempfile.mkstemp(prefix=".network-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path, True


def read(path):
    """Read the same literal values accepted by systemd EnvironmentFile."""
    with Path(path).open(encoding="utf-8") as stream:
        content = stream.read(LIMIT + 1)
    if len(content) > LIMIT:
        raise ValueError("invalid_environment_file")
    values = {}
    for line in content.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        key, equal, value = line.partition("=")
        if not equal or key not in ENV_KEYS and not _GIT_SLOT.fullmatch(key):
            raise ValueError("invalid_environment_file")
        if value.startswith('"'):
            value = json.loads(value)
        if not isinstance(value, str) or any(ord(char) < 32 for char in value):
            raise ValueError("invalid_environment_file")
        values[key] = value
    validate_git_environment(values)
    return values


def apply(values):
    """Apply the selected service network, including explicitly absent settings."""
    for key in list(os.environ):
        if key in ENV_KEYS or _GIT_SLOT.fullmatch(key):
            os.environ.pop(key, None)
    os.environ.update(values)
