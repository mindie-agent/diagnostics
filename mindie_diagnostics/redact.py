"""Pure scanner migrated from mindie-knowledge r2 without changing rule behavior.

File/CLI parsing remains with callers. This module has no knowledge dependency.
"""
from __future__ import annotations
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

class ToolError(ValueError):
    pass

def json_pointer(path):
    result = ""
    for part in path:
        result += f"[{part}]" if isinstance(part, int) else (("." if result else "") + str(part))
    return result or "<document>"

def load_document(path):
    """Standalone allow files use JSON; knowledge retains its YAML CLI adapter."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ToolError("cannot read JSON allow file") from exc

#: Version of the ruleset declared in this file. Bump when a rule tightens.
#:
#: r2: adds ``internal-machine-identifier`` (a machine word followed by a
#: slot number, "remote 131" / "remote_131", and run identifiers that embed a
#: machine slot beside a device index, "microbenchmark_131_npu4_...") and adds
#: ``remote`` to the numbered-host prefixes. Entries cleared under r1 must be
#: re-scanned (sync/rescan.py --profile r2).
#: r2 (correction): ``hostname-numbered`` no longer allows unbounded words
#: between the prefix and the digits, so prose such as ``worker-pool-size-32``
#: and candidate ids such as ``<owner>-<slug>-<hex>`` are not reported. The
#: trailing boundary is ``(?![a-z0-9])`` so a slot token may continue with
#: ``_`` or ``-`` (``remote_000_cann_...`` still hits ``remote_000``). This
#: is a relaxation, not a tightening; entries already cleared under r2 do
#: not need a re-scan.
REDACTION_PROFILE = "r2"


# --------------------------------------------------------------------------- #
# Rules
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Rule:
    id: str
    description: str
    hint: str
    pattern: "re.Pattern[str] | None" = None
    finder: "Callable[[str], Iterable[tuple[int, int]]] | None" = None

    def spans(self, text: str) -> Iterator[tuple[int, int]]:
        if self.pattern is not None:
            for m in self.pattern.finditer(text):
                # Prefer the first capture group when a rule anchors on
                # context (e.g. "--name X" reports only X).
                if m.lastindex:
                    yield m.span(m.lastindex)
                else:
                    yield m.span()
        if self.finder is not None:
            yield from self.finder(text)


_HEX = re.compile(r"^[0-9a-f]+$", re.IGNORECASE)
_UUID = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE
)
_TOKEN_CANDIDATE = re.compile(r"[A-Za-z0-9+/_=-]{32,}")


def _high_entropy_tokens(text: str) -> Iterator[tuple[int, int]]:
    """Long mixed-case alphanumeric blobs that are not hashes or uuids.

    Hex digests (commit SHAs and content digests) and UUIDs are structurally
    identifiable and are not credentials, so they are excluded; everything
    else that is >= 32 chars and mixes upper, lower and digits is reported.
    """
    for m in _TOKEN_CANDIDATE.finditer(text):
        tok = m.group()
        if _HEX.match(tok) or _UUID.match(tok):
            continue
        has_upper = any(c.isupper() for c in tok)
        has_lower = any(c.islower() for c in tok)
        has_digit = any(c.isdigit() for c in tok)
        if has_upper and has_lower and has_digit:
            yield m.span()


_OCTET = r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"
_PLACEHOLDER_START = r"(?![<$\{\*%])"  # <host>, $USER, {name}, *, %USERNAME%

_HOST_PREFIXES = (
    "node|host|server|srv|worker|master|machine|ecs|vm|pod|box|k8s|rack|blade|"
    "compute|bastion|jump|login|cluster|remote"
)
# Machine words that, followed by a bare number *with a space*, name a machine
# in prose ("on remote 131", "machine 7"). Deliberately narrower than
# _HOST_PREFIXES: "node 4" and "host 2" are also how people write rank and
# node *indices*, which are load-bearing method detail and must survive.
_MACHINE_WORDS = "remote|machine|box|server|bastion|workstation"
_INTERNAL_TLDS = "local|localdomain|internal|intranet|lan|corp|home|priv|private|localnet"

RULES: tuple[Rule, ...] = (
    # --- credentials ------------------------------------------------------
    Rule(
        id="credential-known-format",
        description="token in a well-known provider format",
        hint="never submit credentials, not even expired or partial ones",
        pattern=re.compile(
            r"(?<![A-Za-z0-9_])(?:"
            r"gh[pousr]_[A-Za-z0-9]{20,}"
            r"|github_pat_[A-Za-z0-9_]{20,}"
            r"|glpat-[A-Za-z0-9_-]{20,}"
            r"|AKIA[0-9A-Z]{16}"
            r"|sk-[A-Za-z0-9_-]{20,}"
            r"|xox[abprs]-[A-Za-z0-9-]{10,}"
            r"|hf_[A-Za-z0-9]{30,}"
            r"|AIza[0-9A-Za-z_-]{35}"
            r"|eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]*"
            r"|-----BEGIN [A-Z ]*PRIVATE KEY-----"
            r")"
        ),
    ),
    Rule(
        id="credential-url-userinfo",
        description="user:password embedded in a URL",
        hint="strip the userinfo part of the URL",
        pattern=re.compile(r"[a-z][a-z0-9+.-]*://[^/\s:@]+:[^/\s@]+@", re.IGNORECASE),
    ),
    Rule(
        id="credential-bearer",
        description="bearer token",
        hint="never submit credentials",
        pattern=re.compile(r"\bbearer\s+([A-Za-z0-9._~+/=-]{8,})", re.IGNORECASE),
    ),
    Rule(
        id="credential-assignment",
        description="credential-shaped key/value assignment",
        hint="replace the value with a placeholder such as <token>",
        pattern=re.compile(
            r"\b(?:api[_-]?key|apikey|secret(?:[_-]?key)?|access[_-]?key|"
            r"private[_-]?key|token|passw(?:or)?d|passwd|pwd|auth(?:orization)?|"
            r"credentials?)\b\s*[:=]\s*[\"']?" + _PLACEHOLDER_START + r"([^\s,;)\"']{6,})",
            re.IGNORECASE,
        ),
    ),
    # --- network identifiers ------------------------------------------------
    Rule(
        id="mac-address",
        description="MAC address",
        hint="hardware addresses identify a specific machine; remove them",
        pattern=re.compile(
            r"(?<![0-9a-f:.-])(?:[0-9a-f]{2}[:-]){5}[0-9a-f]{2}(?![0-9a-f:.-])",
            re.IGNORECASE,
        ),
    ),
    Rule(
        id="ipv6-address",
        description="IPv6 address",
        hint="describe the network role instead of the address",
        pattern=re.compile(
            r"(?<![0-9a-z:])(?:"
            r"(?:[0-9a-f]{1,4}:){7}[0-9a-f]{1,4}"
            r"|(?:[0-9a-f]{1,4}:){1,7}:"
            r"|(?:[0-9a-f]{1,4}:){1,6}:[0-9a-f]{1,4}"
            r"|(?:[0-9a-f]{1,4}:){1,5}(?::[0-9a-f]{1,4}){1,2}"
            r"|(?:[0-9a-f]{1,4}:){1,4}(?::[0-9a-f]{1,4}){1,3}"
            r"|(?:[0-9a-f]{1,4}:){1,3}(?::[0-9a-f]{1,4}){1,4}"
            r"|(?:[0-9a-f]{1,4}:){1,2}(?::[0-9a-f]{1,4}){1,5}"
            r"|[0-9a-f]{1,4}:(?::[0-9a-f]{1,4}){1,6}"
            r"|:(?:(?::[0-9a-f]{1,4}){1,7}|:)"
            r"|::(?:ffff(?::0{1,4})?:)?(?:" + _OCTET + r"\.){3}" + _OCTET +
            r")(?:%[0-9a-z]+)?(?![0-9a-z:])",
            re.IGNORECASE,
        ),
    ),
    Rule(
        id="ipv4-address",
        description="IPv4 address (optionally a range or CIDR)",
        hint="describe the network role instead of the address; allowlist a "
        "benign 4-part version string explicitly if this is one",
        pattern=re.compile(
            r"(?<![\w.])(?:" + _OCTET + r"\.){3}" + _OCTET +
            r"(?:\s*-\s*(?:(?:" + _OCTET + r"\.){0,3}" + _OCTET + r"))?"
            r"(?:/\d{1,2})?(?![\w])"
        ),
    ),
    # --- people -------------------------------------------------------------
    Rule(
        id="email-address",
        description="e-mail address",
        hint="use a public handle when attribution is needed; omit private contact details",
        pattern=re.compile(r"(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)+(?![\w.-])"),
    ),
    Rule(
        id="username-at-host",
        description="user@host login target",
        hint="remove login targets; they name both a user and a machine",
        pattern=re.compile(
            r"(?<![\w.+-])[a-z_][\w-]{0,31}@[a-z0-9][\w-]*(?:\.[\w-]+)*(?![\w.-])",
            re.IGNORECASE,
        ),
    ),
    Rule(
        id="user-path",
        description="absolute path revealing a user account",
        hint="use a placeholder such as /home/<user>/... or a relative path",
        pattern=re.compile(
            r"(?<![\w/])(?:/home|/Users|/export/home|/data/home|/root)/"
            + _PLACEHOLDER_START + r"[^\s\"'`,;:()\[\]]+"
        ),
    ),
    Rule(
        id="user-path-windows",
        description="Windows profile path revealing a user account",
        hint="use a placeholder such as C:\\Users\\<user>\\...",
        pattern=re.compile(
            r"\b[A-Za-z]:\\Users\\" + _PLACEHOLDER_START + r"([^\\\s\"']+)"
        ),
    ),
    Rule(
        id="user-tilde-path",
        description="~user home directory reference",
        hint="use ~ or $HOME without a user name",
        pattern=re.compile(r"(?<![\w/])~(?![/\s<$\{])([a-z_][\w.-]*)", re.IGNORECASE),
    ),
    Rule(
        id="username-assignment",
        description="user name in a key/value assignment",
        hint="remove the account name or replace it with <user>",
        pattern=re.compile(
            r"\b(?:user(?:name)?|login|account|owner|uid)\s*[:=]\s*[\"']?"
            + _PLACEHOLDER_START + r"([a-z_][\w.-]{1,31})",
            re.IGNORECASE,
        ),
    ),
    Rule(
        id="username-flag",
        description="user name passed via --user",
        hint="remove the account name or replace it with <user>",
        pattern=re.compile(
            r"(?<![\w-])--(?:user|login)[= ]" + _PLACEHOLDER_START + r"([a-z_][\w-]{1,31})",
            re.IGNORECASE,
        ),
    ),
    # --- machines -----------------------------------------------------------
    Rule(
        id="container-command",
        description="container name in a docker/podman command",
        hint="replace the container name with <container>",
        pattern=re.compile(
            r"\b(?:docker|podman|nerdctl|ctr)\s+(?:container\s+)?"
            r"(?:exec|attach|logs|start|stop|restart|rm|kill|inspect|cp|top|"
            r"stats|port|rename|update|pause|unpause|commit)\s+"
            r"(?:-{1,2}[\w-]+(?:=\S+)?\s+)*" + _PLACEHOLDER_START + r"([A-Za-z0-9][\w.-]+)",
            re.IGNORECASE,
        ),
    ),
    Rule(
        id="container-name-flag",
        description="container or host name passed via --name/--hostname",
        hint="replace the name with a placeholder",
        pattern=re.compile(
            r"(?<![\w-])--(?:name|hostname|container)[= ]"
            + _PLACEHOLDER_START + r"([A-Za-z0-9][\w.-]+)",
            re.IGNORECASE,
        ),
    ),
    Rule(
        id="container-name-assignment",
        description="container name in a key/value assignment",
        hint="replace the container name with <container>",
        pattern=re.compile(
            r"\bcontainer(?:[_ -]?(?:name|id))?\s*[:=]\s*[\"']?"
            + _PLACEHOLDER_START + r"(?=[\w.-]*[\d_-])([A-Za-z0-9][\w.-]{2,})",
            re.IGNORECASE,
        ),
    ),
    Rule(
        id="hostname-internal-domain",
        description="host name under an internal-looking domain",
        hint="describe the machine role instead of naming it",
        pattern=re.compile(
            r"(?<![\w.-])(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+(?:"
            + _INTERNAL_TLDS + r")(?![\w-])",
            re.IGNORECASE,
        ),
    ),
    Rule(
        id="hostname-numbered",
        description="numbered host name (node-01, worker3, ...)",
        hint="describe the machine role instead of naming it",
        pattern=re.compile(
            r"(?<![\w.-])(?:" + _HOST_PREFIXES + r")[-_]?\d{1,4}"
            r"(?:[a-z][a-z0-9]*)?(?![a-z0-9])",
            re.IGNORECASE,
        ),
    ),
    Rule(
        id="internal-machine-identifier",
        description="internal machine slot identifier (remote 131, remote_131, "
        "<run>_131_npu4_...)",
        hint="drop the machine identity and keep the method: device model, device "
        "index, software versions, workload shape",
        pattern=re.compile(
            # "remote 131" / "machine 7": a machine word, one space, a slot number.
            r"(?:(?<![\w.-])(?:" + _MACHINE_WORDS + r") \d{1,4}(?![\w])"
            # "microbenchmark_131_npu4_2026-06-03": a run identifier in which a
            # bare number sits directly beside a device index. The number is
            # the machine slot; "npu4" is the device and is *not* reported.
            r"|(?<=_)\d{1,4}(?=_npu\d))",
            re.IGNORECASE,
        ),
    ),
    Rule(
        id="hostname-assignment",
        description="host name in a key/value assignment",
        hint="replace the host name with <host>",
        pattern=re.compile(
            r"\b(?:host(?:name)?|server|node|machine)\s*[:=]\s*[\"']?"
            + _PLACEHOLDER_START + r"(?=[\w.-]*[\d.-])([A-Za-z0-9][\w.-]+)",
            re.IGNORECASE,
        ),
    ),
    # --- identifiers --------------------------------------------------------
    Rule(
        id="employee-id",
        description="employee-style identifier (letter(s) + 8 digits)",
        hint="remove personal identifiers",
        pattern=re.compile(r"(?<![\w-])[a-z]{1,2}\d{8}(?![\w-])", re.IGNORECASE),
    ),
    Rule(
        id="ticket-id",
        description="ticket / tracker identifier (ABC-1234, DTS...)",
        hint="internal tickets are not followable evidence; reference a public "
        "issue, PR or commit instead",
        pattern=re.compile(r"(?<![\w-])(?:[A-Z][A-Z0-9]{1,9}-\d{2,7}|DTS\d{10,16})(?![\w-])"),
    ),
    Rule(
        id="credential-high-entropy",
        description="long mixed-case alphanumeric blob (possible secret)",
        hint="if this is not a secret, allowlist it explicitly",
        finder=_high_entropy_tokens,
    ),
)

RULE_IDS = tuple(r.id for r in RULES)


# --------------------------------------------------------------------------- #
# Allowlist
# --------------------------------------------------------------------------- #

#: Exact (case-insensitive) values that are public by construction.
BUILTIN_ALLOWLIST: frozenset[str] = frozenset(
    s.lower()
    for s in (
        "0.0.0.0",
        "127.0.0.1",
        "255.255.255.0",
        "255.255.255.255",
        "::",
        "::1",
        "localhost",
        "root",
        "SHA-224",
        "SHA-256",
        "SHA-384",
        "SHA-512",
        "UTF-16",
        "UTF-32",
        "ISO-8859",
        "git@github.com",
        # Prose words that follow "password:" / "token:" in a sentence and are
        # not values: "the token: expired" is a symptom, not a credential.
        "expired", "required", "missing", "invalid", "rejected", "redacted",
        "omitted", "unset", "unknown", "changed", "rotated", "present", "absent",
        "correct", "incorrect", "mismatch", "needed", "ignored", "accepted",
    )
)

#: Regular expressions (full match, case-insensitive) for public value classes.
BUILTIN_ALLOW_PATTERNS: tuple[str, ...] = (
    r"192\.0\.2\.\d{1,3}(?:\s*-\s*[\d.]+)?(?:/\d{1,2})?",  # RFC 5737 TEST-NET-1
    r"198\.51\.100\.\d{1,3}(?:\s*-\s*[\d.]+)?(?:/\d{1,2})?",  # RFC 5737 TEST-NET-2
    r"203\.0\.113\.\d{1,3}(?:\s*-\s*[\d.]+)?(?:/\d{1,2})?",  # RFC 5737 TEST-NET-3
    r"2001:0?db8:[0-9a-f:]*",  # RFC 3849 documentation prefix
    r"127\.\d{1,3}\.\d{1,3}\.\d{1,3}",  # loopback
    r"[\w.+-]+@(?:[\w-]+\.)*example\.(?:com|org|net)",  # RFC 2606 e-mail
    r"(?:[\w-]+\.)*example\.(?:com|org|net|local|internal|lan|home|corp|priv|private|localdomain|localnet|intranet)",
    r"[\w.+-]+@(?:users\.)?noreply\.github\.com",
    r"RFC-\d+",
    r"<[^<>]+>",  # <placeholder>
)


class Allowlist:
    """Combined built-in + run-time allowlist."""

    def __init__(self, terms: Iterable[str] = (), patterns: Iterable[str] = ()):
        self.terms: set[str] = set(BUILTIN_ALLOWLIST)
        self.patterns: list["re.Pattern[str]"] = [
            re.compile(p, re.IGNORECASE) for p in BUILTIN_ALLOW_PATTERNS
        ]
        for term in terms:
            self.add(term)
        for pat in patterns:
            self.add("re:" + pat)

    def add(self, term: str) -> None:
        term = term.strip()
        if not term:
            return
        if term.startswith("re:"):
            try:
                self.patterns.append(re.compile(term[3:], re.IGNORECASE))
            except re.error as exc:
                raise ToolError(f"allowlist: invalid regular expression {term[3:]!r}: {exc}") from None
        else:
            self.terms.add(term.lower())

    @classmethod
    def from_args(cls, terms: Sequence[str] = (), files: Sequence[str] = ()) -> "Allowlist":
        allow = cls(terms)
        for raw in files:
            doc = load_document(Path(raw))
            if not isinstance(doc, list) or not all(isinstance(x, str) for x in doc):
                raise ToolError(
                    f"{raw}: allow-file must be a YAML/JSON list of strings "
                    "(prefix a regular expression with 're:')"
                )
            for item in doc:
                allow.add(item)
        return allow

    def is_allowed(self, value: str) -> bool:
        v = value.strip()
        if v.lower() in self.terms:
            return True
        return any(p.fullmatch(v) for p in self.patterns)


# --------------------------------------------------------------------------- #
# Scanner
# --------------------------------------------------------------------------- #

@dataclass
class Finding:
    path: str
    rule: str
    value: str
    hint: str
    file: str | None = None

    def masked(self) -> str:
        v = self.value
        if len(v) <= 4:
            body = "…"
        else:
            body = f"{v[:2]}…{v[-2:]}"
        return f"{body} ({len(v)} chars)"

    def render(self, show_matches: bool = False) -> str:
        where = f"{self.file}: {self.path}" if self.file else self.path
        shown = self.value if show_matches else self.masked()
        return f"{where}: [{self.rule}] {shown} — {self.hint}"


def scan_text(text: str, allow: Allowlist | None = None, path: str = "<text>") -> list[Finding]:
    """Scan one string. Earlier rules claim their spans; later rules skip overlaps."""
    allow = allow or Allowlist()
    findings: list[Finding] = []
    claimed: list[tuple[int, int]] = []
    for rule in RULES:
        for start, end in rule.spans(text):
            if start == end:
                continue
            if any(s < end and start < e for s, e in claimed):
                continue
            value = text[start:end]
            if allow.is_allowed(value):
                claimed.append((start, end))
                continue
            claimed.append((start, end))
            findings.append(Finding(path=path, rule=rule.id, value=value, hint=rule.hint))
    return findings


def scan_tree(tree: Any, allow: Allowlist | None = None, path: tuple[Any, ...] = ()) -> list[Finding]:
    """Scan every string in a parsed YAML/JSON tree, keys included."""
    allow = allow or Allowlist()
    out: list[Finding] = []
    if isinstance(tree, Mapping):
        for key, value in tree.items():
            if isinstance(key, str):
                out.extend(scan_text(key, allow, json_pointer(path + (f"<key {key!r}>",))))
            out.extend(scan_tree(value, allow, path + (key,)))
    elif isinstance(tree, (list, tuple)):
        for index, value in enumerate(tree):
            out.extend(scan_tree(value, allow, path + (index,)))
    elif isinstance(tree, str):
        out.extend(scan_text(tree, allow, json_pointer(path)))
    return out



def redact_text(text: str, allow: Allowlist | None = None) -> str:
    """Replace each scanner span completely; never reveal token prefixes."""
    allow = allow or Allowlist()
    claimed = []
    replacements = []
    for rule in RULES:
        for start, end in rule.spans(text):
            if start == end or any(left < end and start < right for left, right in claimed):
                continue
            claimed.append((start, end))
            if not allow.is_allowed(text[start:end]):
                replacements.append((start, end, "<redacted:" + rule.id + ">"))
    for start, end, replacement in sorted(replacements, reverse=True):
        text = text[:start] + replacement + text[end:]
    return text
