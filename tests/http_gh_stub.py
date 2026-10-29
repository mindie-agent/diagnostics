"""Test-only gh adapter that streams a local HTTP server through real pipes."""

from __future__ import annotations

import http.client
import os
import sys


def main():
    args = sys.argv[1:]
    method = args[args.index("--method") + 1]
    assert args[args.index("--hostname") + 1] == "github.com"
    assert "--include" in args
    path = next(arg for arg in args if arg.startswith(("repos/", "search/")))
    payload = sys.stdin.buffer.read() if "--input" in args else None
    connection = http.client.HTTPConnection(
        "127.0.0.1", int(os.environ["DIAGNOSTICS_TEST_HTTP_PORT"]), timeout=5,
    )
    try:
        connection.request(method, "/" + path, body=payload)
        response = connection.getresponse()
        output = sys.stdout.buffer
        output.write(f"HTTP/1.1 {response.status} {response.reason}\r\n".encode())
        for key, value in response.getheaders():
            output.write(f"{key}: {value}\r\n".encode())
        output.write(b"\r\n")
        while block := response.read(4096):
            output.write(block)
            output.flush()
        return int(response.status >= 400)
    finally:
        connection.close()


if __name__ == "__main__":
    raise SystemExit(main())
