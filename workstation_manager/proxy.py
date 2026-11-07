"""Front door for :8005 / :8006. Callers keep their URLs; requests wait while the
interactive pair is down and go through once it is back.

Only a request marked `X-GPU-Hold: 1` counts as interactive: the owner using the
assistant. It resets the hold that keeps batch off the card. Everything else
(pipelines, captures, compiles, wherever they come from) waits without holding.
"""

from __future__ import annotations

import http.client
import http.server
import json
import select
import socket
import threading
import time
import urllib.parse
from collections import Counter
from typing import Callable, Dict, Optional

# admit() outcomes
ADMITTED, TIMED_OUT, GONE = "admitted", "timed out", "gone"

HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te",
       "trailers", "transfer-encoding", "upgrade", "content-length", "host"}


class Gate:
    def __init__(self, queue_wait_seconds: float):
        self.cond = threading.Condition()
        self.ok: Dict[int, bool] = {}
        self.closed = False
        self.in_flight = 0
        self.waiting = 0
        self.waiting_interactive = 0
        self.last_interactive_at = 0.0
        # Which registered model the llm runs, requests waiting for a specific one,
        # and when each was last asked for by name.
        self.loaded: Optional[str] = None
        self.waiting_for: Counter = Counter()
        self.asked_at: Dict[str, float] = {}
        # Per backend port: requests waiting, requests in flight, and when it was last
        # used (an on-demand service is stopped once idle long enough).
        self.waiting_on: Counter = Counter()
        self.busy_on: Counter = Counter()
        self.used_at: Dict[int, float] = {}
        self.queue_wait_seconds = queue_wait_seconds

    def set_ok(self, port: int, ok: bool) -> None:
        with self.cond:
            self.ok[port] = ok
            self.cond.notify_all()

    def close_and_drain(self, seconds: float) -> int:
        with self.cond:
            self.closed = True
            deadline = time.time() + seconds
            while self.in_flight and time.time() < deadline:
                self.cond.wait(1)
            return self.in_flight

    def reopen(self) -> None:
        with self.cond:
            self.closed = False
            self.cond.notify_all()

    def set_loaded(self, model: Optional[str]) -> None:
        with self.cond:
            self.loaded = model
            self.cond.notify_all()

    def admit(self, backend: int, interactive: bool,
              gone: Optional[Callable[[], bool]] = None, model: Optional[str] = None) -> str:
        """Wait until the backend is up, and running `model` if one is named. A caller
        that disconnects while waiting is dropped, so requests nobody is waiting for
        are never sent on later."""
        with self.cond:
            if interactive:
                self.last_interactive_at = time.time()
                self.waiting_interactive += 1
            if model:
                self.asked_at[model] = time.time()
                self.waiting_for[model] += 1
            self.waiting += 1
            self.waiting_on[backend] += 1
            deadline = time.time() + self.queue_wait_seconds
            try:
                while self.closed or not self.ok.get(backend) or (model and self.loaded != model):
                    left = deadline - time.time()
                    if left <= 0:
                        return TIMED_OUT
                    if gone is not None and gone():
                        return GONE
                    self.cond.wait(min(left, 1))
                self.in_flight += 1
                self.busy_on[backend] += 1
                self.used_at[backend] = time.time()
                return ADMITTED
            finally:
                if model:
                    self.asked_at[model] = time.time()
                    self.waiting_for[model] -= 1
                    if self.waiting_for[model] <= 0:
                        del self.waiting_for[model]
                self.waiting -= 1
                self.waiting_on[backend] -= 1
                if interactive:
                    self.waiting_interactive -= 1

    def done(self, backend: Optional[int] = None) -> None:
        with self.cond:
            self.in_flight -= 1
            if backend is not None:
                self.busy_on[backend] -= 1
                self.used_at[backend] = time.time()
            self.cond.notify_all()


def client_gone(sock) -> bool:
    """True once the caller has closed its end. The request body is already read, so
    a readable socket here means EOF (or a pipelined request, which counts as alive)."""
    try:
        readable, _, _ = select.select([sock], [], [], 0)
        return bool(readable) and sock.recv(1, socket.MSG_PEEK) == b""
    except OSError:
        return True


HOLD_HEADER = "X-GPU-Hold"


def holds_pair(headers) -> bool:
    """Whether a request is the owner using the assistant (`X-GPU-Hold: 1`)."""
    return (headers.get(HOLD_HEADER) or "").strip().lower() in ("1", "true", "yes", "on")


def requested_model(body: bytes, path: str = "") -> Optional[str]:
    """The model a request names: `"model"` in a JSON body, or `?model=` in the URL
    (so a GET such as /props can ask for a specific model)."""
    query = urllib.parse.parse_qs(urllib.parse.urlsplit(path).query).get("model")
    if query:
        return query[0]
    if not body or body[:1] != b"{":
        return None
    try:
        value = json.loads(body).get("model")
    except (ValueError, AttributeError):
        return None
    return value if isinstance(value, str) else None


def make_handler(gate: Gate, backend: int, resolve: Callable[[Optional[str]], Optional[str]] = lambda m: None):
    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def log_message(self, *args):
            pass

        def _body(self) -> bytes:
            n = self.headers.get("Content-Length")
            if n is not None:
                return self.rfile.read(int(n))
            if "chunked" in self.headers.get("Transfer-Encoding", "").lower():
                out = b""
                while True:
                    size = int(self.rfile.readline().split(b";")[0].strip() or b"0", 16)
                    if size == 0:
                        while self.rfile.readline() not in (b"\r\n", b"\n", b""):
                            pass
                        return out
                    out += self.rfile.read(size)
                    self.rfile.readline()
            return b""

        def _plain(self, code: int, text: str) -> None:
            data = text.encode()
            self.send_response(code)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _handle(self):
            body = self._body()
            if self.path.rstrip("/") == "/health" and not (gate.ok.get(backend) and not gate.closed):
                return self._plain(503, "workstation-manager: interactive pair is down; requests queue\n")
            outcome = gate.admit(backend, holds_pair(self.headers),
                                 gone=lambda: client_gone(self.connection),
                                 model=resolve(requested_model(body, self.path)))
            if outcome == GONE:
                self.close_connection = True
                return
            if outcome == TIMED_OUT:
                return self._plain(503, "workstation-manager: timed out waiting for the GPU\n")
            try:
                conn = http.client.HTTPConnection("127.0.0.1", backend, timeout=None)
                headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP}
                if body or self.command in ("POST", "PUT", "PATCH"):
                    headers["Content-Length"] = str(len(body))
                conn.request(self.command, self.path, body=body or None, headers=headers)
                resp = conn.getresponse()
                self.send_response(resp.status, resp.reason)
                for k, v in resp.getheaders():
                    if k.lower() not in HOP:
                        self.send_header(k, v)
                length = resp.getheader("Content-Length")
                if length is not None and "chunked" not in (resp.getheader("Transfer-Encoding") or ""):
                    self.send_header("Content-Length", length)
                self.send_header("Connection", "close")
                self.end_headers()
                while True:
                    chunk = resp.read1(65536)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()
                conn.close()
            except (ConnectionError, http.client.HTTPException, OSError):
                try:
                    self._plain(502, "workstation-manager: backend went away mid-request\n")
                except OSError:
                    pass
            finally:
                gate.done(backend)

        do_GET = do_POST = do_PUT = do_DELETE = do_PATCH = do_OPTIONS = do_HEAD = _handle

    return Handler


def serve(gate: Gate, listen, port: int, backend: int, resolve=lambda m: None):
    servers = []
    for addr in listen:
        srv = http.server.ThreadingHTTPServer((addr, port), make_handler(gate, backend, resolve))
        srv.daemon_threads = True
        threading.Thread(target=srv.serve_forever, daemon=True,
                         name=f"proxy-{addr}:{port}").start()
        servers.append(srv)
    return servers
