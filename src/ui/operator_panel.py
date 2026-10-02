"""The show server: operator panel (laptop), full-screen output window (projector) and audience screen (companion display) over one
local HTTP server (handoff §10.2, §10.3). Standard library only; nothing leaves this machine unless started with --lan.

    /            operator panel: knobs, preview, overlay, calibration, test patterns, demo, re-solve        (this machine only)
    /output      the projector window: put it on the projector display and press F (or use its button)
    /audience    the companion screen: what each look is, and why it reads as one light
    /api/...     state, Server-Sent Events, frames, overlay, actions

Safety. Every POST needs the per-run token that only the operator page is given, comes from this machine, and must carry a
loopback Host header (DNS-rebinding guard, as in the pen tool). The operator page and every action are refused for other machines
even with --lan, which only exposes the read-only pages (/output, /audience, state, frames). Re-solve cannot spend credits unless
the server was started with --allow-spend and the confirm request repeats the exact payload hash and credit cost it was shown.
"""
import json
import os
import re
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from src.projector.output_window import OUTPUT_HTML, sse_message
from src.ui import keys

HERE = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(HERE, "web")
PAGES = {"/": os.path.join(WEB, "operator.html"), "/output": OUTPUT_HTML, "/audience": os.path.join(WEB, "audience.html")}
STATIC = {"/web/common.css": ("common.css", "text/css; charset=utf-8")}
LOOPBACK = ("127.0.0.1", "::1", "::ffff:127.0.0.1")
MAX_BODY = 200_000
KEEPALIVE_S = 15


class ShowServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def _read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def make_handler(session, token, hosts, lan):
    state = dict(output_conns=0)
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        # ---- helpers
        def _send(self, code, body, ctype, extra=None):
            if isinstance(body, str):
                body = body.encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code, obj):
            self._send(code, json.dumps(obj), "application/json")

        def _local(self):
            return self.client_address[0] in LOOPBACK

        def _host_ok(self):
            return self.headers.get("Host") in hosts

        def _guard_operator(self):
            """Operator-only: loopback client and a loopback Host header."""
            if not (self._local() and self._host_ok()):
                self._send(403, "forbidden: operator controls are available on this machine only", "text/plain")
                return False
            return True

        def _guard_read(self):
            if lan:
                return True
            if not self._host_ok():
                self._send(403, "forbidden", "text/plain")
                return False
            return True

        # ---- GET
        def do_GET(self):
            url = urlparse(self.path)
            path = url.path
            if path == "/":
                if not self._guard_operator():
                    return
                page = _read(PAGES["/"]).replace("__TOKEN__", token)
                init = dict(keys=keys.bindings_json(), help={f: keys.help_rows(f) for f in ("relief", "classical")})
                return self._send(200, page.replace("__INIT__", json.dumps(init).replace("<", "\\u003c")), "text/html; charset=utf-8")
            if not self._guard_read():
                return
            if path in ("/output", "/audience"):
                return self._send(200, _read(PAGES[path]), "text/html; charset=utf-8")
            if path in STATIC:
                name, ctype = STATIC[path]
                return self._send(200, _read(os.path.join(WEB, name)), ctype)
            if path == "/api/state":
                return self._json(200, session.state())
            if path == "/api/events":
                return self._events(parse_qs(url.query).get("role", ["page"])[0])
            m = re.fullmatch(r"/api/frame/(\d+)\.png", path)
            if m:
                data = session.hub.frame(int(m.group(1)))
                return self._send(200, data, "image/png") if data else self._send(404, "gone", "text/plain")
            if path == "/api/latest.png":
                seq, data = session.hub.latest()
                return self._send(200, data, "image/png", {"X-Seq": str(seq)}) if data else self._send(404, "no frame yet", "text/plain")
            if path == "/api/overlay.png":
                import io
                buf = io.BytesIO()
                gaps = parse_qs(url.query).get("gaps", ["source"])[0]
                with session.lock:
                    session.overlay_image(gaps).save(buf, "PNG")
                return self._send(200, buf.getvalue(), "image/png")
            m = re.fullmatch(r"/api/asset/([a-z_]+)\.png", path)
            if m:
                p = session.asset_path(m.group(1))
                if p and os.path.exists(p):
                    with open(p, "rb") as f:
                        return self._send(200, f.read(), "image/png")
                return self._send(404, "not found", "text/plain")
            if path == "/api/coherence":
                return self._json(200, session.coherence())
            self._send(404, "not found", "text/plain")

        def _events(self, role):
            q = session.hub.subscribe()
            if role == "output":
                with lock:
                    state["output_conns"] += 1
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            try:
                while True:
                    try:
                        event, data = q.get(timeout=KEEPALIVE_S)
                        self.wfile.write(sse_message(event, data))
                    except Exception as e:                                   # noqa: BLE001
                        if isinstance(e, (BrokenPipeError, ConnectionResetError, OSError)):
                            break
                        self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                session.hub.unsubscribe(q)
                if role == "output":
                    with lock:
                        state["output_conns"] -= 1
                        gone = state["output_conns"] <= 0
                    if gone:
                        session.output_gone()

        # ---- POST
        def do_POST(self):
            path = urlparse(self.path).path
            if not self._host_ok():
                return self._json(403, dict(error="bad host"))
            try:
                n = int(self.headers.get("Content-Length", 0))
                if n > MAX_BODY:
                    return self._json(413, dict(error="body too large"))
                body = json.loads(self.rfile.read(n) or b"{}")
            except (ValueError, OSError) as e:
                return self._json(400, dict(error=f"bad request: {e}"))
            if path == "/api/output/report":                       # the output page reporting its real pixel size (no token: it has none)
                try:
                    session.set_output(body["w"], body["h"], body.get("fullscreen", False), body.get("dpr", 1.0))
                    return self._json(200, dict(ok=True))
                except (KeyError, ValueError, TypeError) as e:
                    return self._json(400, dict(error=str(e)))
            if not self._local() or self.headers.get("X-Token") != token:
                return self._json(403, dict(error="bad token or not the operator machine"))
            try:
                if path == "/api/action":
                    return self._json(200, dict(ok=True, **session.dispatch(body["name"], **(body.get("args") or {}))))
                if path == "/api/resolve/plan":
                    return self._json(200, session.resolve_plan(body.get("engine", "qdrive")))
                if path == "/api/resolve/confirm":
                    return self._json(200, session.resolve_confirm(body["engine"], body["sha256"], body["credits"]))
                self._json(404, dict(error="not found"))
            except PermissionError as e:
                self._json(403, dict(ok=False, error=str(e)))
            except (ValueError, KeyError, TypeError, OSError, FileNotFoundError) as e:
                self._json(400, dict(ok=False, error=f"{type(e).__name__}: {e}"))

    return Handler


def make_server(session, host="127.0.0.1", port=0, lan=False):
    """Create (not start) the server. Returns (server, token, url_base)."""
    token = secrets.token_urlsafe(16)
    server = ShowServer((host, port), lambda *a, **k: None)
    real = server.server_address[1]
    hosts = {f"127.0.0.1:{real}", f"localhost:{real}"}
    if lan:
        import socket
        for ip in {i[4][0] for i in socket.getaddrinfo(socket.gethostname(), None) if ":" not in i[4][0]}:
            hosts.add(f"{ip}:{real}")
        hosts.add(f"{socket.gethostname()}:{real}")
        hosts.add(f"{socket.gethostname()}.local:{real}")
    server.RequestHandlerClass = make_handler(session, token, hosts, lan)
    return server, token, f"http://127.0.0.1:{real}"
