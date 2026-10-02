"""The full-screen output window and the hub that feeds it (handoff §10.2: "projector shows the full-screen output on the projector
display; the laptop shows an operator panel").

The window itself is a web page (src/projector/output.html) served by the show server and opened in a browser on the projector
display, because that needs no extra dependency and is the same mechanism the pen tool's projector window already uses. The page:
  * shows ONE image, the projector-pixel frame the server rendered (warped, glass exactly 0), 1:1 on the device pixels;
  * crossfades between frames in the browser (a blend of two frames that are both 0 in the glass is 0 in the glass);
  * reports its real pixel size and fullscreen state so the server renders at the projector's native resolution;
  * offers "Fullscreen on the projector" (Chrome's Window Management API) or F / double-click to fullscreen where it stands.

FrameHub is the transport: the server publishes finished frames and state; pages subscribe by Server-Sent Events and fetch the
frame bytes by sequence number. It holds only a few recent frames.
"""
import json
import os
import queue
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_HTML = os.path.join(HERE, "output.html")
KEEP_FRAMES = 12


class FrameHub:
    def __init__(self):
        self._lock = threading.Lock()
        self._subs = []
        self._frames = {}
        self.seq = 0
        self.last_state = None

    def publish_frame(self, png_bytes, meta):
        """meta: {"kind": "frame" | "pattern" | "black", "name": ..., "w": int, "h": int, ...}. Returns the sequence number."""
        with self._lock:
            self.seq += 1
            seq = self.seq
            self._frames[seq] = png_bytes
            for old in sorted(self._frames)[:-KEEP_FRAMES]:
                del self._frames[old]
            msg = ("frame", dict(meta, seq=seq, t=time.time()))
            subs = list(self._subs)
        for q in subs:
            q.put(msg)
        return seq

    def publish_state(self, state):
        with self._lock:
            self.last_state = state
            subs = list(self._subs)
        for q in subs:
            q.put(("state", state))

    def frame(self, seq):
        with self._lock:
            return self._frames.get(seq)

    def latest(self):
        with self._lock:
            return (self.seq, self._frames.get(self.seq))

    def subscribe(self):
        q = queue.Queue()
        with self._lock:
            self._subs.append(q)
            seq, state = self.seq, self.last_state
        if state is not None:
            q.put(("state", state))
        if seq:
            q.put(("frame", dict(seq=seq, kind="resend", t=time.time())))
        return q

    def unsubscribe(self, q):
        with self._lock:
            if q in self._subs:
                self._subs.remove(q)

    @property
    def n_subscribers(self):
        with self._lock:
            return len(self._subs)


def sse_message(event, data):
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()
