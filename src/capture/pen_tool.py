#!/usr/bin/env python3
"""Pen tool: draw panel and glass polygons, set plane ids and angles, export labels.json + mask.png.

    python -m src.capture.pen_tool draw  [PHOTO] [--size 1920x1080] [--labels labels.json] [--out runs/scene] [--cap 20]
    python -m src.capture.pen_tool build LABELS [--photo PHOTO] [--out DIR]     # re-export without the GUI

Two ways to draw, and you can switch at any time in the page:
  * NO PHOTO (default): the canvas is the projector's own screen, black, with outline-only shapes. Connect the
    projector as a display and draw while watching where the lines land on the real object. Mirrored display: press F
    for fullscreen and Tab to hide the sidebar. Extended display: "Open projector window", drag it to the projector and
    press F there; it shows your shapes and cursor live, and its size becomes the canvas, so the aspect always matches.
    Shapes are in projector pixels, so no camera-to-projector warp is needed later (labels `source: "projector"`).
  * PHOTO: pass a photo (or pick or drop one in the page) taken from the projector's position and trace over it
    (`source: "photo"`; it will need a warp to projector space).

Shared edges (like perspective editing): new vertices snap to existing vertices and edges (hold Alt to place freely); when a shape closes,
vertices that land on a neighbour's edge are inserted into it, so touching shapes have matching vertices along the shared edge. Drag a
vertex or an edge to reshape: every vertex coincident with it moves too (shared ones show filled yellow). Geometry: pen_geom.js.

`draw` serves the page on 127.0.0.1 (stdlib only, nothing leaves this machine). Outputs, written on Save into --out:
    labels.json          polygons + plane ids + angles (schema in src/capture/labels.py)
    mask.png             canvas-sized; white = projectable, black = impenetrable (glass / off-limits / outside)
    patches_preview.png  the shapes (over the photo, if any) and the patch cells that would get a qubit

The qubit budget (patches + 2 lamps + one polarity qubit per panel) is printed on every save and shown live;
it warns when it exceeds --cap (default 20).
"""
import argparse
import io
import json
import os
import secrets
import sys
import threading
import webbrowser
from urllib.parse import unquote
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from src.capture.labels import CANVAS_WIDTH, GRID_DEFAULT, canvas_for, fill_defaults, save_labels, validate
from src.geometry.planes import build_scene
from src.graph.allocate_qubits import CAP_WARN, budget, format_budget, suggest_grids
from src.graph.patches import build_patches

HERE = os.path.dirname(os.path.abspath(__file__))
MAX_BODY = 5_000_000
MAX_PHOTO = 40_000_000
PROJECTOR_SIZE = (1920, 1080)
DISPLAY_MAX_SIDE = 2400


# ------------------------------------------------------------------ core (no GUI)
def load_photo(src, name=None):
    """Open a photo (path or file object) upright (EXIF orientation applied) as RGB. `name` is used in messages."""
    name = name or str(src)
    try:
        return ImageOps.exif_transpose(Image.open(src)).convert("RGB")
    except Exception as e:                                       # noqa: BLE001 - surface a friendly message
        hint = ""
        if name.lower().endswith((".heic", ".heif")):
            hint = f"\nPillow cannot read HEIC. Convert first:  sips -s format jpeg '{name}' --out photo.jpg"
        raise SystemExit(f"cannot open photo {name!r}: {e}{hint}")


def summarise(labels, cap=CAP_WARN):
    """Validate labels and compute the patch graph size and qubit budget. Returns (summary dict, scene, cells)."""
    labels = validate(fill_defaults(labels))
    scene = build_scene(labels)
    g = labels["grid"]
    cells, _ = build_patches(scene, g["nx"], g["ny"], g["min_cover"])
    b = budget(len(cells), scene.n_panels, cap)
    b["text"] = format_budget(b)
    b["suggestions"] = suggest_grids(scene, cap, g["min_cover"]) if b["over_cap"] else []
    b["cells"] = [[c["i"], c["j"], c["k"]] for c in cells]
    b["canvas"] = labels["canvas"]
    return b, scene, cells


def make_preview(scene, cells, photo):
    """Photo (or dark grey) at canvas size with impenetrable areas darkened, panel outlines, and patch cells."""
    base = photo.resize((scene.W, scene.H)) if photo is not None else Image.new("RGB", (scene.W, scene.H), (60, 60, 60))
    arr = np.array(base).astype(float)
    arr[scene.impenetrable] *= 0.25
    arr[scene.blocked] = arr[scene.blocked] * 0.5 + np.array([40, 90, 200]) * 0.5
    im = Image.fromarray(arr.astype(np.uint8))
    d = ImageDraw.Draw(im)
    font = ImageFont.load_default()
    for p in scene.panels:
        d.polygon([tuple(pt) for pt in p.polygon], outline=(255, 255, 255))
        cx = sum(pt[0] for pt in p.polygon) / len(p.polygon)
        cy = sum(pt[1] for pt in p.polygon) / len(p.polygon)
        d.text((cx - 30, cy - 6), f"{p.name} plane {p.plane_id} {p.angle_deg:g}deg", fill=(255, 255, 255), font=font)
    for n, c in enumerate(cells):
        x0, x1, y0, y1 = c["box"]
        d.rectangle([x0, y0, x1 - 1, y1 - 1], outline=(255, 214, 0))
        d.text((x0 + 3, y0 + 2), str(n), fill=(255, 214, 0), font=font)
    return im


def export_outputs(labels, out_dir, photo=None, cap=CAP_WARN):
    """Write labels.json, mask.png and patches_preview.png into out_dir; return the budget summary."""
    summary, scene, cells = summarise(labels, cap)
    labels = fill_defaults(labels)
    os.makedirs(out_dir, exist_ok=True)
    save_labels(labels, os.path.join(out_dir, "labels.json"))
    Image.fromarray((scene.frame * 255).astype(np.uint8), "L").save(os.path.join(out_dir, "mask.png"))
    make_preview(scene, cells, photo).save(os.path.join(out_dir, "patches_preview.png"))
    summary["files"] = {n: os.path.join(out_dir, n) for n in ("labels.json", "mask.png", "patches_preview.png")}
    return summary


def print_report(summary):
    print(f"\npatch count: {summary['patches']}  (grid cells that are >= min_cover projectable)")
    print(f"qubit budget: {format_budget(summary)}")
    if summary["over_cap"] and summary["suggestions"]:
        s = ", ".join(f"{g['nx']}x{g['ny']} -> {g['total']} qubits" for g in summary["suggestions"])
        print(f"grids that fit under {summary['cap']}: {s}")
    for n, path in summary.get("files", {}).items():
        print(f"  wrote {path}")


# ------------------------------------------------------------------ GUI server
STATIC = {"/pen_draw.js": ("pen_draw.js", "text/javascript; charset=utf-8"),
          "/pen_geom.js": ("pen_geom.js", "text/javascript; charset=utf-8"),
          "/projector": ("projector.html", "text/html; charset=utf-8")}


def display_jpeg(photo):
    disp = photo.copy()
    disp.thumbnail((DISPLAY_MAX_SIDE, DISPLAY_MAX_SIDE))
    buf = io.BytesIO()
    disp.save(buf, "JPEG", quality=88)
    return buf.getvalue()


def clean_name(raw):
    name = os.path.basename(unquote(raw or "photo"))
    return "".join(c for c in name if c.isprintable())[:100] or "photo"


def make_handler(state):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):          # keep the terminal for the budget report
            pass

        def _send(self, code, body, ctype):
            if isinstance(body, str):
                body = body.encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code, obj):
            self._send(code, json.dumps(obj), "application/json")

        def _guard(self):
            if self.headers.get("Host") not in state["hosts"]:          # DNS-rebinding guard
                self._send(403, "forbidden", "text/plain")
                return False
            return True

        def do_GET(self):
            if not self._guard():
                return
            path = self.path.split("?")[0]
            if path == "/":
                with open(os.path.join(HERE, "pen_tool.html")) as f:
                    page = f.read()
                init = json.dumps(state["init"]).replace("<", "\\u003c")
                self._send(200, page.replace("__INIT__", init).replace("__TOKEN__", state["token"]), "text/html; charset=utf-8")
            elif path in STATIC:
                name, ctype = STATIC[path]
                with open(os.path.join(HERE, name)) as f:
                    self._send(200, f.read(), ctype)
            elif path == "/photo.jpg" and state["display_jpeg"]:
                self._send(200, state["display_jpeg"], "image/jpeg")
            else:
                self._send(404, "not found", "text/plain")

        def do_POST(self):
            if not self._guard():
                return
            if self.headers.get("X-Token") != state["token"]:
                return self._json(403, {"error": "bad token"})
            n = int(self.headers.get("Content-Length", 0))
            if n > (MAX_PHOTO if self.path == "/api/photo" else MAX_BODY):
                return self._json(413, {"error": "body too large"})
            try:
                body = self.rfile.read(n)
                if self.path == "/api/photo":                      # optional photo, uploaded from the page
                    name = clean_name(self.headers.get("X-Filename"))
                    photo = load_photo(io.BytesIO(body), name)
                    state.update(photo=photo, display_jpeg=display_jpeg(photo), photo_name=name)
                    print(f"photo: {state['photo_name']} {photo.size[0]}x{photo.size[1]}")
                    return self._json(200, dict(image=dict(file=state["photo_name"], width=photo.size[0], height=photo.size[1])))
                if self.path == "/api/photo/clear":
                    state.update(photo=None, display_jpeg=None, photo_name=None)
                    return self._json(200, {})
                labels = json.loads(body)
                if self.path == "/api/budget":
                    summary, _, _ = summarise(labels, state["cap"])
                    return self._json(200, summary)
                if self.path == "/api/save":
                    summary = export_outputs(labels, state["out"], state["photo"], state["cap"])
                    print_report(summary)
                    return self._json(200, summary)
                self._json(404, {"error": "not found"})
            except SystemExit as e:                                  # load_photo's friendly errors
                self._json(400, {"error": str(e)})
            except (ValueError, KeyError, TypeError, OSError) as e:
                self._json(400, {"error": f"{type(e).__name__}: {e}"})

    return Handler


def run_gui(photo_path, out_dir, labels_path, cap, port, canvas_width, open_browser, size=PROJECTOR_SIZE):
    from src.capture.labels import load_labels
    sys.stdout.reconfigure(line_buffering=True)
    photo = load_photo(photo_path) if photo_path else None
    existing = load_labels(labels_path) if labels_path else None
    if existing and existing.get("source", "photo") == "photo" and photo is None:
        raise SystemExit(f"{labels_path} was drawn on a photo; pass that photo too (or start without --labels)")
    if existing and photo is not None and (existing["image"]["width"], existing["image"]["height"]) != photo.size:
        raise SystemExit(f"{labels_path} was drawn on a {existing['image']['width']}x{existing['image']['height']} "
                         f"image but {photo_path} is {photo.size[0]}x{photo.size[1]}")
    w, h = photo.size if photo else (existing["image"]["width"], existing["image"]["height"]) if existing else size
    state = dict(token=secrets.token_urlsafe(16), cap=cap, out=out_dir, photo=photo, hosts=set(),
                 display_jpeg=display_jpeg(photo) if photo else None, photo_name=os.path.basename(photo_path) if photo_path else None)
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(state))
    real_port = server.server_address[1]
    state["hosts"] = {f"127.0.0.1:{real_port}", f"localhost:{real_port}"}
    state["init"] = dict(image=dict(file=state["photo_name"], width=w, height=h), canvas_width=canvas_width,
                         canvas=existing["canvas"] if existing else canvas_for(w, h, canvas_width), size=list(size),
                         grid=existing["grid"] if existing else dict(GRID_DEFAULT), labels=existing, cap=cap, out=out_dir)
    url = f"http://127.0.0.1:{real_port}/"
    what = f"photo {w}x{h}" if photo else f"no photo: projector space {w}x{h}"
    print(f"pen tool: {url}   ({what}, outputs -> {out_dir})")
    print(f"projector output window: {url}projector   (or use the button in the page)")
    print("Ctrl+C to quit. Save writes labels.json, mask.png, patches_preview.png and prints the qubit budget here.")
    if open_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


def run_build(labels_path, photo_path, out_dir, cap):
    from src.capture.labels import load_labels
    labels = load_labels(labels_path)
    photo = None
    cand = photo_path or (os.path.join(os.path.dirname(labels_path), labels["image"]["file"]) if labels["image"].get("file") else None)
    if cand and os.path.exists(cand):
        photo = load_photo(cand)
    print_report(export_outputs(labels, out_dir, photo, cap))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("draw", help="open the drawing page")
    d.add_argument("photo", nargs="?", help="optional: a photo to trace over; omit to draw directly in projector space")
    d.add_argument("--size", default=f"{PROJECTOR_SIZE[0]}x{PROJECTOR_SIZE[1]}",
                   help="projector resolution WxH to start from when there is no photo (the projector window overrides it)")
    d.add_argument("--labels", help="resume from an existing labels.json (with its photo, if it was drawn on one)")
    d.add_argument("--out", default="runs/scene")
    d.add_argument("--cap", type=int, default=CAP_WARN, help="qubit count above which to warn (default 20)")
    d.add_argument("--port", type=int, default=0, help="default: any free port")
    d.add_argument("--canvas-width", type=int, default=CANVAS_WIDTH, help="working resolution width (default 1200)")
    d.add_argument("--no-browser", action="store_true")
    b = sub.add_parser("build", help="re-export mask.png / preview from labels.json and print the budget")
    b.add_argument("labels")
    b.add_argument("--photo")
    b.add_argument("--out", default=None, help="default: next to labels.json")
    b.add_argument("--cap", type=int, default=CAP_WARN)
    args = ap.parse_args(argv)
    if args.cmd == "draw":
        try:
            size = tuple(int(v) for v in args.size.lower().split("x"))
            assert len(size) == 2 and min(size) > 0
        except (ValueError, AssertionError):
            raise SystemExit(f"--size must look like 1920x1080, got {args.size!r}")
        run_gui(args.photo, args.out, args.labels, args.cap, args.port, args.canvas_width, not args.no_browser, size)
    else:
        run_build(args.labels, args.photo, args.out or os.path.dirname(os.path.abspath(args.labels)), args.cap)


if __name__ == "__main__":
    main()
