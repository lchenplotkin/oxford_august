#!/usr/bin/env python3
"""
Local dev server for the Oxford scansion tool.

Serves static files the same way `python3 -m http.server` would (so
oxford_scansion_gui.html and july_2026/dataset/* keep working unchanged),
and additionally exposes a small JSON file-management API so the page can
list/open/save/delete CSVs under the status folders without the user
manually downloading files and moving them around:

  GET  /api/list                              -> {"folders": {name: [files...]}}
  GET  /api/read?folder=X&name=Y.csv          -> raw CSV text
  POST /api/write  {folder, name, content}    -> writes file (creates folder if needed)
  POST /api/delete {folder, name}             -> removes file (no-op if missing)

`folder` must be one of the known status folders (to_complete, in_progress,
completed_unvetted, gold); `name` must be a plain "<stuff>.csv" filename with
no path separators, to keep this local tool from being tricked into reading
or writing outside those folders.

Usage: python3 server.py [port]   (default port 8721)
"""

import functools
import json
import os
import re
import sys
import threading
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))   # .../july_2026/scansion_tool
STATIC_ROOT = os.path.dirname(SCRIPT_DIR)                 # .../july_2026
FOLDERS_ROOT = SCRIPT_DIR

ALLOWED_FOLDERS = {"to_complete", "in_progress", "completed_unvetted", "gold"}
SAFE_NAME = re.compile(r'^[A-Za-z0-9._\- ]+\.csv$')


def folder_path(folder):
    if folder not in ALLOWED_FOLDERS:
        raise ValueError(f"unknown folder: {folder!r}")
    path = os.path.join(FOLDERS_ROOT, folder)
    os.makedirs(path, exist_ok=True)
    return path


def safe_name(name):
    if not name or not SAFE_NAME.match(name):
        raise ValueError(f"invalid filename: {name!r}")
    return name


class Handler(SimpleHTTPRequestHandler):
    def _send_json(self, obj, status=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_text(self, text, status=200):
        body = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/csv; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self):
        length = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(length) if length else b""
        return json.loads(raw.decode("utf-8")) if raw else {}

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/api/list":
            folders = {}
            for folder in sorted(ALLOWED_FOLDERS):
                path = folder_path(folder)
                folders[folder] = sorted(
                    f for f in os.listdir(path) if f.lower().endswith(".csv")
                )
            self._send_json({"folders": folders})
            return

        if parsed.path == "/api/read":
            qs = parse_qs(parsed.query)
            try:
                folder = qs["folder"][0]
                name = safe_name(qs["name"][0])
                path = os.path.join(folder_path(folder), name)
                if not os.path.isfile(path):
                    self._send_json({"error": "not found"}, status=404)
                    return
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                self._send_text(content)
            except (KeyError, IndexError, ValueError) as e:
                self._send_json({"error": str(e)}, status=400)
            return

        super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        try:
            body = self._read_json_body()

            if parsed.path == "/api/write":
                folder = body["folder"]
                name = safe_name(body["name"])
                content = body["content"]
                path = os.path.join(folder_path(folder), name)
                with open(path, "w", encoding="utf-8", newline="") as f:
                    f.write(content)
                self._send_json({"ok": True})
                return

            if parsed.path == "/api/delete":
                folder = body["folder"]
                name = safe_name(body["name"])
                path = os.path.join(folder_path(folder), name)
                if os.path.isfile(path):
                    os.remove(path)
                self._send_json({"ok": True})
                return

            self._send_json({"error": "not found"}, status=404)
        except (KeyError, ValueError) as e:
            self._send_json({"error": str(e)}, status=400)

    def log_message(self, fmt, *args):
        # Quiet by default; uncomment to debug requests.
        # super().log_message(fmt, *args)
        pass


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8721
    handler = functools.partial(Handler, directory=STATIC_ROOT)
    url = f"http://localhost:{port}/scansion_tool/oxford_scansion_gui.html"
    with ThreadingHTTPServer(("localhost", port), handler) as httpd:
        print(f"Serving {STATIC_ROOT} with file-management API at http://localhost:{port}/")
        print(f"Opening {url}")
        threading.Timer(0.3, lambda: webbrowser.open(url)).start()
        httpd.serve_forever()


if __name__ == "__main__":
    main()
