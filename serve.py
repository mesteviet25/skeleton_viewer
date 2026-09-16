"""Browse the OpenBiomechanics Project in the skeleton viewer.

    PYTHONUTF8=1 ~/miniforge3/python.exe serve.py            # then open http://127.0.0.1:8000

Three routes: the viewer page, the trial index the picker lists, and one trial's frames. The page is
the same `template.html` `build.py` bakes, so a served trial and a baked one render identically.
"""
import argparse
import json
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import build
import sources

OBP_ROOT = Path.home() / "research" / "third_party" / "openbiomechanics"


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        url = urlparse(self.path)
        query = parse_qs(url.query)
        if url.path == "/":
            self.send(200, "text/html; charset=utf-8", build.render([], served=True))
        elif url.path == "/api/index":
            self.send(200, "application/json", json.dumps(self.server.index))
        elif url.path == "/api/trial":
            dataset, trial = query["dataset"][0], query["id"][0]
            if dataset not in sources.OBP:
                return self.send(404, "text/plain", f"unknown dataset {dataset!r}")
            sw = sources.load_obp(self.server.obp_root, dataset, trial)
            self.send(200, "application/json",
                      json.dumps(build.prepare(sw, self.server.up), separators=(",", ":")))
        else:
            self.send(404, "text/plain", "not found")

    def send(self, code, ctype, body):
        body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        print(f"  {fmt % args}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obp", default=str(OBP_ROOT), help="openbiomechanics checkout")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--up", default="auto", choices=["auto", "x", "y", "z"])
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()

    # both splits up front: the first one costs a minute, and it should not land on a page request
    for dataset in sources.OBP:
        sources.obp_trials(args.obp, dataset)

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.obp_root, server.up = args.obp, args.up
    server.index = sources.obp_index(args.obp)
    url = f"http://127.0.0.1:{args.port}"
    print(f"{len(server.index)} OBP trials indexed -- {url}")
    if not args.no_open:
        webbrowser.open(url)
    server.serve_forever()


if __name__ == "__main__":
    main()
