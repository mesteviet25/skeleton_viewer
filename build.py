"""Build a self-contained 3D skeleton viewer from any per-frame position data.

Sources (repeatable `--swing kind:target[:trial_id]`, bare names still mean Theia):

    build.py "Addie Murphy" "Peyton Hendrix"
    build.py --swing theia:"Addie Murphy" --swing csv:landmarks.csv:1031_2
    build.py --swing c3d:000001_000015_72_180_4_FF_850.c3d          # needs --with ezc3d
    build.py --swing theia:"Addie Murphy" --export-csv dist/

Output: dist/viewer.html -- one offline file, athlete PII, stays on this machine.
"""
import argparse
import base64
import json
from pathlib import Path

import numpy as np
import pymysql

import sources

REPO = Path(__file__).resolve().parent
BRAND = Path.home() / ".claude" / "skills" / "driveline-baseball-design"


def get_secret(name):
    """Resolve a secret from three locations, in priority order."""
    import os
    import re
    import subprocess
    val = os.environ.get(name)
    if val:
        return val
    env_file = Path.home() / ".claude" / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8", errors="ignore").splitlines():
            m = re.match(rf'^\s*{re.escape(name)}\s*=\s*(.+)$', line)
            if m:
                return m.group(1).strip()
    if os.name == "nt":
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command",
             f'[Environment]::GetEnvironmentVariable("{name}", "User")'],
            text=True, stderr=subprocess.DEVNULL).strip()
        if out:
            return out
    return None


def connect(database):
    password = get_secret("BIOMECH_DB_PASS")
    if not password:
        raise SystemExit("BIOMECH_DB_PASS not found in env, ~/.claude/.env, or user registry")
    return pymysql.connect(
        host=get_secret("BIOMECH_DB_HOST") or "db1.drivelinebaseball.com",
        port=int(get_secret("BIOMECH_DB_PORT") or 3306),
        user=get_secret("BIOMECH_DB_USER") or "readonlyuser",
        password=password, database=database, autocommit=True,
    )


def data_uri(path, mime):
    """Brand assets are inlined so the page stays a single offline file."""
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()


def export_csv(sw, out_dir):
    """Write one loaded swing in the generic wide shape the csv loader reads back."""
    import pandas as pd
    n = len(next(iter(sw["points"].values())))
    cols = {"time": np.arange(n) / sw["fs"]}
    for name, arr in sw["points"].items():
        for i, ax in enumerate("xyz"):
            cols[f"{name}_{ax}"] = arr[:, i]
    path = Path(out_dir) / f"{sw['label'].lower().replace(' ', '-')}_positions.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cols).to_csv(path, index=False)
    print(f"exported {path}")
    return path


def prepare(sw, up):
    """Normalise units and orientation, resolve links, and flatten to what the page reads."""
    sw["points"], note = sources.normalise(sw["points"], up)
    names = list(sw["points"])
    if sw["links"] is None:
        sw["links"] = sources.infer_links(sw["points"])
        note += f", {len(sw['links'])} links inferred"
    idx = {n: i for i, n in enumerate(names)}
    links = [[idx[a], idx[b]] for a, b in sw["links"] if a in idx and b in idx]
    bat = [1 if ("bat" in a.lower() and "bat" in b.lower()) else 0
           for a, b in sw["links"] if a in idx and b in idx]

    frames = np.round(np.stack([sw["points"][n] for n in names], axis=1), 4)   # (n, p, 3)
    print(f'{sw["label"]:22s} {len(names):3d} points  {frames.shape[0]:4d} frames  '
          f'{sw["fs"]:.0f} Hz  anchor {sw["anchor"]} ({sw["anchor_kind"]})  [{note}]')
    return {
        "label": sw["label"], "sub": sw["sub"], "stats": sw["stats"],
        "fs": sw["fs"], "n": int(frames.shape[0]), "anchor": sw["anchor"],
        "anchor_kind": sw["anchor_kind"], "events": sw["events"],
        "links": links, "bat": bat,
        "com": idx.get("com"),
        "frames": [[[None if not np.isfinite(v) else v for v in p] for p in f] for f in frames],
    }


def parse_spec(spec):
    """`theia:Name` / `csv:path[:trial]` / `parquet:path[:trial]` / `c3d:path`."""
    kind, _, rest = spec.partition(":")
    if not rest:
        return "theia", spec, None
    kind = kind.lower()
    if kind not in ("theia", "csv", "parquet", "c3d"):
        return "theia", spec, None
    # a Windows path keeps its drive letter: only split a trial id off the tail
    target, sep, trial = rest.rpartition(":")
    if sep and len(target) > 1 and not target.endswith(":"):
        return kind, target, trial
    return kind, rest, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", help="Theia athlete names (shorthand for --swing theia:)")
    ap.add_argument("--swing", action="append", default=[], help="kind:target[:trial_id]")
    ap.add_argument("--db", default="theia_softball_hitting_db")
    ap.add_argument("--fs", type=float, help="sample rate for a table with no `time` column")
    ap.add_argument("--up", default="auto", choices=["auto", "x", "y", "z"])
    ap.add_argument("--export-csv", metavar="DIR", help="also dump each swing as a wide CSV")
    ap.add_argument("--out", default=str(REPO / "dist" / "viewer.html"))
    args = ap.parse_args()

    specs = [("theia", n, None) for n in args.names] + [parse_spec(s) for s in args.swing]
    if not specs:
        raise SystemExit("nothing to draw -- pass an athlete name or --swing")

    conn = connect(args.db) if any(k == "theia" for k, _, _ in specs) else None
    loaded = []
    for kind, target, trial in specs:
        if kind == "theia":
            loaded.append(sources.load_theia(conn.cursor(), target))
        elif kind == "c3d":
            loaded.append(sources.load_c3d(target))
        else:
            loaded.append(sources.load_table(target, trial, args.fs))
    if conn:
        conn.close()

    if args.export_csv:
        for sw in loaded:
            export_csv(sw, args.export_csv)

    swings = [prepare(sw, args.up) for sw in loaded]

    html = (Path(REPO / "template.html").read_text(encoding="utf-8")
            .replace("__DATA__", json.dumps(swings, separators=(",", ":")))
            .replace("__LOGO__", data_uri(
                BRAND / "icons" / "driveline-baseball_logo_full_orange_transparent.png",
                "image/png"))
            .replace("__GOTHAM_MEDIUM__", data_uri(
                BRAND / "fonts" / "gotham_ssv" / "Gotham-Medium.otf", "font/otf"))
            .replace("__GOTHAM_ULTRA__", data_uri(
                BRAND / "fonts" / "gotham_ssv" / "Gotham-Ultra.otf", "font/otf")))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out}  ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
