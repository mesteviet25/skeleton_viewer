"""Loaders that turn any per-frame 3D position data into one shape the viewer can play.

Every loader returns:

    {"label": str,            # panel heading
     "sub": str,              # small grey line under the heading
     "stats": [(name, value, unit), ...],
     "points": {name: (n, 3) float array},
     "fs": float,             # Hz
     "anchor": int,           # frame that becomes tick 0
     "anchor_kind": str,      # "contact" or "start" -- only changes the clock label
     "events": {name: frame},
     "links": [(name, name), ...] or None}   # None -> infer_links() works it out

Sources so far: Theia (`segment_positions_raw`), a wide CSV/parquet table, and C3D.
"""
import base64
import gzip
import json
import re
from pathlib import Path

import numpy as np

# Theia segment codes: rpv pelvis, rta thorax, rhe head, l/r ar upper arm, fa forearm, ha hand,
# th thigh, sk shank, ft foot. Each is a prox->dist pair.
THEIA_SEGMENTS = ["rpv", "rta", "rhe",
                  "lar", "lfa", "lha", "rar", "rfa", "rha",
                  "lth", "lsk", "lft", "rth", "rsk", "rft", "bat"]

# Theia segments do not touch at the joints (a hip sits ~0.10 m off the pelvis end, a shoulder
# ~0.15 m off the thorax end), so the body reads as loose sticks without these connectors.
THEIA_LINKS = [("rpv_prox", "rta_prox"), ("rta_dist", "rhe_prox"),
               ("rpv_dist", "lth_prox"), ("rpv_dist", "rth_prox"),
               ("rta_dist", "lar_prox"), ("rta_dist", "rar_prox"),
               ("lha_dist", "bat_prox"), ("rha_dist", "bat_prox")]

PRE_LOAD_S = 0.15      # Theia window starts this far before load
POST_CONTACT_S = 0.25

RIGID_CV = 0.02        # a pair whose separation varies less than this fraction is one segment
MM_THRESHOLD = 10.0    # a human is ~2 units tall in metres and ~2000 in millimetres


# --------------------------------------------------------------------------- shared geometry

def normalise(points, up="auto"):
    """Put any source into the viewer's convention: metres, axis 2 vertical.

    Returns (points, note) where note describes what was done, so a wrong guess is visible in the
    build log rather than silently rendering a hitter lying on their side.
    """
    stack = np.stack(list(points.values()))                    # (p, n, 3)
    span = np.nanmax(stack, axis=(0, 1)) - np.nanmin(stack, axis=(0, 1))
    notes = []

    if np.nanmax(span) > MM_THRESHOLD:
        points = {k: v / 1000.0 for k, v in points.items()}
        stack = stack / 1000.0
        notes.append("mm -> m")

    if up == "auto":
        # a standing athlete is taller than they are deep, measured per frame so a swing's
        # horizontal bat travel across the whole capture cannot outvote their height
        extent = np.nanmax(stack, axis=0) - np.nanmin(stack, axis=0)   # (n, 3)
        axis = int(np.argmax(np.nanmedian(extent, axis=0)))
    else:
        axis = "xyz".index(up)
    if axis != 2:
        order = [i for i in (0, 1, 2) if i != axis] + [axis]
        points = {k: v[:, order] for k, v in points.items()}
        notes.append(f"up axis {'xyz'[axis]} -> z")

    return points, ", ".join(notes) or "already metres, z up"


def infer_links(points):
    """Connect point pairs whose separation stays near-constant -- that is what a segment is.

    Two passes. First, points that sit on top of each other for the whole capture (a knee is both
    the thigh's distal end and the shank's proximal end) collapse to one node, so the same bone is
    not offered under several names. Then every rigid pair between distinct nodes is an edge
    weighted by its length, and the minimum spanning forest over those edges is the skeleton --
    the shortest set of rigid connections that still holds the body together. Without the forest
    step you also draw every incidentally-rigid pair, e.g. shoulder-to-opposite-hip.
    """
    names = list(points)
    dist = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            d = np.linalg.norm(points[a] - points[b], axis=1)
            d = d[np.isfinite(d)]
            dist[(a, b)] = (float(d.mean()), float(d.std())) if d.size else None

    parent = {n: n for n in names}

    def find(n):
        while parent[n] != n:
            parent[n] = parent[parent[n]]
            n = parent[n]
        return n

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        parent[ra] = rb
        return True

    for (a, b), md in dist.items():                      # pass 1: collapse coincident points
        if md and md[0] < 1e-3:
            union(a, b)

    edges = sorted((md[0], a, b) for (a, b), md in dist.items()
                   if md and md[0] >= 1e-3 and md[1] / md[0] < RIGID_CV)
    return [(a, b) for _, a, b in edges if union(a, b)]   # pass 2: minimum spanning forest


# --------------------------------------------------------------------------- Theia

def _decode(b64):
    """gzip-JSON blob -> (3, n_frames) float array."""
    raw = base64.b64decode(b64)
    d = json.loads(gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw)
    return np.array([d["0"], d["1"], d["2"]], dtype=float)


def load_theia(cur, name):
    """Highest clean bat_speed_max swing for one athlete, trimmed to load..contact."""
    cur.execute("""
        SELECT u.name, t.session_trial, t.handedness, t.fs_point, s.date, s.lab,
               p.bat_speed_max, p.exit_velo_mph, p.attack_angle_contact,
               e.load_time, e.fp_time, e.bat_contact_time, e.body_contact_time
        FROM trials t
        JOIN sessions s ON s.session = t.session
        JOIN users u    ON u.user = s.user
        JOIN poi p      ON p.session_trial = t.session_trial
        JOIN events e   ON e.session_trial = t.session_trial
        WHERE u.name = %s AND t.needs_review = 0 AND t.bad_datas_point = 0
          AND p.bat_speed_max BETWEEN 25 AND 100
        ORDER BY p.bat_speed_max DESC LIMIT 1
    """, (name,))
    row = cur.fetchone()
    if row is None:
        raise SystemExit(f"no clean swing with a usable bat speed for {name!r}")
    keys = ["name", "session_trial", "handedness", "fs", "date", "lab", "bat_speed", "exit_velo",
            "attack_angle", "load_time", "fp_time", "bat_contact_time", "body_contact_time"]
    sw = dict(zip(keys, row))
    for k in ("fs", "bat_speed", "exit_velo", "attack_angle", "load_time", "fp_time",
              "body_contact_time"):
        sw[k] = float(sw[k])
    # bat_contact_time is the precise instant; body_contact_time can sit ~8 frames off
    contact = (float(sw["bat_contact_time"]) if sw["bat_contact_time"] is not None
               else sw["body_contact_time"])
    fs = sw["fs"]

    cols = [f"{s}_{e}" for s in THEIA_SEGMENTS for e in ("proxEndPos", "distEndPos")]
    cols.append("model_CoMPos")
    sel = ", ".join(f"TO_BASE64({c}) AS {c}" for c in cols)
    cur.execute(f"SELECT {sel} FROM segment_positions_raw WHERE session_trial = %s",
                (sw["session_trial"],))
    row = cur.fetchone()
    if row is None:
        raise SystemExit(f"no segment_positions_raw row for {sw['session_trial']}")
    raw = {}
    for col, blob in zip(cols, row):
        if blob is None:
            raise SystemExit(f"{sw['session_trial']}: {col} blob is NULL")
        raw[col.replace("proxEndPos", "prox").replace("distEndPos", "dist")
               .replace("model_CoMPos", "com")] = _decode(blob).T
    n = {a.shape[0] for a in raw.values()}
    if len(n) != 1:
        raise SystemExit(f"{sw['session_trial']}: segments disagree on frame count: {sorted(n)}")
    n = n.pop()
    cur.execute("SELECT COUNT(*) FROM joint_angles WHERE session_trial = %s", (sw["session_trial"],))
    n_angles = cur.fetchone()[0]
    if n_angles != n:
        raise SystemExit(f"{sw['session_trial']}: blob frames != {n_angles} joint_angles rows")

    # sanity: the bat is a rigid 0.5-1.2 m stick, feet live on the floor
    bat_len = np.linalg.norm(raw["bat_dist"] - raw["bat_prox"], axis=1)
    bat_len = bat_len[np.isfinite(bat_len)]
    assert 0.5 < np.median(bat_len) < 1.2, f"bat length {np.median(bat_len):.3f} m out of range"
    assert bat_len.std() < 0.02, f"bat length not rigid (sd {bat_len.std():.3f} m)"
    foot_z = np.nanmedian(raw["lft_dist"][:, 2])
    assert abs(foot_z) < 0.25, f"lead foot sits at z={foot_z:.2f} m, expected near the floor"

    i0 = max(int(round((sw["load_time"] - PRE_LOAD_S) * fs)), 0)
    i1 = min(int(round((contact + POST_CONTACT_S) * fs)) + 1, n)
    points = {k: v[i0:i1] for k, v in raw.items()}
    events = {"load": int(round(sw["load_time"] * fs)) - i0,
              "foot plant": int(round(sw["fp_time"] * fs)) - i0,
              "contact": int(round(contact * fs)) - i0}

    links = [(f"{s}_prox", f"{s}_dist") for s in THEIA_SEGMENTS] + THEIA_LINKS
    return {
        "label": sw["name"],
        "sub": (f"{sw['handedness']}HH &middot; {sw['date']} &middot; {sw['lab']} &middot; "
                f"{sw['session_trial']} &middot; Theia {fs:.0f} Hz"),
        "stats": [("Bat speed", round(sw["bat_speed"], 1), "mph"),
                  ("Exit velo", round(sw["exit_velo"], 1), "mph"),
                  ("Attack angle", round(sw["attack_angle"], 1), "deg")],
        "points": points, "fs": fs, "anchor": events["contact"], "anchor_kind": "contact",
        "events": events, "links": links,
    }


# --------------------------------------------------------------------------- generic table

COORD = re.compile(r"^(?P<point>.+)_(?P<axis>[xyz])$", re.I)


def load_table(path, trial_id=None, fs=None, key=None):
    """Wide CSV/parquet: one row per frame, `<point>_x/_y/_z` columns.

    This is the openbiomechanics `full_sig/landmarks.csv` shape (joint centre positions keyed on
    `session_swing` / `session_pitch` plus `time`), and the shape `build.py --export-csv` writes.
    """
    import pandas as pd

    df = pd.read_parquet(path) if str(path).endswith(".parquet") else pd.read_csv(path)

    if key is None:
        key = next((c for c in df.columns if c.lower() in
                    ("session_swing", "session_pitch", "session_trial", "trial", "id")), None)
    if key is not None:
        ids = df[key].unique()
        if trial_id is not None:
            df = df[df[key].astype(str) == str(trial_id)]
            if df.empty:
                raise SystemExit(f"{path}: no rows with {key} == {trial_id!r}")
        elif len(ids) > 1:
            raise SystemExit(f"{path}: {len(ids)} trials in {key} -- name one, e.g. "
                             f"csv:{path}:{ids[0]}")
        trial_id = str(df[key].iloc[0])

    time_col = next((c for c in df.columns if c.lower() == "time"), None)
    if fs is None:
        if time_col is None:
            raise SystemExit(f"{path}: no `time` column -- pass --fs")
        fs = 1.0 / float(np.median(np.diff(df[time_col].to_numpy(dtype=float))))

    points = {}
    for col in df.columns:
        m = COORD.match(col)
        if m:
            points.setdefault(m["point"], {})[m["axis"].lower()] = df[col].to_numpy(dtype=float)
    points = {k: np.stack([v["x"], v["y"], v["z"]], axis=1)
              for k, v in points.items() if len(v) == 3}
    if not points:
        raise SystemExit(f"{path}: no `<point>_x/_y/_z` column triples found")

    return {"label": trial_id or Path(path).stem,
            "sub": f"{Path(path).name} &middot; {fs:.0f} Hz",
            "stats": [], "points": points, "fs": float(fs), "anchor": 0,
            "anchor_kind": "start", "events": {}, "links": None}


# --------------------------------------------------------------------------- c3d

def load_c3d(path):
    """Raw marker trajectories, e.g. openbiomechanics `data/c3d/`. Needs `uv run --with ezc3d`."""
    import ezc3d

    c = ezc3d.c3d(str(path))
    arr = c["data"]["points"]                      # (4, n_markers, n_frames), homogeneous
    labels = [s.strip() for s in c["parameters"]["POINT"]["LABELS"]["value"]]
    fs = float(c["parameters"]["POINT"]["RATE"]["value"][0])
    points = {lab: arr[:3, i, :].T for i, lab in enumerate(labels)}
    return {"label": Path(path).stem,
            "sub": f"{Path(path).name} &middot; C3D &middot; {fs:.0f} Hz",
            "stats": [], "points": points, "fs": fs, "anchor": 0,
            "anchor_kind": "start", "events": {}, "links": None}
