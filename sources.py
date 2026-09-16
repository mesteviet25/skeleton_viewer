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
     "links": [(name, name), ...] or None,   # None -> infer_links() works it out
     "yaw0": float}           # starting camera yaw; the view buttons stay absolute

Sources so far: Theia (`segment_positions_raw`), a wide CSV/parquet table, and C3D.
"""
import base64
import gzip
import json
import re
import zipfile
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


# --------------------------------------------------------------------------- openbiomechanics

# The public OBP release (github.com/drivelineresearch/openbiomechanics). `landmarks.csv` is one
# joint-centre table per discipline holding every trial, already metres and z up, with each trial's
# event times repeated on every row. Frames are split to one parquet per trial on first use because
# the hitting CSV is 221 MB and a viewer request must not read it.
OBP = {
    "pitching": {
        "module": "baseball_pitching", "key": "session_pitch", "fs": 360.0,
        "events": {"peak knee height": "pkh_time", "foot plant": "fp_100_time",
                   "MER": "MER_time", "release": "BR_time", "MIR": "MIR_time"},
        "anchor": "release",
        "stats": [("Pitch speed", "pitch_speed_mph", "mph")],
        "links": [("rear_ankle_jc", "rear_knee_jc"), ("rear_knee_jc", "rear_hip"),
                  ("lead_ankle_jc", "lead_knee_jc"), ("lead_knee_jc", "lead_hip"),
                  ("rear_hip", "lead_hip"), ("rear_hip", "thorax_dist"), ("lead_hip", "thorax_dist"),
                  ("thorax_dist", "thorax_prox"),
                  ("thorax_prox", "shoulder_jc"), ("thorax_prox", "glove_shoulder_jc"),
                  ("shoulder_jc", "elbow_jc"), ("elbow_jc", "wrist_jc"), ("wrist_jc", "hand_jc"),
                  ("glove_shoulder_jc", "glove_elbow_jc"), ("glove_elbow_jc", "glove_wrist_jc"),
                  ("glove_wrist_jc", "glove_hand_jc")],
    },
    "hitting": {
        "module": "baseball_hitting", "key": "session_swing", "fs": 360.0,
        "events": {"foot plant": "fp_100_time", "contact": "contact_time"},
        "anchor": "contact",
        "stats": [("Exit velo", "exit_velo_mph_x", "mph"),
                  ("Bat speed", "bat_speed_mph_contact_x", "mph"),
                  ("Attack angle", "attack_angle_contact_x", "deg")],
        # l/r hjc are HAND joint centres, not hips -- the hips are `left_hip` / `right_hip`
        "links": [("lajc", "lkjc"), ("lkjc", "left_hip"), ("rajc", "rkjc"), ("rkjc", "right_hip"),
                  ("left_hip", "right_hip"),
                  ("left_hip", "thorax_dist"), ("right_hip", "thorax_dist"),
                  ("thorax_dist", "thorax_prox"),
                  ("thorax_prox", "lsjc"), ("thorax_prox", "rsjc"),
                  ("lsjc", "lejc"), ("lejc", "lwjc"), ("lwjc", "lhjc"),
                  ("rsjc", "rejc"), ("rejc", "rwjc"), ("rwjc", "rhjc"),
                  ("lhjc", "bat_prox"), ("rhjc", "bat_prox"), ("bat_prox", "bat_dist")],
    },
}

# `centerofmass` gets the red CoM dot the viewer draws, and the Blast hand / sweet spot pair is the
# bat, named the way the Theia loader names it so both sources render it in goldenrod.
OBP_RENAME = {"centerofmass": "com", "blast_hand": "bat_prox", "sweet_spot": "bat_dist"}


def obp_trials(root, dataset):
    """Split `landmarks.csv` into one parquet per trial, once. Returns the directory."""
    import pandas as pd

    spec = OBP[dataset]
    full_sig = Path(root) / spec["module"] / "data" / "full_sig"
    out = full_sig / "by_trial"
    if out.is_dir():
        return out
    zip_path = full_sig / "landmarks.zip"
    if not zip_path.exists():
        raise SystemExit(f"{zip_path} missing -- run the OBP downloader for {dataset}")
    tmp = out.with_name("by_trial.part")
    tmp.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z, z.open("landmarks.csv") as f:
        df = pd.read_csv(f)
    for trial, g in df.groupby(spec["key"], sort=False):
        g.to_parquet(tmp / f"{trial}.parquet", index=False)
    tmp.rename(out)
    print(f"split {len(df)} {dataset} rows into {len(list(out.iterdir()))} trials -> {out}")
    return out


def obp_index(root):
    """One row per OBP trial, for the picker: ids, athlete context, and the headline metric."""
    import pandas as pd

    rows = []
    for dataset, spec in OBP.items():
        data = Path(root) / spec["module"] / "data"
        m = pd.read_csv(data / "metadata.csv")
        common = {"dataset": dataset, "trial": m[spec["key"]].astype(str),
                  "session": m["session"], "user": m["user"]}
        if dataset == "pitching":
            # handedness is the one picker field that lives in the POI table, not metadata
            m = m.merge(pd.read_csv(data / "poi" / "poi_metrics.csv",
                                    usecols=["session_pitch", "p_throws"]),
                        on="session_pitch", how="left")
            out = pd.DataFrame({**common, "level": m["playing_level"], "age": m["age_yrs"],
                                "side": m["p_throws"], "speed": m["pitch_speed_mph"],
                                "height_m": m["session_height_m"], "mass_kg": m["session_mass_kg"]})
        else:
            out = pd.DataFrame({**common, "level": m["highest_playing_level"],
                                "age": m["athlete_age"], "side": m["hitter_side"],
                                "speed": m["exit_velo_mph_x"],
                                "height_m": m["session_height_in"] * 0.0254,
                                "mass_kg": m["session_mass_lbs"] * 0.45359237})
        out = out.astype(object)                       # NaN is not JSON; None survives only as object
        rows += out.where(pd.notna(out), None).to_dict("records")
    return rows


def load_obp(root, dataset, trial_id):
    """One OpenBiomechanics trial: joint centres plus the event times stored beside them."""
    import pandas as pd

    spec = OBP[dataset]
    path = obp_trials(root, dataset) / f"{trial_id}.parquet"
    if not path.exists():
        raise SystemExit(f"no {dataset} trial {trial_id!r} in {path.parent}")
    df = pd.read_parquet(path)
    t = df["time"].to_numpy(dtype=float)
    fs = spec["fs"]

    points = {}
    for col in df.columns:
        m = COORD.match(col)
        if m:
            points.setdefault(m["point"], {})[m["axis"].lower()] = df[col].to_numpy(dtype=float)
    points = {OBP_RENAME.get(k, k): np.stack([v["x"], v["y"], v["z"]], axis=1)
              for k, v in points.items() if len(v) == 3}

    # event columns repeat one time per row; they are seconds on the same clock as `time`
    events = {}
    for name, col in spec["events"].items():
        val = df[col].iloc[0]
        if pd.notna(val):
            events[name] = int(np.argmin(np.abs(t - float(val))))
    anchor = events.get(spec["anchor"], 0)

    meta = pd.read_csv(Path(root) / spec["module"] / "data" / "metadata.csv")
    meta = meta[meta[spec["key"]].astype(str) == str(trial_id)]
    poi = pd.read_csv(Path(root) / spec["module"] / "data" / "poi" / "poi_metrics.csv")
    poi = poi[poi[spec["key"]].astype(str) == str(trial_id)]
    row = {**meta.iloc[0].to_dict(), **poi.iloc[0].to_dict()}
    stats = [(label, round(float(row[col]), 1), unit) for label, col, unit in spec["stats"]
             if pd.notna(row[col])]
    side = row.get("p_throws") or row.get("hitter_side") or "?"
    level = row.get("playing_level") or row.get("highest_playing_level") or "?"

    return {
        "label": f"{dataset[:4].upper()} {trial_id}",
        "sub": f"{side} &middot; {level} &middot; OBP {dataset} &middot; {fs:.0f} Hz",
        "stats": stats, "points": points, "fs": fs,
        "anchor": anchor, "anchor_kind": spec["anchor"] if anchor else "start",
        "events": events, "links": spec["links"],
        # OBP x runs pitcher -> plate, so yaw 0 (catcher) foreshortens the stride: open side-on
        "yaw0": np.pi / 2,
    }
