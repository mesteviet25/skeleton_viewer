"""Build a self-contained 3D skeleton viewer for each athlete's fastest clean swing.

Source: theia_softball_hitting_db.segment_positions_raw -- one row per trial, gzip-JSON blobs
{"0":[x...],"1":[y...],"2":[z...]} in metres, one value per frame at trials.fs_point Hz.
NOTE the schema name: theia_softball_hitting_db, NOT the softball_theia_* alias in Caesar's contract.

Output: dist/viewer.html, offline, no CDN, athlete PII -- stays on this machine.

Usage: PYTHONUTF8=1 ~/miniforge3/python.exe build.py "Addie Murphy" "Peyton Hendrix"
"""
import argparse
import base64
import gzip
import json
import os
import re
import subprocess
from pathlib import Path

import numpy as np
import pymysql

REPO = Path(__file__).resolve().parent
BRAND = Path.home() / ".claude" / "skills" / "driveline-baseball-design"

# prox->dist pairs drawn as lines. Theia segment codes: rpv pelvis, rta thorax, rhe head,
# l/r ar upper arm, fa forearm, ha hand, th thigh, sk shank, ft foot.
SEGMENTS = ["rpv", "rta", "rhe",
            "lar", "lfa", "lha", "rar", "rfa", "rha",
            "lth", "lsk", "lft", "rth", "rsk", "rft"]
BAT = "bat"

# Theia segments do not touch at the joints (a hip sits ~0.10 m off the pelvis end, a shoulder
# ~0.15 m off the thorax end), so the body reads as loose sticks without these connectors.
LINKS = [("rpv", 0, "rta", 0), ("rta", 1, "rhe", 0),
         ("rpv", 1, "lth", 0), ("rpv", 1, "rth", 0),
         ("rta", 1, "lar", 0), ("rta", 1, "rar", 0),
         ("lha", 1, "bat", 0), ("rha", 1, "bat", 0)]

PRE_LOAD_S = 0.15    # window starts this far before load
POST_CONTACT_S = 0.25


def get_secret(name: str) -> str | None:
    """Resolve a secret from three locations, in priority order."""
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


def decode(b64):
    """gzip-JSON blob -> (3, n_frames) float array."""
    raw = base64.b64decode(b64)
    d = json.loads(gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw)
    return np.array([d["0"], d["1"], d["2"]], dtype=float)


def data_uri(path, mime):
    """Brand assets are inlined so the page stays a single offline file."""
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()


def best_swing(cur, name):
    """Highest clean bat_speed_max swing for one athlete, with its session + event times."""
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
    # bat_contact_time is the precise instant; body_contact_time can sit ~8 frames off.
    sw["contact_time"] = (float(sw["bat_contact_time"]) if sw["bat_contact_time"] is not None
                          else sw["body_contact_time"])
    sw["date"] = str(sw["date"])
    return sw


def pull_points(cur, session_trial):
    """{point_name: (3, n)} for every segment endpoint plus the whole-body COM."""
    cols = [f"{s}_{e}" for s in SEGMENTS + [BAT] for e in ("proxEndPos", "distEndPos")]
    cols.append("model_CoMPos")
    sel = ", ".join(f"TO_BASE64({c}) AS {c}" for c in cols)
    cur.execute(f"SELECT {sel} FROM segment_positions_raw WHERE session_trial = %s", (session_trial,))
    row = cur.fetchone()
    if row is None:
        raise SystemExit(f"no segment_positions_raw row for {session_trial}")
    pts = {}
    for name, blob in zip(cols, row):
        if blob is None:
            raise SystemExit(f"{session_trial}: {name} blob is NULL")
        pts[name] = decode(blob)
    n = {a.shape[1] for a in pts.values()}
    if len(n) != 1:
        raise SystemExit(f"{session_trial}: segments disagree on frame count: {sorted(n)}")
    cur.execute("SELECT COUNT(*) FROM joint_angles WHERE session_trial = %s", (session_trial,))
    n_angles = cur.fetchone()[0]
    if n_angles != n.pop():
        raise SystemExit(f"{session_trial}: blob frames != {n_angles} joint_angles rows")
    return pts


def build_swing(cur, name):
    sw = best_swing(cur, name)
    pts = pull_points(cur, sw["session_trial"])
    fs, n = sw["fs"], next(iter(pts.values())).shape[1]

    # sanity: the bat is a rigid 0.5-1.2 m stick, feet live on the floor
    bat_len = np.linalg.norm(pts["bat_distEndPos"] - pts["bat_proxEndPos"], axis=0)
    bat_len = bat_len[np.isfinite(bat_len)]
    assert 0.5 < np.median(bat_len) < 1.2, f"bat length {np.median(bat_len):.3f} m out of range"
    assert bat_len.std() < 0.02, f"bat length not rigid (sd {bat_len.std():.3f} m)"
    foot_z = np.nanmedian(pts["lft_distEndPos"][2])
    assert abs(foot_z) < 0.25, f"lead foot sits at z={foot_z:.2f} m, expected near the floor"

    i0 = max(int(round((sw["load_time"] - PRE_LOAD_S) * fs)), 0)
    i1 = min(int(round((sw["contact_time"] + POST_CONTACT_S) * fs)) + 1, n)
    order = [f"{s}_{e}" for s in SEGMENTS + [BAT] for e in ("proxEndPos", "distEndPos")]
    order.append("model_CoMPos")
    frames = np.stack([pts[k][:, i0:i1] for k in order], axis=1)   # (3, n_pts, n_win)
    frames = np.round(frames.transpose(2, 1, 0), 4)                # (n_win, n_pts, 3)

    return {
        "name": sw["name"], "session_trial": sw["session_trial"], "date": sw["date"],
        "lab": sw["lab"], "hand": sw["handedness"], "fs": fs,
        "bat_speed": round(sw["bat_speed"], 1), "exit_velo": round(sw["exit_velo"], 1),
        "attack_angle": round(sw["attack_angle"], 1),
        "contact": int(round(sw["contact_time"] * fs)) - i0,
        "load": int(round(sw["load_time"] * fs)) - i0,
        "fp": int(round(sw["fp_time"] * fs)) - i0,
        "n": int(frames.shape[0]),
        "frames": [[[None if not np.isfinite(v) else v for v in p] for p in f] for f in frames],
    }


HTML = r"""<!doctype html>
<meta charset="utf-8">
<title>Swing Skeletons</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Lato:wght@300;400;700&display=swap" rel="stylesheet">
<style>
  @font-face { font-family:"Gotham"; font-weight:500;
               src:url(__GOTHAM_MEDIUM__) format("opentype"); }
  @font-face { font-family:"Gotham"; font-weight:700;
               src:url(__GOTHAM_ULTRA__) format("opentype"); }

  :root {
    --gold:#FFA300; --burnt:#CF7F00; --black:#000000; --white:#FFFFFF;
    --card:#111111; --bar:#262626; --muted:#9CA3AF; --red:#EF4444;
  }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--black); color:var(--white);
         font-family:"Lato",sans-serif; font-weight:400; font-size:14px; line-height:1.45; }
  .gotham { font-family:"Gotham","Lato",sans-serif; text-transform:uppercase;
            letter-spacing:.08em; }

  header { display:flex; align-items:center; gap:20px; padding:20px 28px; background:var(--bar); }
  header img { height:26px; }
  header h1 { margin:0; font-size:20px; font-weight:700; }
  header p { margin:2px 0 0; color:var(--muted); font-size:12px; max-width:820px; }
  header b { color:var(--white); font-weight:700; }

  .picker { display:flex; align-items:center; gap:10px; flex-wrap:wrap; padding:16px 28px 0; }
  .views { display:flex; gap:20px; padding:16px 28px 24px; align-items:stretch; }
  .view { flex:1; min-width:0; background:var(--card); border-radius:8px; overflow:hidden; }
  .view[hidden] { display:none; }
  .meta { padding:16px 20px 10px; }
  .meta h2 { margin:0; font-size:17px; font-weight:700; letter-spacing:.04em;
             text-transform:none; }
  .meta .sub { color:var(--muted); font-size:11px; margin-top:3px; }
  .stats { display:flex; gap:26px; margin-top:12px; }
  .stats i { display:block; color:var(--muted); font-style:normal; font-size:10px; }
  .stats b { font-family:"Lato",sans-serif; font-size:22px; font-weight:700; letter-spacing:0; }
  .stats b span { font-size:11px; font-weight:400; color:var(--muted); margin-left:2px; }
  canvas { display:block; width:100%; height:420px; cursor:grab; }
  canvas:active { cursor:grabbing; }

  .bar { display:flex; align-items:center; gap:10px; flex-wrap:wrap;
         padding:14px 28px; background:var(--bar); }
  button { font-family:"Gotham","Lato",sans-serif; text-transform:uppercase; letter-spacing:.08em;
           font-size:11px; font-weight:500; padding:8px 14px; border-radius:8px;
           border:1px solid #262626; background:var(--card); color:var(--white); cursor:pointer; }
  button:hover { border-color:var(--gold); color:var(--gold); }
  button:focus { outline:2px solid var(--gold); outline-offset:1px; }
  button.on { background:var(--gold); border-color:var(--gold); color:var(--black); }
  button.on:hover { background:var(--burnt); border-color:var(--burnt); color:var(--black); }
  #chips { display:flex; gap:8px; }
  .picker button { font-size:12px; padding:9px 16px; text-transform:none;
                   letter-spacing:.02em; }

  #scrub { flex:1; min-width:220px; accent-color:var(--gold); }
  #clock { font-variant-numeric:tabular-nums; min-width:126px; color:var(--muted); font-size:12px; }
  .sep { width:1px; height:22px; background:#111111; }
  .lbl { font-family:"Gotham","Lato",sans-serif; text-transform:uppercase; letter-spacing:.1em;
         font-size:10px; color:var(--muted); }
</style>

<header>
  <img src="__LOGO__" alt="Driveline Baseball">
  <div>
    <h1 class="gotham">Swing Skeletons</h1>
    <p>Theia mocap at __FS__ Hz, drawn from segment end-points. Playback is
       <b>contact-aligned</b> &mdash; tick 0 is ball contact for every hitter, so each frame compares
       the same instant of the swing. Drag to orbit, scroll to zoom.</p>
  </div>
</header>

<div class="picker">
  <span class="lbl">Hitter</span>
  <span id="chips"></span>
</div>
<div class="views" id="views"></div>

<div class="bar">
  <button id="play">Play</button>
  <input type="range" id="scrub">
  <span id="clock"></span>
  <span class="sep"></span>
  <span class="lbl">Speed</span>
  <button class="spd" data-s="0.1">0.1x</button>
  <button class="spd on" data-s="0.25">0.25x</button>
  <button class="spd" data-s="1">1x</button>
  <span class="sep"></span>
  <span class="lbl">Jump</span>
  <button data-jump="load">Load</button>
  <button data-jump="fp">Foot plant</button>
  <button data-jump="contact">Contact</button>
  <span class="sep"></span>
  <span class="lbl">View</span>
  <button data-view="catcher">Catcher</button>
  <button data-view="side">Side</button>
  <button data-view="top">Top</button>
</div>

<script>
const SWINGS = __DATA__;
const NSEG = __NSEG__, BAT_SEG = NSEG - 1, COM = NSEG * 2, LINKS = __LINKS__;

// tick 0 = contact; the shared range is the overlap of every swing on screen
const tMin = -Math.min(...SWINGS.map(s => s.contact));
const tMax =  Math.min(...SWINGS.map(s => s.n - 1 - s.contact));

const views = SWINGS.map(s => {
  const el = document.createElement('div');
  el.className = 'view';
  el.innerHTML = '<div class="meta"><h2 class="gotham">' + s.name + '</h2>'
    + '<div class="sub gotham">' + s.hand + 'HH &middot; ' + s.date + ' &middot; ' + s.lab
    + ' &middot; ' + s.session_trial + '</div>'
    + '<div class="stats">'
    + '<div><i class="gotham">Bat speed</i><b>' + s.bat_speed + '<span>mph</span></b></div>'
    + '<div><i class="gotham">Exit velo</i><b>' + s.exit_velo + '<span>mph</span></b></div>'
    + '<div><i class="gotham">Attack angle</i><b>' + s.attack_angle + '<span>deg</span></b></div>'
    + '</div></div><canvas></canvas>';
  document.getElementById('views').appendChild(el);

  // camera orbits the pelvis (averaged over the swing) but is anchored to the FLOOR in z, so
  // every hitter shares one ground line and stature is comparable between panels
  const c = [0, 0, 0];
  let nc = 0;
  for (const f of s.frames) {
    if (f[0][0] === null) continue;
    c[0] += f[0][0]; c[1] += f[0][1];
    nc++;
  }
  const cv = el.querySelector('canvas');
  return { s, el, cv, ctx: cv.getContext('2d'), centre: c.map(v => v / nc),
           yaw: 0, pitch: 0.12, zoom: 1, drag: null };
});

// axis 0 = pitcher axis, 1 = lateral, 2 = vertical
function project(v, k) {
  const x = v[0] - k.centre[0], y = v[1] - k.centre[1], z = v[2] - k.centre[2];
  const cy = Math.cos(k.yaw), sy = Math.sin(k.yaw);
  const right = -sy * x + cy * y, depth = cy * x + sy * y;
  return [right, z * Math.cos(k.pitch) - depth * Math.sin(k.pitch)];
}

function draw(k, tick) {
  if (k.el.hidden) return;
  const ctx = k.ctx, cv = k.cv;
  const dpr = window.devicePixelRatio || 1;
  const w = cv.clientWidth, h = cv.clientHeight;
  if (cv.width !== Math.round(w * dpr)) { cv.width = w * dpr; cv.height = h * dpr; }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);

  // fixed 2.0 m vertical field so every hitter is drawn at the same scale
  const scale = k.zoom * h / 2.0;
  const px = v => { const p = project(v, k); return [w / 2 + p[0] * scale, h * 0.88 - p[1] * scale]; };
  const line = (a, b) => { const p = px(a), q = px(b);
    ctx.beginPath(); ctx.moveTo(p[0], p[1]); ctx.lineTo(q[0], q[1]); ctx.stroke(); };

  // floor grid on z = 0, centred under the hitter
  ctx.strokeStyle = '#262626'; ctx.lineWidth = 1;
  const gx = k.centre[0], gy = k.centre[1];
  for (let g = -1.5; g <= 1.5001; g += 0.5) {
    line([gx + g, gy - 1.5, 0], [gx + g, gy + 1.5, 0]);
    line([gx - 1.5, gy + g, 0], [gx + 1.5, gy + g, 0]);
  }

  const f = k.s.frames[tick + k.s.contact];
  if (!f) return;
  ctx.lineCap = 'round';
  ctx.strokeStyle = '#9CA3AF'; ctx.lineWidth = 2;
  for (const pair of LINKS) {
    if (f[pair[0]][0] === null || f[pair[1]][0] === null) continue;
    line(f[pair[0]], f[pair[1]]);
  }
  for (let i = 0; i < NSEG; i++) {
    const a = f[i * 2], b = f[i * 2 + 1];
    if (a[0] === null || b[0] === null) continue;
    ctx.strokeStyle = i === BAT_SEG ? '#FFA300' : '#FFFFFF';
    ctx.lineWidth = i === BAT_SEG ? 5 : 3.5;
    line(a, b);
  }
  const com = f[COM];
  if (com[0] !== null) {
    const p = px(com);
    ctx.fillStyle = '#EF4444';
    ctx.beginPath(); ctx.arc(p[0], p[1], 4, 0, 7); ctx.fill();
  }
}

let tick = 0, playing = false, speed = 0.25, last = 0;
const scrub = document.getElementById('scrub');
const playBtn = document.getElementById('play');
scrub.min = tMin; scrub.max = tMax; scrub.value = 0;

function render() {
  for (const k of views) draw(k, tick);
  scrub.value = tick;
  const ms = (tick / SWINGS[0].fs * 1000).toFixed(0);
  document.getElementById('clock').textContent = (tick >= 0 ? '+' : '') + ms + ' ms vs contact';
}

function stop() { playing = false; playBtn.textContent = 'Play'; playBtn.classList.remove('on'); }

// hitter picker: toggle panels on and off, but never leave the page empty
const chips = document.getElementById('chips');
views.forEach((k, i) => {
  const b = document.createElement('button');
  b.textContent = k.s.name;
  b.className = 'on';
  b.onclick = () => {
    const shown = views.filter(v => !v.el.hidden);
    if (!k.el.hidden && shown.length === 1) return;
    k.el.hidden = !k.el.hidden;
    b.classList.toggle('on', !k.el.hidden);
    render();
  };
  chips.appendChild(b);
});

function loop(ts) {
  if (playing) {
    if (!last) last = ts;
    const step = (ts - last) / 1000 * SWINGS[0].fs * speed;
    if (step >= 1) {
      last = ts;
      tick += Math.floor(step);
      if (tick > tMax) tick = tMin;
      render();
    }
  }
  requestAnimationFrame(loop);
}
requestAnimationFrame(loop);

playBtn.onclick = () => {
  playing = !playing; last = 0;
  playBtn.textContent = playing ? 'Pause' : 'Play';
  playBtn.classList.toggle('on', playing);
};
scrub.oninput = e => { tick = +e.target.value; render(); };
document.querySelectorAll('.spd').forEach(b => b.onclick = () => {
  speed = +b.dataset.s; last = 0;
  document.querySelectorAll('.spd').forEach(o => o.classList.toggle('on', o === b));
});
document.querySelectorAll('[data-jump]').forEach(b => b.onclick = () => {
  const key = b.dataset.jump;
  const shown = views.filter(v => !v.el.hidden);
  const t = key === 'contact' ? 0
    : Math.round(shown.reduce((a, v) => a + v.s[key] - v.s.contact, 0) / shown.length);
  tick = Math.max(tMin, Math.min(tMax, t));
  stop(); render();
});
const PRESETS = { catcher: [0, 0.12], side: [Math.PI / 2, 0.12], top: [0, 1.45] };
document.querySelectorAll('[data-view]').forEach(b => b.onclick = () => {
  const p = PRESETS[b.dataset.view];
  for (const k of views) { k.yaw = p[0]; k.pitch = p[1]; k.zoom = 1; }
  render();
});

for (const k of views) {
  k.cv.onmousedown = e => { k.drag = [e.clientX, e.clientY]; };
  window.addEventListener('mouseup', () => { k.drag = null; });
  window.addEventListener('mousemove', e => {
    if (!k.drag) return;
    k.yaw += (e.clientX - k.drag[0]) * 0.01;
    k.pitch = Math.max(-1.4, Math.min(1.5, k.pitch + (e.clientY - k.drag[1]) * 0.006));
    k.drag = [e.clientX, e.clientY];
    render();
  });
  k.cv.onwheel = e => {
    e.preventDefault();
    k.zoom = Math.max(0.4, Math.min(4, k.zoom * (e.deltaY < 0 ? 1.1 : 0.91)));
    render();
  };
}
window.addEventListener('resize', render);
render();
</script>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    ap.add_argument("--db", default="theia_softball_hitting_db")
    ap.add_argument("--out", default=str(REPO / "dist" / "viewer.html"))
    args = ap.parse_args()

    conn = connect(args.db)
    cur = conn.cursor()
    swings = [build_swing(cur, n) for n in args.names]
    conn.close()

    fs = {s["fs"] for s in swings}
    if len(fs) != 1:
        raise SystemExit(f"swings captured at different rates: {sorted(fs)} -- alignment would lie")

    for s in swings:
        print(f'{s["name"]:20s} {s["session_trial"]:8s} {s["bat_speed"]:5.1f} mph  '
              f'{s["n"]:4d} frames  contact@{s["contact"]}  load@{s["load"]}  fp@{s["fp"]}')

    order = SEGMENTS + [BAT]
    links = [[order.index(a) * 2 + ai, order.index(b) * 2 + bi] for a, ai, b, bi in LINKS]
    html = (HTML.replace("__DATA__", json.dumps(swings, separators=(",", ":")))
                .replace("__LINKS__", json.dumps(links))
                .replace("__NSEG__", str(len(SEGMENTS) + 1))
                .replace("__FS__", f"{swings[0]['fs']:.0f}")
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
