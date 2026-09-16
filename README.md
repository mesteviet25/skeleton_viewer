# Skeleton Viewer

Animates a 3D skeleton from **any per-frame position data** — Driveline's Theia mocap, the
OpenBiomechanics Project, a wide `landmarks.csv`, a C3D marker file, or anything you export yourself
— several captures side by side in one self-contained offline HTML file. No CDN, no dependencies at
view time.

Two ways in: `serve.py` for browsing all 1,088 OpenBiomechanics trials and loading any of them on
click, `build.py` for baking chosen captures into one offline file.

```
# browse OBP: pick any pitching or hitting session from the table and watch it
PYTHONUTF8=1 ~/miniforge3/python.exe serve.py          # -> http://127.0.0.1:8000
```

```
# Theia, by athlete name (picks each athlete's fastest clean swing)
PYTHONUTF8=1 ~/miniforge3/python.exe build.py "Addie Murphy" "Peyton Hendrix"

# any wide position table; :ID picks one trial when the file holds several
PYTHONUTF8=1 ~/miniforge3/python.exe build.py --swing csv:landmarks.csv:1031_2

# C3D markers (needs ezc3d, so run the whole build under uv)
uv run --no-project --with ezc3d --with pandas --with numpy --with pymysql \
    python build.py --swing c3d:000001_000015_72_180_4_FF_850.c3d

# openbiomechanics trials, baked offline instead of served
PYTHONUTF8=1 ~/miniforge3/python.exe build.py --swing obp:pitching:1031_2 --swing obp:hitting:6_1

# mix sources in one page, and dump a swing in the generic shape
PYTHONUTF8=1 ~/miniforge3/python.exe build.py \
    --swing theia:"Addie Murphy" --swing csv:obp_swing.csv --export-csv dist/

# -> dist/viewer.html   (open it directly in a browser)
```

`build.py` flags: `--db` (default `theia_softball_hitting_db`, use `theia_hitting_db` for
baseball), `--fs` for a table with no `time` column, `--up auto|x|y|z`, `--export-csv DIR`, `--out`,
`--obp` (checkout path). `serve.py` flags: `--obp`, `--port`, `--up`, `--no-open`.

Build deps: `numpy`, `pandas`, `pymysql` (+`ezc3d` for C3D). Theia needs `BIOMECH_DB_*` creds
(env -> `~/.claude/.env` -> registry) and VPN / on-site access to `db1.drivelinebaseball.com`.
Non-Theia sources need neither.

## Input contract

A source is anything that yields **named 3D points over frames**. `sources.py` has three loaders and
they all return the same dict (`label`, `points`, `fs`, `anchor`, `events`, `links`):

| Loader | Accepts |
|---|---|
| `load_theia` | `segment_positions_raw` for an athlete's fastest clean swing. Brings its own segment naming and its own event times. |
| `load_table` | CSV or parquet, one row per frame, `<point>_x/_y/_z` column triples (case-insensitive). A `time` column sets the rate, else pass `--fs`. A key column (`session_swing`, `session_pitch`, `session_trial`, `trial`, `id`) with more than one value must be narrowed with `:ID`. This is the openbiomechanics `full_sig/landmarks.csv` shape. |
| `load_obp` | One OpenBiomechanics trial by `dataset` + id, out of that discipline's `landmarks.csv`. Brings its own segment links, event times, and POI stats. |
| `load_c3d` | Marker trajectories via `ezc3d`; labels and rate come from the file. |

Everything then goes through two source-agnostic steps:

- **`normalise`** — millimetres are detected by extent (`> 10` units tall) and divided out; the
  vertical axis is found as the one with the largest median per-frame extent (a standing athlete is
  taller than they are deep) and moved to axis 2. Both are printed at build time and `--up` overrides
  the guess.
- **`infer_links`** — a source that does not name its own segments gets them inferred: points whose
  separation holds constant across the capture are rigidly connected, and the skeleton is the
  minimum spanning forest over those pairs. Coincident points collapse first (a knee is both the
  thigh's distal end and the shank's proximal end), so the same bone is not drawn twice. Points that
  reach no rigid pair are drawn as dots rather than dropped.

## OpenBiomechanics

`serve.py` reads a local checkout of [openbiomechanics](https://github.com/drivelineresearch/openbiomechanics)
(default `~/research/third_party/openbiomechanics`) and serves three routes: the viewer page,
`/api/index` (every trial, for the picker table) and `/api/trial?dataset=&id=` (one trial's frames,
through the same `build.prepare` a baked page uses — a served trial and a baked one are identical).

Getting the data: the POI and metadata CSVs are in the git tree, the position data is not. Only the
two `landmarks` release assets are needed here (33 MB + 66 MB of the 1.1 GB release):

```python
# checksums come from scripts/release_checksums.sha256 in the checkout; verify before placing
url  = "https://github.com/drivelineresearch/openbiomechanics/releases/download/dataset-v1/"
# pitching_landmarks.zip -> baseball_pitching/data/full_sig/landmarks.zip
# hitting_landmarks.zip  -> baseball_hitting/data/full_sig/landmarks.zip
```

The repo's own `scripts/download_data.sh` does this for the whole release but needs an authenticated
`gh` CLI, which is not installed on this machine; the release assets are public, so plain HTTPS plus
the committed SHA-256 manifest is the same guarantee.

| | Pitching | Hitting |
|---|---|---|
| Trials / athletes | 411 fastballs, 100 | 677 swings, 98 |
| Join key | `session_pitch` | `session_swing` |
| Landmark points | 17 joint centres + CoM | 19 + CoM, incl. the Blast hand and bat sweet spot |
| Events on every row | `pkh_time`, `fp_10_time`, `fp_100_time`, `MER_time`, `BR_time`, `MIR_time` | `fp_10_time`, `fp_100_time`, `contact_time` |
| Tick 0 | ball release | contact |

`landmarks.csv` is one 110 MB / 221 MB table holding every trial, so the first run splits it into one
parquet per trial under `<module>/data/full_sig/by_trial/` (a minute, once). After that a trial loads
in ~40 ms. `full_sig/` is gitignored in OBP, so the cache sits beside the archive it came from.

OBP also ships its own `swing_visualizer/`: hitting only, reads the C3Ds, and does overlay alignment
and size matching this viewer does not. Use it for that; use this one to put OBP next to Theia.

## The viewer

- **Trial table** (served mode only): filter across discipline, trial, session, athlete, level and
  handedness; click a row to load it alongside whatever is already up.
- **Picker** at the top: each capture is a toggle, and the `x` beside it drops the capture entirely.
  Deselect one and the rest go full width; the last visible panel cannot be turned off.
- **One millisecond clock across all panels.** Each panel indexes its own frames at its own rate, so
  a 300 Hz Theia swing and a 360 Hz C3D play together with nothing resampled. The playable range is
  the overlap of whatever is on screen.
- **Tick 0** is ball contact when the source knows it (Theia: `events.bat_contact_time`, which is
  precise — `body_contact_time` can sit ~8 frames off), otherwise the start of the capture. The clock
  label says which.
- **Jump buttons are generated from the events the visible panels actually have.** Data with no
  event tags simply shows the scrubber.
- Fixed 2.0 m vertical scale anchored to the floor, so stature and posture compare across panels.
  Bat goldenrod `#FFA300`, body white, grey connectors, red dot for centre of mass where the source
  has one. Play / scrub / 0.1x-0.25x-1x, drag to orbit, scroll to zoom, catcher / side / top presets.
  A source may set `yaw0` to open on a sensible angle for its own lab axes; the presets stay
  absolute.
- **Driveline branded** off `~/.claude/skills/driveline-baseball-design`: black canvas, `#111` cards,
  `#262626` chrome, Gotham labels over Lato data, zero shadows, 8 px radii. The logo PNG and Gotham
  OTFs are inlined as data URIs so the page keeps its identity offline; Lato is the one network
  reference and degrades to system sans. Athlete names stay Title Case per the brand rule.

Layout lives in `template.html` — edit it without touching Python; `build.render` only fills
`__DATA__`, `__SERVED__`, `__LOGO__` and the two `__GOTHAM_*__` slots. `__SERVED__` is the only
difference between the served page and a baked one: it turns the trial table on.

## Tests

```
PYTHONUTF8=1 ~/miniforge3/python.exe build.py "Addie Murphy" --export-csv dist/
PYTHONUTF8=1 ~/miniforge3/python.exe tests/test_sources.py
```

The fixture is a real Theia swing exported through `--export-csv`, so the round-trip the tests
assert is exactly "a known skeleton sent through the generic path comes back". They check the table
loader, that `normalise` is a no-op on metres/z-up and undoes a mm + y-up mangle, and that
`infer_links` recovers all 16 known Theia segments **by geometry** (not by name — coincident points
collapse, so the forearm may legitimately come back as `rar_dist -> rfa_dist`).

`tests/make_c3d_fixture.py` writes that same CSV out as a mm, y-up C3D, which is how the c3d loader
gets exercised without an openbiomechanics download — OBP's marker files are not in the git tree,
they are distributed as GitHub Release assets.

## Data

`theia_softball_hitting_db.segment_positions_raw` — **this is the skeleton table**. One row per
trial; every column is a gzip-JSON blob `{"0":[x...],"1":[y...],"2":[z...]}` in metres, one value
per frame on the `trials.fs_point` clock. 17 segments x `proxEndPos` / `distEndPos` / `CoMPos`:

`rpv` pelvis, `rta` thorax, `rhe` head, `l/rar` upper arm, `l/rfa` forearm, `l/rha` hand,
`l/rth` thigh, `l/rsk` shank, `l/rft` foot, plus `bat`, `ball0`, and `model_CoMPos`.

The Theia window is trimmed to `load_time - 0.15 s` .. `bat_contact_time + 0.25 s`; the raw capture
is ~760 frames of mostly standing around. Other sources are played whole.

## Traps

- **The schema is `theia_softball_hitting_db`.** Caesar's contract lists it as
  `softball_theia_hitting_db`, which does not exist — that alias raises `Unknown database`. Baseball
  is `theia_hitting_db`.
- **`segment_positions_raw` is not in any contract or skill doc**, and several repos
  (`hitting_ucm`, `wiki/concepts/hitting-variance-structure.md`) state that no joint-centre source
  exists in the Theia schema. That conclusion came from looking only at `landmarks_raw` (bat +
  heels + ball). Full-body positions do exist, here. `rot_mats_raw` carries segment orientations.
- **Theia segments do not touch at the joints.** A hip end sits ~0.10 m off the pelvis segment end
  and a shoulder ~0.15 m off the thorax end, which is why `THEIA_LINKS` exists. Within a limb the
  chain closes exactly (thigh distal == shank proximal, 0.000 m) — that is the check that the
  prox/dist pairing is right.
- **Blobs are gzipped here** (`landmarks_raw` is too, `ang_vels_raw` is not) — sniff the
  `\x1f\x8b` magic rather than assuming.
- **In Git Bash, a `--swing c3d:/c/Users/...` path is not translated.** MSYS rewrites POSIX paths in
  argv to Windows paths only when the argument looks like a path; the `c3d:` prefix defeats that and
  ezc3d then fails with `iostream stream error`. Write the spec with a `C:/...` path.
- **OBP hitting `lhjc` / `rhjc` are HAND joint centres, not hips.** The hips are `left_hip` and
  `right_hip`. Wiring knees to `lhjc` gives a skeleton that looks tilted rather than obviously
  broken — check segment rigidity (knee-to-hip holds 0.445 +- 0.017 m, knee-to-hand swings
  0.977 +- 0.152 m) rather than trusting the name.
- **OBP `time` is rounded to 4 dp**, so deriving the rate from it yields 357 Hz against a real
  360 Hz and drifts the shared clock ~1%. `load_obp` takes 360 Hz as documented and finds event
  frames by nearest `time` value instead.
- **OBP `thorax_prox` is the superior end** (z ~1.50) and `thorax_dist` the inferior (z ~1.24), so
  the hips hang off `thorax_dist` and the shoulders off `thorax_prox` — the opposite of what
  prox/dist suggests if you read it as "proximal to the pelvis".
- **OBP x runs pitcher -> plate**, which is exactly the catcher view, so a hitter's stance
  foreshortens into a single leg at yaw 0. Both OBP loaders open side-on (`yaw0 = pi/2`).
- `poi.bat_speed_max` carries tracking misses (0 / 4 / 311 mph), so Theia swing selection gates
  25-100 mph on top of `needs_review = 0 AND bad_datas_point = 0`.

## PII

Athlete names and mocap, including anything written by `--export-csv`. Local machine only — no
remote, no external service, no Artifact publish. `dist/` is gitignored. OpenBiomechanics is public,
de-identified data and is not covered by that rule, but its checkout stays outside this repo and its
licence forbids commercial use and any use by pro-org or financial-firm staff.
