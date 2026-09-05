# Skeleton Viewer

Animates a hitter's **full 3D mocap skeleton** for their fastest clean swing, two athletes side by
side, contact-aligned. Output is one self-contained offline HTML file — no CDN, no server, no
dependencies at view time.

```
PYTHONUTF8=1 ~/miniforge3/python.exe build.py "Addie Murphy" "Peyton Hendrix"
# -> dist/viewer.html   (open it directly in a browser)

# baseball hitters instead of softball:
PYTHONUTF8=1 ~/miniforge3/python.exe build.py --db theia_hitting_db "Some Hitter" "Other Hitter"
```

Build deps: `numpy`, `pymysql`, DB creds `BIOMECH_DB_*` (env -> `~/.claude/.env` -> registry).
Requires VPN / on-site access to `db1.drivelinebaseball.com`.

## The viewer

- **Hitter picker** at the top: each name is a toggle. Deselect one and the remaining hitter goes
  full width; the last visible hitter cannot be turned off. `build.py` takes any number of names, so
  the picker scales past two.
- One panel per hitter, drawn at a **fixed 2.0 m vertical scale** and anchored to the floor
  (z = 0), so stature and posture are directly comparable between panels.
- **Playback is contact-aligned**: tick 0 is `events.bat_contact_time` for both hitters, so every
  frame compares the same instant of the swing even though their swings run different lengths.
  The tick range is the overlap of the two swings.
- Bat is goldenrod `#FFA300`, body is white, grey connectors bridge the joints, red dot is
  `model_CoMPos`. Play / scrub / 0.1x-0.25x-1x, jump to load / foot plant / contact,
  catcher / side / top camera presets, drag to orbit, scroll to zoom.
- **Driveline branded** off `~/.claude/skills/driveline-baseball-design`: black canvas, `#111`
  cards, `#262626` header and control bar, goldenrod accent, Gotham labels over Lato data, zero
  shadows, 8 px radii. The logo PNG and the Gotham OTFs are inlined as data URIs so the page keeps
  its identity offline; Lato is the one network reference (Google Fonts) and degrades to the system
  sans without it. Athlete names stay Title Case per the brand rule; every other label is uppercase
  Gotham.

## Data

`theia_softball_hitting_db.segment_positions_raw` — **this is the skeleton table**. One row per
trial; every column is a gzip-JSON blob `{"0":[x...],"1":[y...],"2":[z...]}` in metres, one value
per frame on the `trials.fs_point` clock. 17 segments x `proxEndPos` / `distEndPos` / `CoMPos`:

`rpv` pelvis, `rta` thorax, `rhe` head, `l/rar` upper arm, `l/rfa` forearm, `l/rha` hand,
`l/rth` thigh, `l/rsk` shank, `l/rft` foot, plus `bat`, `ball0`, and `model_CoMPos`.

The window is trimmed to `load_time - 0.15 s` .. `bat_contact_time + 0.25 s`; the raw capture is
~760 frames of mostly standing around.

## Traps

- **The schema is `theia_softball_hitting_db`.** Caesar's contract lists it as
  `softball_theia_hitting_db`, which does not exist — that alias raises
  `Unknown database`. Baseball is `theia_hitting_db`.
- **`segment_positions_raw` is not in any contract or skill doc**, and several repos
  (`hitting_ucm`, `wiki/concepts/hitting-variance-structure.md`) state that no joint-centre source
  exists in the Theia schema. That conclusion came from looking only at `landmarks_raw` (bat +
  heels + ball). Full-body positions do exist, here. `rot_mats_raw` carries the segment
  orientations if a future version needs them.
- **Segments do not touch at the joints.** A hip end sits ~0.10 m off the pelvis segment end and a
  shoulder ~0.15 m off the thorax end, so the body reads as loose sticks unless the `LINKS`
  connectors are drawn. Within a limb the chain closes exactly (thigh distal == shank proximal,
  gap 0.000 m) — that is the check that the prox/dist pairing is right.
- **Blobs are gzipped here** (`landmarks_raw` is too, `ang_vels_raw` is not) — sniff the
  `\x1f\x8b` magic rather than assuming.
- **`bat_contact_time` is the alignment anchor, not `body_contact_time`** — the body estimate can
  sit ~8 frames off, which is enough to break a side-by-side comparison at contact.
- **`fs_point` is not always 300.** The build refuses to align two swings captured at different
  rates rather than silently stretching one.
- `poi.bat_speed_max` carries tracking misses (0 / 4 / 311 mph), so swing selection gates
  25-100 mph on top of `needs_review = 0 AND bad_datas_point = 0`.

## PII

Athlete names and mocap. Local machine only — no remote, no external service, no Artifact publish.
`dist/` is gitignored.
