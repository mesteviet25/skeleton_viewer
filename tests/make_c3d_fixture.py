"""Turn an exported wide CSV into a C3D so the c3d loader can be tested on real positions.

Deliberately writes millimetres with the vertical on axis 1 (y-up) -- the two conventions
`sources.normalise` has to undo. Round-tripping through this is how the c3d path is checked without
an openbiomechanics download.

    uv run --no-project --with ezc3d --with pandas --with numpy python tests/make_c3d_fixture.py \
        dist/addie-murphy_positions.csv dist/addie.c3d
"""
import sys

import ezc3d
import numpy as np
import pandas as pd

src, out = sys.argv[1], sys.argv[2]
df = pd.read_csv(src)

pts = {}
for c in df.columns:
    if c[-2:] in ("_x", "_y", "_z"):
        pts.setdefault(c[:-2], {})[c[-1]] = df[c].to_numpy(float)
names = [k for k, v in pts.items() if len(v) == 3]

arr = np.ones((4, len(names), len(df)))
for i, k in enumerate(names):
    v = np.stack([pts[k]["x"], pts[k]["y"], pts[k]["z"]], axis=0)
    arr[:3, i, :] = v[[0, 2, 1]] * 1000.0

c = ezc3d.c3d()
c["parameters"]["POINT"]["RATE"]["value"] = [300.0]
c["parameters"]["POINT"]["LABELS"]["value"] = names
c["data"]["points"] = arr
c.write(out)
print(f"wrote {out}  {len(names)} markers x {len(df)} frames, mm, y-up")
