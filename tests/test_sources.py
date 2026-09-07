"""Invariants for the generic loaders. Run: PYTHONUTF8=1 python tests/test_sources.py

Uses the exported CSVs in dist/ as the fixture -- build.py --export-csv writes them from the DB, so
the round-trip these assert is exactly "a Theia swing sent through the generic path comes back".
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import sources

CSV = REPO / "dist" / "addie-murphy_positions.csv"


def test_table_roundtrip():
    sw = sources.load_table(CSV)
    assert abs(sw["fs"] - 300) < 1e-6, sw["fs"]
    assert sw["anchor"] == 0 and sw["anchor_kind"] == "start"
    assert sw["events"] == {} and sw["links"] is None
    assert len(sw["points"]) == 33, len(sw["points"])
    n = {v.shape for v in sw["points"].values()}
    assert n == {(405, 3)}, n
    assert np.isfinite(sw["points"]["bat_prox"]).all()
    print("table roundtrip           OK  33 points, 405 frames, 300 Hz")


def test_normalise_is_a_noop_on_metres():
    sw = sources.load_table(CSV)
    before = sw["points"]["rhe_dist"].copy()
    after, note = sources.normalise(sw["points"], "auto")
    assert np.array_equal(before, after["rhe_dist"]), "metres/z-up data must pass through untouched"
    assert note == "already metres, z up", note
    print("normalise noop            OK  " + note)


def test_normalise_undoes_mm_and_y_up():
    sw = sources.load_table(CSV)
    # same data in millimetres with y as the vertical axis
    mangled = {k: (v[:, [0, 2, 1]] * 1000.0) for k, v in sw["points"].items()}
    fixed, note = sources.normalise(mangled, "auto")
    assert "mm -> m" in note and "up axis y -> z" in note, note
    assert np.allclose(fixed["rhe_dist"], sw["points"]["rhe_dist"], atol=1e-9)
    print("normalise mm + y-up       OK  " + note)


def test_inferred_links_recover_the_known_skeleton():
    """Compare GEOMETRY, not names: coincident points collapse, so a segment can come back under
    either of its two equivalent names (rar_dist == rfa_prox, so the forearm may be reported as
    rar_dist -> rfa_dist)."""
    sw = sources.load_table(CSV)
    P = sw["points"]
    links = sources.infer_links(P)

    def endpoints(pair):
        return tuple(sorted(tuple(np.round(np.nanmean(P[n], axis=0), 4)) for n in pair))

    drawn = {endpoints(p) for p in links}
    missing = [s_ for s_ in sources.THEIA_SEGMENTS
               if endpoints((f"{s_}_prox", f"{s_}_dist")) not in drawn]
    assert not missing, f"inference lost real segments: {missing}"
    assert len(links) <= 32, len(links)
    print(f"inferred skeleton         OK  {len(links)} links, all "
          f"{len(sources.THEIA_SEGMENTS)} known segments drawn")


for fn in (test_table_roundtrip, test_normalise_is_a_noop_on_metres,
           test_normalise_undoes_mm_and_y_up, test_inferred_links_recover_the_known_skeleton):
    fn()
print("all good")
