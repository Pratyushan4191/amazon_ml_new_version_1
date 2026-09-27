"""
The vectorised threshold sweep must agree with the set-based implementation it
replaced, on every threshold -- not just on the selected one.

The reference below is the previous implementation verbatim. It rebuilds the
prediction SETS for each of the ~99 thresholds and intersects them in Python;
this test asserts the new count-based implementation produces the same
macro-F0.5 curve, the same `t_star` (including its tie-breaking), and the same
`max_f05`.

Run:  python -m pytest tests/test_threshold_sweep.py -v
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.threshold import sweep_threshold
from eval import macro_f05


def _reference_sweep(val_preds_path, ground_truth_path, threshold_output_path,
                     sweep_start=0.01, sweep_stop=0.99, sweep_step=0.01,
                     candidate_col="candidate_entity_id",
                     candidate_pairs_path=None):
    """Previous implementation, verbatim."""
    val_df = pd.read_parquet(val_preds_path)
    gt_df = pd.read_csv(ground_truth_path, sep="\t", dtype=str, keep_default_na=False)
    if candidate_col not in val_df.columns and "candidate_id" in val_df.columns:
        val_df = val_df.rename(columns={"candidate_id": candidate_col})
    val_s1_ids = set(val_df["source1_entity_id"].astype(str).unique().tolist())

    zero_candidate_ids = set()
    if candidate_pairs_path is not None:
        cp_df = pd.read_csv(candidate_pairs_path, sep="\t", dtype=str,
                            keep_default_na=False)
        zero_candidate_ids = set(
            cp_df.loc[cp_df["candidate_entity_ids"].astype(str).str.len() == 0,
                      "source1_entity_id"].astype(str)
        )
        zero_candidate_ids -= val_s1_ids

    full_population = val_s1_ids | zero_candidate_ids
    gt_map = {str(r["source1_entity_id"]): set(str(r.get("matched_entity_ids", "")).split(",")) - {""}
              for _, r in gt_df.iterrows()}
    truth_dict = {sid: gt_map.get(sid, set()) for sid in full_population}

    thresholds = np.round(np.arange(sweep_start, sweep_stop + sweep_step, sweep_step), 4)
    best_t, max_f = 0.5, 0.0
    curve = []
    for t in thresholds:
        above = val_df[val_df["score"] >= t]
        pred_dict = {sid: set() for sid in full_population}
        for s1_id, grp in above.groupby("source1_entity_id"):
            if s1_id in pred_dict:
                pred_dict[s1_id] = set(grp[candidate_col].tolist())
        f05 = macro_f05(pred_dict, truth_dict)
        curve.append(f05)
        if f05 >= max_f:
            best_t, max_f = float(t), float(f05)

    Path(threshold_output_path).parent.mkdir(parents=True, exist_ok=True)
    result = {"t_star": best_t, "max_f05": max_f,
              "n_population": len(full_population),
              "n_with_candidates": len(val_s1_ids),
              "n_zero_candidates_included": len(zero_candidate_ids)}
    with open(threshold_output_path, "w") as f:
        json.dump(result, f, indent=2)
    return result, np.array(curve)


def _fixture(tmp_path, n_s1=300, seed=3):
    """Entities covering: singletons, multi-match, zero-candidate, all-below."""
    rng = np.random.default_rng(seed)
    rows, gt = [], []
    for i in range(n_s1):
        s1 = f"S1-{i:05d}"
        k = int(rng.integers(0, 6))          # 0 -> zero-candidate entity
        truth = []
        for j in range(k):
            c = f"S2-{int(rng.integers(0, 400)):05d}"
            truth.append(c)
            rows.append({
                "source1_entity_id": s1,
                "candidate_entity_id": c,
                "candidate_source": "S2",
                # half true, half distractors
                "score": float(rng.random()) if rng.random() < 0.5 else float(rng.random() * 0.3),
            })
        # distractors that are never true
        for _ in range(int(rng.integers(0, 4))):
            c = f"S3-{int(rng.integers(0, 400)):05d}"
            rows.append({
                "source1_entity_id": s1, "candidate_entity_id": c,
                "candidate_source": "S3", "score": float(rng.random()),
            })
        gt.append({"source1_entity_id": s1, "matched_entity_ids": ",".join(truth)})

    val_p = tmp_path / "val.parquet"
    pd.DataFrame(rows).to_parquet(val_p, index=False)
    gt_p = tmp_path / "gt.tsv"
    pd.DataFrame(gt).to_csv(gt_p, sep="\t", index=False)

    # candidate_pairs.tsv: include the zero-candidate entities explicitly
    cp_p = tmp_path / "cands.tsv"
    with open(cp_p, "w") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        by_s1 = {}
        for r in rows:
            by_s1.setdefault(r["source1_entity_id"], []).append(r["candidate_entity_id"])
        for g in gt:
            f.write(f"{g['source1_entity_id']}\t{','.join(by_s1.get(g['source1_entity_id'], []))}\n")
    return val_p, gt_p, cp_p


def test_matches_reference_curve_and_selection(tmp_path):
    val_p, gt_p, cp_p = _fixture(tmp_path)
    ref, ref_curve = _reference_sweep(val_p, gt_p, tmp_path / "ref.json",
                                      candidate_pairs_path=cp_p)
    new = sweep_threshold(val_p, gt_p, tmp_path / "new.json",
                          candidate_pairs_path=cp_p)
    assert new["t_star"] == ref["t_star"]
    assert new["max_f05"] == pytest.approx(ref["max_f05"], abs=1e-12)
    assert new["n_population"] == ref["n_population"]
    assert new["n_with_candidates"] == ref["n_with_candidates"]
    assert new["n_zero_candidates_included"] == ref["n_zero_candidates_included"]


def test_f05_curve_identical_at_every_threshold(tmp_path):
    """Not just the argmax -- the whole curve must match."""
    val_p, gt_p, cp_p = _fixture(tmp_path)
    # Compare against the reference WITHOUT the candidate_pairs population
    # widening, so the two curves describe the same population. (Comparing a
    # widened new curve against a non-widened reference would compare different
    # denominators.)
    _, ref_curve = _reference_sweep(val_p, gt_p, tmp_path / "ref2.json")

    # Recompute the new curve directly.
    from src.threshold import _f05_from_counts, _sweep_counts
    val_df = pd.read_parquet(val_p)
    gt_df = pd.read_csv(gt_p, sep="\t", dtype=str, keep_default_na=False)
    population = sorted(set(val_df["source1_entity_id"]))
    ent_index = {s: i for i, s in enumerate(population)}
    gt_map = {str(r["source1_entity_id"]): set(str(r["matched_entity_ids"]).split(",")) - {""}
              for _, r in gt_df.iterrows()}
    n_ent = len(population)
    n_true = np.array([len(gt_map.get(s, set())) for s in population])
    re = val_df["source1_entity_id"].map(ent_index).to_numpy()
    sc = val_df["score"].to_numpy(dtype=np.float64)
    cd = val_df["candidate_entity_id"].astype(str).to_numpy()
    thr = np.round(np.arange(0.01, 0.99 + 0.01, 0.01), 4)

    # A prediction is a SET, so duplicate (s1, candidate) rows must collapse to
    # one entry, keeping the max score. This mirrors what the implementation
    # does; without it this test would be measuring a different (wrong) curve.
    o = np.lexsort((-sc, cd, re))
    re, cd, sc = re[o], cd[o], sc[o]
    first = np.empty(len(re), dtype=bool)
    first[0] = True
    if len(re) > 1:
        np.logical_or(re[1:] != re[:-1], cd[1:] != cd[:-1], out=first[1:])
    re, cd, sc = re[first], cd[first], sc[first]

    o = np.lexsort((-sc, re))
    n_pred = _sweep_counts(sc[o], re[o], n_ent, thr)
    lookup = {(i, c) for i, s in enumerate(population) for c in gt_map.get(s, set())}
    is_t = np.fromiter(((int(e), str(c)) in lookup for e, c in zip(re, cd)),
                       bool, len(re))
    o2 = np.lexsort((-sc[is_t], re[is_t]))
    tp = _sweep_counts(sc[is_t][o2], re[is_t][o2], n_ent, thr)
    new_curve = _f05_from_counts(n_pred, n_true, tp).mean(axis=0)

    np.testing.assert_allclose(new_curve, ref_curve, rtol=0, atol=1e-12)


def test_zero_candidate_population_is_included(tmp_path):
    val_p, gt_p, cp_p = _fixture(tmp_path)
    new = sweep_threshold(val_p, gt_p, tmp_path / "z.json", candidate_pairs_path=cp_p)
    assert new["n_zero_candidates_included"] > 0
    assert new["n_population"] == new["n_with_candidates"] + new["n_zero_candidates_included"]


def test_without_candidate_pairs_path_is_backward_compatible(tmp_path):
    val_p, gt_p, cp_p = _fixture(tmp_path)
    new = sweep_threshold(val_p, gt_p, tmp_path / "np.json")
    assert new["n_zero_candidates_included"] == 0
    ref, _ = _reference_sweep(val_p, gt_p, tmp_path / "npref.json")
    assert new["t_star"] == ref["t_star"]


def test_vectorised_f05_matches_scalar_on_edges():
    from eval import f05_score
    from src.threshold import _f05_from_counts
    # (n_pred, n_true, tp) with tp <= min(n_pred, n_true) so the triple is
    # realisable by actual sets.
    cases = [
        (0, 0, 0),   # no truth, no pred -> 1.0
        (3, 0, 0),   # no truth, has pred -> 0.0
        (0, 2, 0),   # truth, no pred -> 0.0
        (2, 2, 1),   # partial
        (2, 2, 2),   # perfect
        (5, 2, 0),   # all wrong
        (1, 1, 1),   # perfect, single
        (3, 1, 1),   # one of three correct
        (2, 3, 2),
        (4, 4, 3),
    ]
    n_pred = np.array([c[0] for c in cases])
    n_true = np.array([c[1] for c in cases])
    tp = np.array([c[2] for c in cases])
    got = _f05_from_counts(n_pred, n_true, tp)
    for i, (n_pd, n_tr, t) in enumerate(cases):
        # Build concrete sets realising exactly (|pred|, |truth|, tp).
        pred = set(range(n_pd))
        truth = set(range(n_pd - t, n_pd - t + n_tr))
        assert len(truth) == n_tr
        assert len(pred & truth) == t
        assert got[i, 0] == pytest.approx(f05_score(pred, truth), abs=1e-12), cases[i]


def test_single_threshold_sweep(tmp_path):
    val_p, gt_p, cp_p = _fixture(tmp_path)
    new = sweep_threshold(val_p, gt_p, tmp_path / "one.json",
                          sweep_start=0.5, sweep_stop=0.5, sweep_step=0.01,
                          candidate_pairs_path=cp_p)
    # np.arange(0.5, 0.51, 0.01) yields BOTH 0.5 and 0.51 in floating point,
    # and ties resolve to the last threshold -- so just assert the selection is
    # one of the swept values rather than pinning an exact one.
    assert new["t_star"] in (0.5, 0.51)
