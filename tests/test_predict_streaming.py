"""
The streaming inference in `src.predict` must emit exactly the same
`matching_results.tsv` as the previous load-everything implementation.

The reference below is the previous implementation verbatim. It is kept here
because the streaming rewrite changes *how* memory is used, and the only thing
that makes that safe is that it changes nothing about the answer.

Run:  python -m pytest tests/test_predict_streaming.py -v
"""

import json
import os
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from src.features import FEATURE_COLS


def _reference_run_inference(test_features_path, model_path, threshold_path,
                             output_path, all_s1_ids=None):
    """Previous implementation, verbatim."""
    model = lgb.Booster(model_file=str(model_path))
    with open(threshold_path) as f:
        t_star = float(json.load(f)["t_star"])
    test_df = pd.read_parquet(test_features_path)
    all_s1 = (
        list(dict.fromkeys(all_s1_ids))
        if all_s1_ids is not None
        else sorted(test_df["source1_entity_id"].unique().tolist())
    )
    scores = model.predict(test_df[FEATURE_COLS].astype(np.float32))
    test_df["score"] = scores
    match_dict = {s: [] for s in all_s1}
    above = test_df[test_df["score"] >= t_star]
    for s1, grp in above.groupby("source1_entity_id"):
        match_dict[s1] = sorted(
            list(dict.fromkeys(grp["candidate_entity_id"].tolist())),
            key=lambda x: (x[:2], int(x[3:]) if x[3:].isdigit() else 0),
        )
    rows = [{"source1_entity_id": k, "matched_entity_ids": ",".join(v)}
            for k, v in match_dict.items()]
    out_df = pd.DataFrame(rows)
    out_df["matched_entity_ids"] = out_df["matched_entity_ids"].astype(str).replace("nan", "")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, sep="\t", index=False, na_rep="")


def _fixture(tmp_path, n_s1=200, n_cand=600, seed=5):
    rng = np.random.default_rng(seed)
    s1_ids = [f"S1-{i:05d}" for i in range(n_s1)]
    cand_ids = [f"S2-{i:05d}" for i in range(n_cand // 2)] + \
               [f"S3-{i:05d}" for i in range(n_cand // 2)]

    rows_s1, rows_c, rows_s = [], [], []
    feats = []
    for s1 in s1_ids:
        for _ in range(rng.integers(0, 9)):   # includes zero-candidate entities
            c = cand_ids[rng.integers(0, len(cand_ids))]
            rows_s1.append(s1)
            rows_c.append(c)
            rows_s.append("S2" if c.startswith("S2-") else "S3")
            feats.append(rng.random(len(FEATURE_COLS)).astype(np.float32))
    df = pd.DataFrame({
        "source1_entity_id": rows_s1,
        "candidate_entity_id": rows_c,
        "candidate_source": rows_s,
    })
    for j, c in enumerate(FEATURE_COLS):
        df[c] = np.array([f[j] for f in feats], dtype=np.float32)

    feat_p = tmp_path / "feats.parquet"
    df.to_parquet(feat_p, index=False)

    # A tiny model, trained on the same features.
    y = (rng.random(len(df)) < 0.3).astype(np.int32)
    m = lgb.train(
        {"objective": "binary", "verbose": -1, "num_leaves": 7,
         "num_threads": 2, "min_data_in_leaf": 1, "seed": 1},
        lgb.Dataset(df[FEATURE_COLS].to_numpy(np.float32), label=y),
        num_boost_round=15,
    )
    model_p = tmp_path / "m.lgb"
    m.save_model(str(model_p))

    # A threshold that actually splits the population, so the test is not
    # trivially "everything above" or "nothing above".
    scores = m.predict(df[FEATURE_COLS].to_numpy(np.float32))
    t = float(np.median(scores))
    thr_p = tmp_path / "thr.json"
    thr_p.write_text(json.dumps({"t_star": t, "max_f05": 0.0}))
    return feat_p, model_p, thr_p, s1_ids, t


def test_matches_reference_output(tmp_path):
    feat_p, model_p, thr_p, s1_ids, t = _fixture(tmp_path)

    ref_out = tmp_path / "ref.tsv"
    _reference_run_inference(feat_p, model_p, thr_p, ref_out, all_s1_ids=s1_ids)

    from src.predict import run_inference
    new_out = tmp_path / "new.tsv"
    run_inference(feat_p, model_p, thr_p, new_out, all_s1_ids=s1_ids)

    assert ref_out.read_text() == new_out.read_text(), (
        "streaming inference diverged from the reference implementation"
    )


def test_batch_size_is_transparent(tmp_path):
    """Row-group batch size must not change a single output byte."""
    feat_p, model_p, thr_p, s1_ids, t = _fixture(tmp_path)
    from src.predict import run_inference

    outs = []
    for bs in (1, 7, 1000, 10_000_000):
        o = tmp_path / f"o_{bs}.tsv"
        run_inference(feat_p, model_p, thr_p, o, all_s1_ids=s1_ids, batch_rows=bs)
        outs.append(o.read_text())
    assert all(o == outs[0] for o in outs)


def test_every_s1_present_including_zero_candidate(tmp_path):
    feat_p, model_p, thr_p, s1_ids, t = _fixture(tmp_path)
    from src.predict import run_inference
    out = tmp_path / "cover.tsv"
    run_inference(feat_p, model_p, thr_p, out, all_s1_ids=s1_ids)
    got = pd.read_csv(out, sep="\t", dtype=str, keep_default_na=False)
    assert list(got["source1_entity_id"]) == s1_ids
    assert got["matched_entity_ids"].isna().sum() == 0
    # entities with no candidate rows at all must still be present, blank
    zero = set(s1_ids) - set(pd.read_parquet(feat_p)["source1_entity_id"])
    assert zero, "fixture should contain zero-candidate entities"
    for s in zero:
        assert got.loc[got.source1_entity_id == s, "matched_entity_ids"].iloc[0] == ""


def test_candidate_sort_is_numeric_not_lexicographic(tmp_path):
    from src.predict import _candidate_sort_key
    assert sorted(["S2-10", "S2-2", "S2-1"], key=_candidate_sort_key) == \
        ["S2-1", "S2-2", "S2-10"]
    assert sorted(["S3-9", "S2-100"], key=_candidate_sort_key) == \
        ["S2-100", "S3-9"]


def test_all_s1_ids_deduped_preserving_order(tmp_path):
    feat_p, model_p, thr_p, s1_ids, t = _fixture(tmp_path)
    from src.predict import run_inference
    out = tmp_path / "dedup.tsv"
    dup = [s1_ids[0]] + s1_ids + [s1_ids[1]]
    run_inference(feat_p, model_p, thr_p, out, all_s1_ids=dup)
    got = pd.read_csv(out, sep="\t", dtype=str, keep_default_na=False)
    assert list(got["source1_entity_id"]) == s1_ids
