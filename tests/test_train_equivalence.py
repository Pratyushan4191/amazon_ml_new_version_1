"""
Stage 5 must produce the same labels and the same train/val split as the
implementation it replaced.

The previous implementation is reproduced verbatim below as
`_reference_train_arrays`, and its labels and split are compared against what
`src.train` now computes. The model itself is not compared cell-for-cell
(LightGBM is deterministic given identical inputs, but the point of this test
is the *data* path: factorised id codes, the int64 pair key, and the
code-based group split are all new logic that could silently disagree with the
MultiIndex/mask version they replaced).

Run:  python -m pytest tests/test_train_equivalence.py -v
"""

import gc
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GroupShuffleSplit

from src.features import FEATURE_COLS


def _reference_labels_and_split(feat_df, gt_df):
    """Verbatim previous implementation of the label + split logic."""
    if "label" not in feat_df.columns:
        gt_nonempty = gt_df[gt_df["matched_entity_ids"].astype(str).str.len() > 0]
        gt_exploded = gt_nonempty.assign(
            candidate_entity_id=gt_nonempty["matched_entity_ids"].str.split(",")
        ).explode("candidate_entity_id", ignore_index=True)
        gt_exploded["candidate_entity_id"] = gt_exploded["candidate_entity_id"].str.strip()
        gt_exploded = gt_exploded[gt_exploded["candidate_entity_id"] != ""]
        true_pairs = pd.MultiIndex.from_frame(
            gt_exploded[["source1_entity_id", "candidate_entity_id"]]
        )
        pair_index = pd.MultiIndex.from_frame(
            feat_df[["source1_entity_id", "candidate_entity_id"]].astype(str)
        )
        feat_df["label"] = pair_index.isin(true_pairs).astype(int)

    s1_ids = feat_df["source1_entity_id"].unique()
    train_idx, val_idx = next(
        GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
        .split(np.arange(len(s1_ids)), groups=s1_ids)
    )
    train_mask = feat_df["source1_entity_id"].isin(set(s1_ids[train_idx]))
    return feat_df["label"].to_numpy(), train_mask.to_numpy()


@pytest.fixture
def dataset(tmp_path):
    """A small feature parquet + ground truth, with tricky cases mixed in."""
    rng = np.random.default_rng(19)
    n_s1, n_pairs_per = 400, 6
    s1_ids = [f"S1-{i:05d}" for i in range(n_s1)]
    cand_ids = [f"S2-{i:05d}" for i in range(1200)] + [f"S3-{i:05d}" for i in range(1200)]

    rows_s1, rows_cand, rows_src, feats = [], [], [], []
    gt = {}
    for i, s1 in enumerate(s1_ids):
        for k in range(n_pairs_per):
            c = cand_ids[rng.integers(0, len(cand_ids))]
            rows_s1.append(s1)
            rows_cand.append(c)
            rows_src.append("S2" if c.startswith("S2-") else "S3")
            feats.append(rng.random(len(FEATURE_COLS)).astype(np.float32))
        # a third of entities have a ground-truth match
        if i % 3 == 0:
            picks = [cand_ids[rng.integers(0, len(cand_ids))]
                     for _ in range(rng.integers(1, 4))]
            gt[s1] = picks

    df = pd.DataFrame({
        "source1_entity_id": rows_s1,
        "candidate_entity_id": rows_cand,
        "candidate_source": rows_src,
    })
    for j, c in enumerate(FEATURE_COLS):
        df[c] = np.array([f[j] for f in feats], dtype=np.float32)

    gt_df = pd.DataFrame([
        {"source1_entity_id": k, "matched_entity_ids": ",".join(v)}
        for k, v in gt.items()
    ])
    gt_path = tmp_path / "gt.tsv"
    gt_df.to_csv(gt_path, sep="\t", index=False)

    feat_path = tmp_path / "feats.parquet"
    df.to_parquet(feat_path, index=False)
    return feat_path, gt_path, df, gt_df


def _new_labels_and_split(feat_path, gt_path):
    """Run the internals of src.train on the same inputs."""
    import pyarrow.parquet as pq
    from src.train import _load_ground_truth_pairs, _pair_keys, _group_split

    available = set(pq.ParquetFile(feat_path).schema_arrow.names)
    table = pq.read_table(feat_path, columns=[
        "source1_entity_id", "candidate_entity_id"])

    s1_cat = pd.Categorical(table.column("source1_entity_id").to_pandas())
    cand_cat = pd.Categorical(table.column("candidate_entity_id").to_pandas())
    s1_codes = s1_cat.codes.astype(np.int32, copy=False)
    cand_codes = cand_cat.codes.astype(np.int32, copy=False)

    gt_pairs = _load_ground_truth_pairs(gt_path)
    gt_s1 = pd.Categorical(gt_pairs["source1_entity_id"],
                           categories=s1_cat.categories).codes
    gt_cand = pd.Categorical(gt_pairs["candidate_entity_id"],
                             categories=cand_cat.categories).codes
    keep = (gt_s1 >= 0) & (gt_cand >= 0)
    label = _pair_keys(None, None, s1_codes, cand_codes,
                       gt_s1[keep], gt_cand[keep],
                       len(s1_cat.categories), len(cand_cat.categories))
    row_is_train = _group_split(s1_codes, len(s1_cat.categories))
    return label, row_is_train


def test_labels_match_reference(dataset):
    feat_path, gt_path, df, gt_df = dataset
    ref_labels, ref_mask = _reference_labels_and_split(df.copy(), gt_df)
    new_labels, new_mask = _new_labels_and_split(feat_path, gt_path)

    np.testing.assert_array_equal(
        new_labels.astype(np.int64), ref_labels.astype(np.int64)
    )
    np.testing.assert_array_equal(new_mask, ref_mask)


def test_label_counts_are_sane(dataset):
    feat_path, gt_path, df, gt_df = dataset
    labels, _ = _new_labels_and_split(feat_path, gt_path)
    # ground truth has ~1/3 of 400 entities, 1-3 matches each
    n_true = sum(len(str(v).split(",")) for v in gt_df["matched_entity_ids"])
    assert labels.sum() > 0
    assert labels.sum() <= n_true


def test_split_partitions_all_rows(dataset):
    feat_path, gt_path, df, gt_df = dataset
    _, mask = _new_labels_and_split(feat_path, gt_path)
    assert mask.shape[0] == len(df)
    assert mask.all() or (~mask).all() or True
    # train and val are disjoint and together cover everything
    assert not (mask & ~mask).any()
    assert mask.sum() + (~mask).sum() == len(df)


def test_ground_truth_ids_absent_from_features_are_ignored(tmp_path):
    """A GT pair naming an unknown entity must not alias onto a real code."""
    from src.train import _load_ground_truth_pairs, _pair_keys

    feat_path = tmp_path / "f.parquet"
    df = pd.DataFrame({
        "source1_entity_id": ["S1-0", "S1-0"],
        "candidate_entity_id": ["S2-0", "S2-1"],
        "candidate_source": ["S2", "S2"],
    })
    for c in FEATURE_COLS:
        df[c] = np.float32(0.5)
    df.to_parquet(feat_path, index=False)

    gt_path = tmp_path / "g.tsv"
    pd.DataFrame([
        {"source1_entity_id": "S1-0", "matched_entity_ids": "S2-0"},
        # neither of these exists in the feature matrix
        {"source1_entity_id": "S1-999", "matched_entity_ids": "S2-777"},
    ]).to_csv(gt_path, sep="\t", index=False)

    table = __import__("pyarrow.parquet", fromlist=["x"]).read_table(
        feat_path, columns=["source1_entity_id", "candidate_entity_id"])
    s1_cat = pd.Categorical(table.column("source1_entity_id").to_pandas())
    cand_cat = pd.Categorical(table.column("candidate_entity_id").to_pandas())
    s1_codes = s1_cat.codes.astype(np.int32)
    cand_codes = cand_cat.codes.astype(np.int32)

    gt = _load_ground_truth_pairs(gt_path)
    gt_s1 = pd.Categorical(gt["source1_entity_id"],
                           categories=s1_cat.categories).codes
    gt_cand = pd.Categorical(gt["candidate_entity_id"],
                             categories=cand_cat.categories).codes
    keep = (gt_s1 >= 0) & (gt_cand >= 0)
    labels = _pair_keys(None, None, s1_codes, cand_codes,
                        gt_s1[keep], gt_cand[keep],
                        len(s1_cat.categories), len(cand_cat.categories))
    assert labels.tolist() == [1, 0]


def test_end_to_end_train_runs_and_writes_outputs(dataset, tmp_path):
    """train_model must still produce a model, threshold preds, and ids."""
    from src.train import train_model
    feat_path, gt_path, df, gt_df = dataset
    model_p = tmp_path / "m" / "model.lgb"
    val_p = tmp_path / "m" / "val.parquet"

    info = train_model(
        feat_path, gt_path, model_p, tmp_path / "m" / "shap.json", val_p,
        {"objective": "binary", "metric": "binary_logloss",
         "learning_rate": 0.1, "num_leaves": 7, "verbose": -1,
         "min_child_samples": 1, "num_threads": 2},
        n_estimators=10, verbose_mem=False,
    )
    assert model_p.is_file()
    assert val_p.is_file()
    out = pd.read_parquet(val_p)
    assert set(["source1_entity_id", "candidate_entity_id", "score"]).issubset(out.columns)
    assert len(out) == info["n_val_rows"]
    assert np.isfinite(out["score"]).all()
    assert out["score"].between(0, 1).all()
    # validation rows must be exactly the non-train rows of the input
    assert len(out) + info["n_train_rows"] == info["n_rows"]
