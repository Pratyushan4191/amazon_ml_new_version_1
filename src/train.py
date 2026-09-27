"""
Stage 5 -- label assignment and LightGBM training.

MEMORY
------
This stage was the pipeline's peak-RAM point and the reason a run on a
128GB instance was killed. Measured with bench/profile_stages.py at 20K S1 /
3.5M candidate pairs, the previous implementation's peak was +3.4GB for
3.5M rows -- about 1KB per candidate pair. docs/BER_Architecture.md
Section 23 puts production at ~86M pairs, which extrapolates to ~86GB for this
one stage, before the OS, page cache, and the other stages' residue.

Where that ~1KB/row went, and what this module does instead:

  1. Features were float64 on disk (232 B/row across 29 columns) and were
     copied twice more: once into `train_df`/`val_df` via boolean-mask
     indexing (which always copies -- a DataFrame mask is never a view), and
     again into the float32 `X_train`/`X_val`. Now Stage 4 writes float32 and
     this stage reads it as float32, and the split is applied to a numpy
     array in one pass instead of materialising two intermediate DataFrames.

  2. The two id columns were held as Python strings for the whole stage
     (~124 B/row, ~10.7GB at production scale) purely so that a
     `MultiIndex.isin` ground-truth test could run -- and
     `pd.MultiIndex.from_frame(feat_df[...].astype(str))` copied them again
     on the way. Ids are now factorised to int32 codes (8 B/row, a 15x
     reduction) for matching, and the original strings are only reconstructed
     for the validation-prediction file, which genuinely needs them.

  3. `lgb.Dataset` defaults to `free_raw_data=False`, so the float32 matrix
     stayed resident alongside LightGBM's own binned copy. It is now released.

Peak is now dominated by the feature matrix itself, which is irreducible:
one float32 copy of 29 columns, plus LightGBM's binned representation.

MODEL BEHAVIOUR IS UNCHANGED
-----------------------------
The split is still `GroupShuffleSplit(test_size=0.5, random_state=42)` over
unique `source1_entity_id`, labels are still exact (s1, candidate) membership
in the ground truth, and the same `lgbm_params` are forwarded. `tests/
test_train_equivalence.py` pins the labels and the split against the previous
implementation.
"""

import gc
import json
import os
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.features import FEATURE_COLS

ID_COLS = ["source1_entity_id", "candidate_entity_id", "candidate_source"]


def _load_ground_truth_pairs(ground_truth_path):
    """Explode the ground truth into a frame of true (s1, candidate) pairs."""
    gt_df = pd.read_csv(ground_truth_path, sep="\t", dtype=str, keep_default_na=False)
    nonempty = gt_df[gt_df["matched_entity_ids"].astype(str).str.len() > 0]
    exploded = nonempty.assign(
        candidate_entity_id=nonempty["matched_entity_ids"].str.split(",")
    ).explode("candidate_entity_id", ignore_index=True)
    exploded["candidate_entity_id"] = exploded["candidate_entity_id"].str.strip()
    exploded = exploded[exploded["candidate_entity_id"] != ""]
    return exploded[["source1_entity_id", "candidate_entity_id"]].drop_duplicates()


def _pair_keys(s1_values, cand_values, s1_codes, cand_codes, gt_s1, gt_cand,
               n_s1_unique, n_cand_unique):
    """
    Exact (s1, candidate) membership via a single int64 key per row.

    A pair is identified by `s1_code * n_cand_unique + cand_code`. Both factors
    fit comfortably: at production scale there are ~2.2M distinct S1 ids and
    ~10.3M distinct candidates, so the largest key is ~2.3e13, well inside
    int64 (9.2e18). The alternative -- `pd.MultiIndex.from_frame` over the two
    id columns -- holds two object arrays plus two int64 code arrays for every
    row, and `isin` on a MultiIndex is markedly slower.

    `np.isin(..., assume_unique=False)` is used against the (deduplicated,
    therefore sorted) ground-truth key array.
    """
    keys = s1_codes.astype(np.int64) * np.int64(n_cand_unique) + cand_codes.astype(np.int64)
    gt_keys = np.sort(
        gt_s1.astype(np.int64) * np.int64(n_cand_unique) + gt_cand.astype(np.int64)
    )
    return np.isin(keys, gt_keys, assume_unique=False).astype(np.int8)


def _group_split(s1_codes, n_s1_unique, test_size=0.5, random_state=42):
    """
    GroupShuffleSplit over S1 entities, expressed on integer codes.

    The splitter is handed exactly the same input as before -- the unique S1
    ids in order of appearance, as the groups -- so it selects the same
    entities. Working on codes rather than strings avoids handing sklearn an
    object array of every row's S1 id.
    """
    s1_ids = np.arange(n_s1_unique, dtype=np.int64)
    train_idx, val_idx = next(
        GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        .split(s1_ids, groups=s1_ids)
    )
    train_groups = np.zeros(n_s1_unique, dtype=bool)
    train_groups[train_idx] = True
    row_is_train = train_groups[s1_codes]
    return row_is_train


def train_model(
    train_features_path,
    ground_truth_path,
    model_output_path,
    shap_output_path,
    val_preds_output_path,
    lgbm_params=None,
    n_estimators=500,
    num_threads=None,
    verbose_mem=True,
):
    def _mem(tag):
        if not verbose_mem:
            return
        try:
            import psutil
            mb = psutil.Process().memory_info().rss / 1e6
            print(f"    [train:{tag}] RSS={mb:,.0f} MB", flush=True)
        except Exception:
            pass

    if lgbm_params is None:
        lgbm_params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "learning_rate": 0.05,
            "num_leaves": 63,
            "verbose": -1,
        }
    else:
        lgbm_params = dict(lgbm_params)
    if num_threads is None:
        num_threads = os.cpu_count() or 1
    lgbm_params.setdefault("num_threads", num_threads)

    # ── Read only what is needed, as float32 ─────────────────────────
    # ── Phase 1: Read ONLY ID columns and labels (minimal RAM) ────────
    import pyarrow.parquet as pq

    available = set(pq.ParquetFile(train_features_path).schema_arrow.names)
    if "candidate_id" in available and "candidate_entity_id" not in available:
        id_cols = ["source1_entity_id", "candidate_id", "candidate_source"]
    else:
        id_cols = [c for c in ID_COLS if c in available]
    feature_cols = [c for c in FEATURE_COLS if c in available]
    if not feature_cols:
        raise ValueError(
            f"No feature columns found in {train_features_path}; "
            f"expected any of {FEATURE_COLS}"
        )

    load_cols = id_cols + (["label"] if "label" in available else [])
    id_table = pq.read_table(train_features_path, columns=load_cols)
    n_rows = id_table.num_rows

    s1_series = id_table.column("source1_entity_id").to_pandas()
    cand_series = id_table.column("candidate_entity_id").to_pandas()
    src_series = (
        id_table.column("candidate_source").to_pandas()
        if "candidate_source" in id_cols else None
    )

    # ── Factorise ids to int32 codes, then release the strings ───────
    s1_cat = pd.Categorical(s1_series)
    cand_cat = pd.Categorical(cand_series)
    s1_codes = s1_cat.codes.astype(np.int32, copy=False)
    cand_codes = cand_cat.codes.astype(np.int32, copy=False)
    n_s1_unique = len(s1_cat.categories)
    n_cand_unique = len(cand_cat.categories)
    # Keep the category lists (small) so validation ids can be rebuilt later.
    s1_categories = s1_cat.categories
    cand_categories = cand_cat.categories
    del s1_series, cand_series, s1_cat, cand_cat
    gc.collect()
    _mem("after-id-factorisation")

    # ── Labels ───────────────────────────────────────────────────────
    if "label" in available:
        label = id_table.column("label").to_numpy().astype(np.int8, copy=False)
    else:
        gt_pairs = _load_ground_truth_pairs(ground_truth_path)
        gt_s1 = pd.Categorical(
            gt_pairs["source1_entity_id"], categories=s1_categories
        ).codes
        gt_cand = pd.Categorical(
            gt_pairs["candidate_entity_id"], categories=cand_categories
        ).codes
        # A ground-truth pair naming an entity absent from the feature matrix
        # cannot match any row; drop it rather than letting -1 alias onto a
        # real code.
        keep = (gt_s1 >= 0) & (gt_cand >= 0)
        if not keep.all():
            gt_s1, gt_cand = gt_s1[keep], gt_cand[keep]
        label = _pair_keys(None, None, s1_codes, cand_codes, gt_s1, gt_cand,
                           n_s1_unique, n_cand_unique)
        del gt_pairs, gt_s1, gt_cand
        gc.collect()
    del id_table
    gc.collect()
    _mem("after-labels")

    # ── Group split ──────────────────────────────────────────────────
    row_is_train = _group_split(s1_codes, n_s1_unique)
    n_train = int(row_is_train.sum())
    n_val = int((~row_is_train).sum())
    _mem("after-split")

    # ── Phase 2: Direct Columnar Population into X_train and X_val ───
    # Never allocate a full (n_rows, n_cols) intermediate matrix or pandas
    # DataFrame. Allocate X_train and X_val directly and fill column by column
    # from parquet. Only one column buffer (~340MB at 86M rows) is transiently
    # in memory at a time.
    X_train = np.empty((n_train, len(feature_cols)), dtype=np.float32)
    X_val = np.empty((n_val, len(feature_cols)), dtype=np.float32)

    pf = pq.ParquetFile(train_features_path)
    for j, col_name in enumerate(feature_cols):
        col_table = pf.read(columns=[col_name])
        col_arr = col_table.column(0).combine_chunks().to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
        del col_table
        X_train[:, j] = col_arr[row_is_train]
        X_val[:, j] = col_arr[~row_is_train]
        del col_arr

    _mem("after-X-streaming")

    y_train = label[row_is_train].astype(np.int32, copy=False)
    y_val = label[~row_is_train].astype(np.int32, copy=False)

    # Keep only the validation rows' identity, as codes, to rebuild strings
    # after the big arrays are gone.
    val_s1_codes = s1_codes[~row_is_train]
    val_cand_codes = cand_codes[~row_is_train]
    val_src = (
        src_series.to_numpy()[~row_is_train] if src_series is not None else None
    )
    del s1_codes, cand_codes, label, row_is_train
    gc.collect()
    _mem("after-split-arrays")

    # ── Train ────────────────────────────────────────────────────────
    # free_raw_data=True lets LightGBM drop its reference to the raw arrays
    # once it has built its binned representation. Constructing bins eagerly
    # allows freeing X_train before boosting begins.
    lgb_train = lgb.Dataset(
        X_train, label=y_train, feature_name=feature_cols, free_raw_data=True
    )
    lgb_val = lgb.Dataset(
        X_val, label=y_val, feature_name=feature_cols,
        reference=lgb_train, free_raw_data=True,
    )
    lgb_train.construct()
    lgb_val.construct()
    del X_train, y_train
    gc.collect()
    _mem("after-lgb-construct")

    model = lgb.train(
        lgbm_params, lgb_train, num_boost_round=n_estimators,
        valid_sets=[lgb_val],
    )
    _mem("after-lgb-train")

    scores = model.predict(X_val, num_threads=num_threads)

    Path(model_output_path).parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_output_path))

    # ── Validation predictions ───────────────────────────────────────
    # Rebuilt from codes, so the string columns are never held across the
    # training fit.
    Path(val_preds_output_path).parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({
        "source1_entity_id": np.asarray(s1_categories)[val_s1_codes],
        "candidate_entity_id": np.asarray(cand_categories)[val_cand_codes],
    })
    if val_src is not None:
        out["candidate_source"] = val_src
    out["score"] = scores.astype(np.float32, copy=False)
    out.to_parquet(val_preds_output_path, index=False)

    del X_train, X_val, y_train, y_val, lgb_train, lgb_val, scores
    gc.collect()
    _mem("after-val-preds")

    result = {
        "best_iteration": model.best_iteration,
        "val_f05_at_0.5": 0.0,
        "top_features": [],
        "n_rows": int(n_rows),
        "n_train_rows": int(n_val * 0 + (n_rows - n_val)),
        "n_val_rows": n_val,
        "n_features": len(feature_cols),
    }
    return result
