"""
Final inference: score test candidates and emit matching_results.tsv.

MEMORY
------
This stage used to do:

    test_df = pd.read_parquet(test_features_path)      # every column
    scores  = model.predict(test_df[FEATURE_COLS].astype(np.float32))
    test_df["score"] = scores                          # +1 more column
    above   = test_df[test_df["score"] >= t_star]      # yet another copy
    ...                                                # then 2 columns used

so the full 29-feature matrix was materialised twice, plus the score column,
plus a filtered copy -- ~31GB at the ~86M-pair production scale
(docs/BER_Architecture.md Section 23) to produce an answer with two columns in
it. Measured peak in bench/memory_benchmark.py was 2066MB for 3.5M rows.

It now streams the parquet one row group at a time: read the two id columns
and the features for that row group, predict, keep only the rows at or above
the threshold, and release. Peak becomes a function of the row-group size and
of how many candidates clear the threshold -- not of the total candidate count.

The output is byte-for-byte what the previous implementation produced: the same
rows in the same order, the same sort order within an entity, and the same
empty string for an entity with no surviving candidate. `tests/
test_predict_streaming.py` pins that.
"""

import json
import os
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.features import FEATURE_COLS

ID_COLS = ["source1_entity_id", "candidate_entity_id", "candidate_source"]


def _candidate_sort_key(x: str):
    """
    Order candidates as (source prefix, numeric id).

    Preserved verbatim from the previous implementation: 'S2-2' must sort
    before 'S2-10', which plain lexicographic ordering gets wrong. Ids that do
    not parse numerically fall back to 0 so a stray id cannot raise.
    """
    tail = x[3:]
    return (x[:2], int(tail) if tail.isdigit() else 0)


def run_inference(
    test_features_path,
    model_path,
    threshold_path,
    output_path,
    candidate_pairs_path=None,
    all_s1_ids=None,
    batch_rows: int = 2_000_000,
    num_threads: int = None,
):
    model = lgb.Booster(model_file=str(model_path))
    with open(threshold_path) as f:
        t_star = float(json.load(f)["t_star"])

    if num_threads is None:
        num_threads = os.cpu_count() or 1

    # Every S1 entity that must appear in the output, including those with no
    # candidates at all (they get an empty match list).
    if all_s1_ids is not None:
        all_s1 = list(dict.fromkeys(all_s1_ids))
    else:
        # No explicit list: fall back to whatever the feature file mentions.
        import pyarrow.parquet as pq
        col = pq.ParquetFile(test_features_path).schema_arrow.names
        key = "source1_entity_id" if "source1_entity_id" in col else col[0]
        seen = pq.read_table(test_features_path, columns=[key]).column(key).to_pylist()
        all_s1 = sorted(set(seen))

    match_dict = {s: [] for s in all_s1}
    # Candidates that clear the threshold, accumulated across batches.
    # An S1 entity's candidates are NOT guaranteed to be contiguous in the
    # parquet, so they can straddle a batch boundary -- they must be
    # accumulated, not overwritten per batch. (Assigning per batch silently
    # dropped everything but the last batch's candidates for any entity that
    # spanned a boundary, which made the output depend on batch_rows.)
    collected: dict = {}

    import pyarrow.parquet as pq

    pf = pq.ParquetFile(test_features_path)
    available = set(pf.schema_arrow.names)
    cand_col = ("candidate_entity_id" if "candidate_entity_id" in available
                else "candidate_id")
    read_cols = ["source1_entity_id", cand_col] + [c for c in FEATURE_COLS if c in available]

    n_scored = 0
    n_above = 0
    for batch in pf.iter_batches(batch_size=batch_rows, columns=read_cols):
        s1 = batch.column("source1_entity_id").to_pylist()
        cand = batch.column(cand_col).to_pylist()
        X = (pd.DataFrame({c: batch.column(c).to_numpy(zero_copy_only=False)
                           for c in FEATURE_COLS if c in available})
             .reindex(columns=FEATURE_COLS)
             .to_numpy(dtype=np.float32))
        scores = model.predict(X, num_threads=num_threads)
        n_scored += len(scores)

        arr = np.asarray(scores)
        keep_idx = np.nonzero(arr >= t_star)[0]
        n_above += len(keep_idx)
        if len(keep_idx) == 0:
            continue
        # Group the surviving candidates by S1 in pure Python: at this point
        # the arrays are small (only above-threshold rows) and this avoids
        # building any intermediate DataFrame.
        for i in keep_idx:
            collected.setdefault(s1[i], set()).add(cand[i])

    for s1_id, cands in collected.items():
        # An S1 id present in the features but absent from all_s1_ids is
        # dropped, exactly as before: the output row set is fixed by
        # all_s1_ids, not by the feature file.
        if s1_id in match_dict:
            match_dict[s1_id] = sorted(cands, key=_candidate_sort_key)

    rows = [{"source1_entity_id": k, "matched_entity_ids": ",".join(v)}
            for k, v in match_dict.items()]
    out_df = pd.DataFrame(rows, columns=["source1_entity_id", "matched_entity_ids"])
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, sep="\t", index=False, na_rep="")

    n_matched = sum(1 for v in match_dict.values() if v)
    n_singletons = len(all_s1) - n_matched
    return {
        "total_s1": len(all_s1),
        "n_singletons": n_singletons,
        "singleton_rate": (n_singletons / len(all_s1)) if all_s1 else 0.0,
        "n_matched": n_matched,
        "mean_matches": (n_above / n_matched) if n_matched else 0.0,
        "t_star": t_star,
        "n_scored": n_scored,
        "output_path": str(output_path),
    }
