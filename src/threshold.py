"""
Stage 6 -- threshold sweep for macro-F0.5.

WHY THIS WAS REWRITTEN
----------------------
The sweep evaluates ~99 thresholds. The previous implementation evaluated each
one with

    above = val_df[val_df["score"] >= t]                 # full boolean mask
    for s1_id, grp in above.groupby("source1_entity_id"): # full groupby
        pred_dict[s1_id] = set(grp[candidate_col])       # Python set build
    f05 = macro_f05(pred_dict, truth_dict)                # Python set intersects

i.e. 99 passes over the entire validation frame, 99 groupbys, and 99 * n_s1
Python-level set intersections. At the ~20K-entity scale used in
bench/memory_benchmark.py that already costs 11-22s; at production, with
~2.2M entities and tens of millions of validation rows, it is minutes of pure
overhead and it is the reason this stage is worth touching at all.

The key observation: F0.5 depends on the prediction set only through two
counts --

    |pred|        and        |pred INTERSECT truth|

    f05 = 1.25 * p * r / (0.25 * p + r),  p = tp/|pred|,  r = tp/|truth|

so the sets themselves never have to be built. Both counts are thresholded
counts, and because an entity's candidates are sorted by score once, the
candidates above any threshold form a PREFIX of that entity's sorted rows. So
for every threshold:

    |pred|(e, t)       = #rows of e with score >= t
    |pred INTERSECT T|(e, t) = #TRUE rows of e with score >= t

which are prefix lengths, computable for every entity at once with a sort, a
cumsum and a difference at the group boundaries. The whole sweep becomes a
handful of numpy operations per threshold instead of a groupby and a Python
set intersection per entity.

The population fix (zero-candidate S1 entities must be scored) from the
promoted implementation is preserved exactly, including the reported
population counts.

Equivalence with the previous implementation is pinned by
tests/test_threshold_sweep.py.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from eval import macro_f05  # noqa: F401  (kept for API compatibility; vectorised below)


def _f05_from_counts(n_pred, n_true, tp):
    """
    Vectorised `f05_score`, matching eval.f05_score branch for branch.

    n_pred, n_true, tp are integer arrays. Returned as float64.

    Branches, in the same order as the scalar original:
      * truth empty  -> 1.0 if pred empty else 0.0
      * pred empty   -> 0.0            (implied by the next line, but kept so
                                       the no-truth/no-pred case is explicit)
      * otherwise    -> 1.25*p*r / (0.25*p + r), or 0.0 when p == r == 0
    """
    n_pred = np.asarray(n_pred, dtype=np.float64)
    n_true = np.asarray(n_true, dtype=np.float64)
    tp = np.asarray(tp, dtype=np.float64)

    # Normalise each array independently to 2-D (n_entities, n_thresholds).
    # n_true arrives per-entity only, so it needs the threshold axis too --
    # and it must be broadcast by the same rule as the others, or a
    # (n_entities,) truth array broadcasts against an (n_entities, n_thresh)
    # prediction array into an (n_entities, n_entities) mess.
    if n_pred.ndim == 1:
        n_pred = n_pred[:, None]
    if tp.ndim == 1:
        tp = tp[:, None]
    if n_true.ndim == 1:
        n_true = n_true[:, None]

    out = np.zeros(tp.shape, dtype=np.float64)

    truth_empty = n_true == 0
    pred_empty = n_pred == 0

    # truth empty: perfect if we also predicted nothing, else zero
    out[truth_empty & pred_empty] = 1.0
    out[truth_empty & ~pred_empty] = 0.0

    scored = ~truth_empty
    # pred empty (with truth non-empty) stays 0.0
    both = scored & ~pred_empty
    if both.any():
        p = np.zeros(tp.shape, dtype=np.float64)
        r = np.zeros(tp.shape, dtype=np.float64)
        np.divide(tp, n_pred, out=p, where=n_pred > 0)
        np.divide(tp, n_true, out=r, where=n_true > 0)
        denom = 0.25 * p + r
        val = np.zeros(tp.shape, dtype=np.float64)
        ok = (p > 0) | (r > 0)
        np.divide(1.25 * p * r, denom, out=val, where=ok & (denom > 0))
        out[both] = val[both]
    return out


def _group_bounds(ent_sorted):
    """
    Start (inclusive) / end (exclusive) row index of each entity's contiguous
    block, plus the entity code each block belongs to.
    """
    n_rows = len(ent_sorted)
    if n_rows == 0:
        z = np.zeros(0, dtype=np.int64)
        return z, z, z
    is_start = np.empty(n_rows, dtype=bool)
    is_start[0] = True
    if n_rows > 1:
        np.not_equal(ent_sorted[1:], ent_sorted[:-1], out=is_start[1:])
    starts = np.nonzero(is_start)[0]
    ends = np.empty(len(starts), dtype=np.int64)
    ends[:-1] = starts[1:]
    ends[-1] = n_rows
    return starts, ends, ent_sorted[starts]


def _sweep_counts(scores_sorted, ent_sorted, n_entities, thresholds):
    """
    |{rows of e with score >= t}| for every entity e and threshold t.

    Returns int64 of shape (n_entities, len(thresholds)).

    MEMORY: this deliberately accumulates ONE THRESHOLD AT A TIME. The obvious
    vectorisation -- build the whole (n_rows, n_thresholds) keep-mask and cumsum
    it along axis 0 -- needs an int64 that is n_rows x n_thresholds, i.e. 1.4GB
    at 1.75M validation rows and ~34GB at production's ~43M, and the vstack to
    pad it doubles that. That would have reintroduced exactly the OOM this
    rewrite was meant to remove. Here the only large temporary is a single
    1-D cumsum of n_rows, so peak is O(n_rows) regardless of how many
    thresholds are swept.
    """
    n_t = len(thresholds)
    out = np.zeros((n_entities, n_t), dtype=np.int64)
    starts, ends, group_entities = _group_bounds(ent_sorted)
    if len(starts) == 0:
        return out

    n_groups = len(starts)
    block = np.empty(n_groups, dtype=np.int64)
    # Value to subtract per group: 0 for the first group (whose start is row 0
    # and so has no preceding row), cum[start-1] for the rest. `starts` is
    # strictly increasing and starts[0] == 0, so starts[k] >= 1 for every k > 0.
    prev = np.zeros(n_groups, dtype=np.int64)
    for j, t in enumerate(thresholds):
        # np.cumsum gives the INCLUSIVE prefix count: cum[i] = # kept in [0, i].
        # A group spans [starts[k], ends[k]), so its kept count is
        # cum[ends[k]-1] - cum[starts[k]-1].
        cum = np.cumsum(scores_sorted >= t, dtype=np.int64)
        if n_groups > 1:
            prev[1:] = cum[starts[1:] - 1]
        np.subtract(cum[ends - 1], prev, out=block)
        out[group_entities, j] = block
    return out


def sweep_threshold(
    val_preds_path,
    ground_truth_path,
    threshold_output_path,
    sweep_start=0.01,
    sweep_stop=0.99,
    sweep_step=0.01,
    candidate_col="candidate_entity_id",
    candidate_pairs_path=None,
):
    """
    Choose the score threshold maximising macro-F0.5 on the validation split.

    `candidate_pairs_path` widens the scored population to include S1 entities
    that blocking produced zero candidates for. Those entities never reach the
    feature matrix (there is nothing to featurise), but the real submission
    scores every S1 entity, and an entity with no candidates always gets an
    empty prediction -- F0.5 = 1.0 if it is a true singleton, 0.0 if it has a
    true match. Leaving them out made `max_f05` optimistic relative to the
    population the submission is actually judged on.

    Note on the effect: each such entity contributes the same value at every
    threshold, and macro_f05 is a population mean, so adding a fixed set of
    threshold-invariant entities is a positive affine transform of the f05(t)
    curve. A positive affine transform does not move its argmax, so this
    corrects the reported `max_f05` without typically moving `t_star`. Both are
    worth having: the reported number should describe the real population.
    """
    # Read ONLY the three columns the sweep needs. The validation parquet also
    # carries all 29 feature columns plus the label (train_model writes the
    # whole frame), and loading them just to read a score column cost more
    # memory than any other stage: measured peak was 3311MB against 1664MB for
    # training, for a computation that only needs
    # (source1_entity_id, candidate_entity_id, score).
    import pyarrow.parquet as pq

    _available = set(pq.ParquetFile(val_preds_path).schema_arrow.names)
    _cand_col = (candidate_col if candidate_col in _available
                 else ("candidate_id" if "candidate_id" in _available
                       else "candidate_entity_id"))
    _read = ["source1_entity_id", _cand_col, "score"]
    val_df = pd.read_parquet(val_preds_path, columns=_read)
    if candidate_col not in val_df.columns and "candidate_id" in val_df.columns:
        val_df = val_df.rename(columns={"candidate_id": candidate_col})

    gt_df = pd.read_csv(ground_truth_path, sep="\t", dtype=str,
                        keep_default_na=False)

    val_s1_ids = set(val_df["source1_entity_id"].astype(str).unique().tolist())

    # Zero-candidate S1 ids: present in the blocking output, but with an empty
    # candidate list. Read in chunks so the full Stage-3 TSV is never resident.
    zero_candidate_ids = set()
    if candidate_pairs_path is not None:
        for chunk in pd.read_csv(
            candidate_pairs_path, sep="\t", dtype=str, keep_default_na=False,
            usecols=["source1_entity_id", "candidate_entity_ids"], chunksize=500_000,
        ):
            zero_candidate_ids.update(
                chunk.loc[
                    chunk["candidate_entity_ids"].astype(str).str.len() == 0,
                    "source1_entity_id",
                ].astype(str).tolist()
            )
        # Should already be disjoint by construction; guard rather than assume.
        zero_candidate_ids -= val_s1_ids

    full_population = val_s1_ids | zero_candidate_ids
    gt_map = {
        str(r["source1_entity_id"]): set(
            str(r.get("matched_entity_ids", "")).split(",")) - {""}
        for _, r in gt_df.iterrows()
    }

    thresholds = np.round(np.arange(sweep_start, sweep_stop + sweep_step, sweep_step), 4)
    n_t = len(thresholds)

    # Entities are scored in a stable order so the reported numbers do not
    # depend on set iteration order.
    population = sorted(full_population)
    ent_index = {s: i for i, s in enumerate(population)}
    n_ent = len(population)

    n_true = np.zeros(n_ent, dtype=np.int64)
    for s in population:
        n_true[ent_index[s]] = len(gt_map.get(s, set()))

    # Rows of val_df that belong to the scored population.
    row_ent = val_df["source1_entity_id"].astype(str).map(ent_index)
    keep_row = row_ent.notna().to_numpy()
    row_ent = row_ent[keep_row].to_numpy(dtype=np.int64)
    cand = val_df[candidate_col].astype(str).to_numpy()[keep_row]
    score = val_df["score"].to_numpy(dtype=np.float64)[keep_row]

    # A prediction is a SET, so a (s1, candidate) pair that appears on several
    # rows contributes 1 to |pred|, not one per row. Deduplicate on
    # (entity, candidate), keeping the MAXIMUM score: the pair enters the
    # prediction set at the lowest threshold that any of its rows clears, which
    # is exactly what a set built from the full frame would do.
    # (Counting rows instead of distinct pairs inflates |pred| and depresses
    # precision, which biases the sweep toward a lower threshold.)
    if len(row_ent):
        order = np.lexsort((-score, cand, row_ent))   # ent, cand, score DESC
        row_ent, cand, score = row_ent[order], cand[order], score[order]
        first_of_pair = np.empty(len(row_ent), dtype=bool)
        first_of_pair[0] = True
        if len(row_ent) > 1:
            np.logical_or(
                row_ent[1:] != row_ent[:-1],
                cand[1:] != cand[:-1],
                out=first_of_pair[1:],
            )
        row_ent, cand, score = row_ent[first_of_pair], cand[first_of_pair], score[first_of_pair]

    # Which rows are true matches. F0.5 needs only the COUNT of true
    # candidates above each threshold, so this is a membership test, not a
    # set that gets rebuilt 99 times.
    truth_lookup = set()
    for si, s in enumerate(population):
        for c in gt_map.get(s, set()):
            truth_lookup.add((si, c))
    is_true = np.fromiter(
        ((int(e), str(cc)) in truth_lookup for e, cc in zip(row_ent, cand)),
        dtype=bool, count=len(row_ent),
    )

    # n_pred(e, t) over all rows.
    order_all = np.lexsort((-score, row_ent))
    n_pred = _sweep_counts(score[order_all], row_ent[order_all], n_ent, thresholds)

    # tp(e, t) over true rows only.
    tp = np.zeros((n_ent, n_t), dtype=np.int64)
    if is_true.any():
        t_ent, t_score = row_ent[is_true], score[is_true]
        order_t = np.lexsort((-t_score, t_ent))
        tp = _sweep_counts(t_score[order_t], t_ent[order_t], n_ent, thresholds)

    scores = _f05_from_counts(n_pred, n_true, tp)
    per_threshold = scores.mean(axis=0)

    best_t, max_f = 0.5, 0.0
    for i, t in enumerate(thresholds):
        f = float(per_threshold[i])
        if f >= max_f:            # ties resolve to the LAST threshold, as before
            best_t, max_f = float(t), f

    Path(threshold_output_path).parent.mkdir(parents=True, exist_ok=True)
    result = {
        "t_star": best_t, "max_f05": max_f,
        "n_population": len(full_population),
        "n_with_candidates": len(val_s1_ids),
        "n_zero_candidates_included": len(zero_candidate_ids),
    }
    with open(threshold_output_path, "w") as f:
        json.dump(result, f, indent=2)
    return result
