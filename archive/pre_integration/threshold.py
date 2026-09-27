import json, numpy as np, pandas as pd
from pathlib import Path
from eval import macro_f05
def sweep_threshold(val_preds_path, ground_truth_path, threshold_output_path, sweep_start=0.01, sweep_stop=0.99, sweep_step=0.01, candidate_col="candidate_entity_id", candidate_pairs_path=None):
    """
    P1 fix (FINAL_PRODUCTION_READINESS_REPORT.md §12): PREVIOUSLY, the
    threshold sweep's population (`val_s1_ids`, and therefore `truth_dict`)
    was built only from `val_df["source1_entity_id"].unique()` -- but
    `load_candidate_pairs()` (src/features.py) drops every S1 entity with
    zero blocked candidates BEFORE feature computation, so those entities
    never reach `val_df` at all. They were entirely invisible to the sweep,
    even though the real submission metric (see `run_inference`'s
    `all_s1_ids` handling) scores every S1 entity, including ones with no
    candidates (which always get an empty prediction: F0.5=0 if they have
    a true match, F0.5=1 if they're a true singleton).

    FIX: if `candidate_pairs_path` is given (the full Stage-3 output for
    this split, which -- unlike the post-`load_candidate_pairs` feature
    matrix -- still contains a row for every S1 entity, with an empty
    `candidate_entity_ids` field for zero-candidate ones), those
    zero-candidate S1 ids are added to the sweep's population with an
    always-empty prediction set at every threshold, exactly matching how
    `run_inference` already treats them for the real submission. This does
    NOT change the F0.5 formula, singleton semantics, or which candidates
    are scored for entities that DO have candidates -- it only widens the
    population `truth_dict`/`pred_dict` are built over.

    Note on argmax invariance (worth stating plainly, not hiding): because
    each zero-candidate entity's F0.5 contribution is identical at every
    threshold (its prediction is always the empty set, since there is
    nothing to threshold), and macro_f05 is a population mean, adding a
    fixed set of threshold-invariant entities to the population is a
    POSITIVE affine transform of the sweep's f05(t) curve (see
    FIX_CHANGELOG.md for the derivation). A positive affine transform does
    not change where a curve is maximized -- so this fix corrects
    `max_f05` (the reported quality number) to no longer be optimistic
    relative to the true population, but in this codebase's exact metric
    structure it will typically NOT change `t_star` itself. Both effects
    are real and worth having: the reported number should be trustworthy
    even where it happens not to move the selected threshold.

    Backward compatible: if `candidate_pairs_path` is omitted, behavior is
    identical to before this fix (population = val_df's S1 ids only).
    """
    val_df = pd.read_parquet(val_preds_path)
    gt_df = pd.read_csv(ground_truth_path, sep="\t", dtype=str, keep_default_na=False)
    if candidate_col not in val_df.columns and "candidate_id" in val_df.columns: val_df = val_df.rename(columns={"candidate_id": candidate_col})
    val_s1_ids = set(val_df["source1_entity_id"].astype(str).unique().tolist())

    zero_candidate_ids = set()
    if candidate_pairs_path is not None:
        cp_df = pd.read_csv(candidate_pairs_path, sep="\t", dtype=str, keep_default_na=False)
        zero_candidate_ids = set(
            cp_df.loc[
                cp_df["candidate_entity_ids"].astype(str).str.len() == 0,
                "source1_entity_id",
            ].astype(str)
        )
        # Defensive: these two populations should already be disjoint by
        # construction (an S1 id either has candidates, ending up in
        # val_df, or has none, ending up here -- never both), but guard
        # against double-counting explicitly rather than assuming it.
        zero_candidate_ids -= val_s1_ids

    full_population = val_s1_ids | zero_candidate_ids
    gt_map = {str(r["source1_entity_id"]): set(str(r.get("matched_entity_ids","")).split(",")) - {""} for _, r in gt_df.iterrows()}
    truth_dict = {sid: gt_map.get(sid, set()) for sid in full_population}

    thresholds = np.round(np.arange(sweep_start, sweep_stop + sweep_step, sweep_step), 4)
    best_t, max_f = 0.5, 0.0
    for t in thresholds:
        above = val_df[val_df["score"] >= t]
        # zero_candidate_ids always contribute an empty prediction set here,
        # at every threshold -- there is nothing in val_df to threshold for
        # them, matching how run_inference treats a zero-candidate S1 entity
        # for the real submission.
        pred_dict = {sid: set() for sid in full_population}
        for s1_id, grp in above.groupby("source1_entity_id"):
            if s1_id in pred_dict:
                pred_dict[s1_id] = set(grp[candidate_col].tolist())
        f05 = macro_f05(pred_dict, truth_dict)
        if f05 >= max_f: best_t, max_f = float(t), float(f05)

    Path(threshold_output_path).parent.mkdir(parents=True, exist_ok=True)
    result = {
        "t_star": best_t, "max_f05": max_f,
        "n_population": len(full_population),
        "n_with_candidates": len(val_s1_ids),
        "n_zero_candidates_included": len(zero_candidate_ids),
    }
    with open(threshold_output_path, "w") as f: json.dump(result, f, indent=2)
    return result
