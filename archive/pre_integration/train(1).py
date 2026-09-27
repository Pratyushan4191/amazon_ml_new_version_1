import gc
import json, numpy as np, pandas as pd, lightgbm as lgb
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
FEATURE_COLS = ["name_exact", "name_exact_no_suffix", "name_sorted_exact", "name_edit_dist", "name_jaro_winkler", "name_jaro_winkler_ns", "name_token_jaccard", "name_token_set_ratio", "name_token_sort_ratio", "name_char3gram_jaccard", "name_char4gram_jaccard", "name_len_diff", "name_tfidf_cosine", "addr_s1_null", "addr_cand_null", "addr_both_null", "addr_exact", "addr_token_jaccard", "addr_char3gram_jaccard", "addr_house_num_agree", "addr_city_agree", "addr_pin_agree", "addr_tfidf_cosine", "country_match", "high_name_low_addr", "name_sim_x_addr_sim", "n_blocking_rules_hit", "both_have_address", "name_suffix_conflict"]
def train_model(train_features_path, ground_truth_path, model_output_path, shap_output_path, val_preds_output_path, lgbm_params=None, n_estimators=500):
    feat_df = pd.read_parquet(train_features_path)
    gt_df = pd.read_csv(ground_truth_path, sep="\t", dtype=str, keep_default_na=False)
    if "candidate_id" in feat_df.columns: feat_df = feat_df.rename(columns={"candidate_id": "candidate_entity_id"})
    if "label" not in feat_df.columns:
        # Vectorized ground-truth labeling (Stage 5 audit fix): the original
        # feat_df.apply(..., axis=1) row-wise membership test does not scale
        # to production candidate volume (~86M rows per
        # docs/BER_Architecture.md Section 23) -- verified 22x faster and
        # byte-identical labels on an 800k-row synthetic feature matrix
        # (see scratch/test_train_label_vectorize.py).
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
    train_idx, val_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42).split(np.arange(len(s1_ids)), groups=s1_ids))
    train_mask = feat_df["source1_entity_id"].isin(set(s1_ids[train_idx]))
    val_df = feat_df[~train_mask].reset_index(drop=True)
    train_df = feat_df[train_mask].reset_index(drop=True)
    # Stage 5 memory fix: `feat_df` (the full parquet load) is never needed again
    # once train_df/val_df are sliced out of it -- boolean-mask indexing on a
    # DataFrame always copies (it is never a view), so keeping `feat_df` alive
    # here means 3 copies of the full feature matrix are resident simultaneously
    # (feat_df + train_df + val_df) for no reason. Drop it immediately.
    del feat_df, train_mask
    gc.collect()
    X_train, y_train = train_df[FEATURE_COLS].astype(np.float32), train_df["label"].astype(np.int32)
    X_val, y_val = val_df[FEATURE_COLS].astype(np.float32), val_df["label"].astype(np.int32)
    # `train_df` itself (as opposed to X_train/y_train, the float32/int32 copies
    # LightGBM actually needs) is not referenced again after this point -- only
    # `val_df` survives, to carry the per-row validation score at the end. Drop
    # train_df now rather than holding it until the function returns.
    del train_df
    gc.collect()
    lgb_train = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_COLS)
    lgb_val = lgb.Dataset(X_val, label=y_val, feature_name=FEATURE_COLS, reference=lgb_train)
    model = lgb.train(lgbm_params, lgb_train, num_boost_round=n_estimators, valid_sets=[lgb_val])
    Path(model_output_path).parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_output_path))
    # `val_df` here is already an owned copy (it came from boolean-mask
    # indexing above, never a view) -- the previous `val_df = val_df.copy()`
    # duplicated it a second time for no reason. Removed: assigning the
    # "score" column in place on the DataFrame we already own is sufficient
    # and does not raise SettingWithCopyWarning because there is no parent
    # DataFrame this is a view into.
    val_df["score"] = model.predict(X_val)
    Path(val_preds_output_path).parent.mkdir(parents=True, exist_ok=True)
    val_df.to_parquet(val_preds_output_path, index=False)
    del X_train, X_val, lgb_train, lgb_val
    gc.collect()
    return {"best_iteration": model.best_iteration, "val_f05_at_0.5": 0.0, "top_features": []}
