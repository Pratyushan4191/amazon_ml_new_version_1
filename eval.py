from __future__ import annotations
import numpy as np, pandas as pd
from typing import Dict, Set

def f05_score(pred: Set[str], truth: Set[str]) -> float:
    if not truth: return 1.0 if not pred else 0.0
    if not pred: return 0.0
    tp = len(pred & truth)
    p, r = tp / len(pred), tp / len(truth)
    return (1.25 * p * r) / (0.25 * p + r) if (p > 0 or r > 0) else 0.0

def macro_f05(pred_dict: Dict[str, Set[str]], truth_dict: Dict[str, Set[str]]) -> float:
    scores = [f05_score(pred_dict.get(s1, set()), truth) for s1, truth in truth_dict.items()]
    return float(np.mean(scores)) if scores else 0.0

def measure_blocking_recall(candidate_mapping: Dict[str, list], ground_truth_path: str) -> Dict[str, float]:
    gt_df = pd.read_csv(ground_truth_path, sep="\t", dtype=str, keep_default_na=False)
    true_pairs, gt_map = set(), {}
    for _, row in gt_df.iterrows():
        s1_id = str(row["source1_entity_id"]).strip()
        m = str(row.get("matched_entity_ids", "")).strip()
        matches = {x.strip() for x in m.split(",") if x.strip()}
        gt_map[s1_id] = matches
        for cid in matches: true_pairs.add((s1_id, cid))
    covered = sum(len(truth & set(candidate_mapping.get(s1_id, []))) for s1_id, truth in gt_map.items())
    return {"recall": covered / len(true_pairs) if true_pairs else 1.0, "n_true_pairs": len(true_pairs), "n_covered": covered}
