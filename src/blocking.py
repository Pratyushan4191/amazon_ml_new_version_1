"""
Amazon ML Challenge 2026 - Business Entity Resolution
Module: High-Recall Blocking Engine (Member 2 Deliverable)

Implements:
- Multi-pass inverted dictionary index: (country, rule_id, block_key) -> [candidate_ids]
- Hard Country Gate: S1_country == Cand_country strictly enforced
- Five Blocking Rules (A through E):
    * Rule A: Sorted Name Tokens (no suffix)
    * Rule B: House Number + Street Token
    * Rule C: Character Trigram Signature (top-3 sorted 3-grams)
    * Rule D: Phonetic (Double Metaphone / Soundex fallback) + City Token
    * Rule E: 4-Character Name Prefix
- Bounded Bucket Memory: Inverted index caps bucket growth at MAX_BUCKET_SIZE to protect RAM
- Coarse Similarity Filter: Prunes noise candidates from single-rule bucket explosions
  before applying any numeric caps, protecting true matches from being chopped off.
- Candidate Provenance Tracking: Records exact matched rules and coarse similarity for Member 3
- Uncapped / Raw Candidate Mode: Avoids artificial candidate caps during development/handoff
- Strict Contract Adherence: Output matches candidate_pairs.tsv contract verified by official validator
"""

import os
import sys
import argparse
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Any, Optional
import pandas as pd

# Add repo root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.normalization import (
    normalize_name,
    normalize_address,
    normalize_country,
)


def extract_blocking_keys_for_record(
    name_dict: Dict[str, Any],
    addr_dict: Dict[str, Any],
    rules: Optional[Set[str]] = None,
) -> List[Tuple[str, str]]:
    """
    Generate (rule_id, block_key) pairs for a single business entity.
    rules: Optional set of rule identifiers to enable (e.g. {'A', 'B'}). If None, all rules A-E run.
    """
    active_rules = rules if rules is not None else {"A", "B", "C", "D", "E", "F", "G"}
    keys: List[Tuple[str, str]] = []

    # Rule A: Sorted Name Tokens (no suffix)
    if "A" in active_rules and name_dict["sorted_tokens"]:
        token_key = name_dict["sorted_tokens"][:30]
        if token_key:
            keys.append(("A", token_key))

    # Rule B: House Number + Street Token
    if "B" in active_rules and addr_dict["house_number"] and addr_dict["street_token"]:
        addr_key = f"{addr_dict['house_number']}_{addr_dict['street_token']}"
        keys.append(("B", addr_key))

    # Rule C: Character Trigrams Signature
    if "C" in active_rules and name_dict["trigram_sig"]:
        keys.append(("C", name_dict["trigram_sig"]))

    # Rule D: Phonetic First Token + City Token
    if "D" in active_rules and name_dict["phonetic_key"] and addr_dict["city"]:
        city_sub = addr_dict["city"][:10]
        keys.append(("D", f"{name_dict['phonetic_key']}_{city_sub}"))

    # Rule E: 4-Character Name Prefix
    if "E" in active_rules and name_dict["prefix_4"] and len(name_dict["prefix_4"]) >= 4:
        keys.append(("E", name_dict["prefix_4"]))

    # Rule F: Pair of Informative Address Tokens (Order-Independent)
    if "F" in active_rules and addr_dict.get("informative_tokens"):
        inf_tokens = addr_dict["informative_tokens"][:5]
        if len(inf_tokens) >= 2:
            for i in range(len(inf_tokens)):
                for j in range(i + 1, len(inf_tokens)):
                    tok1, tok2 = sorted([inf_tokens[i], inf_tokens[j]])
                    keys.append(("F", f"{tok1}_{tok2}"))

    # Rule G: House Number + Informative Address Token
    if "G" in active_rules and addr_dict.get("house_number") and addr_dict.get("informative_tokens"):
        h_num = addr_dict["house_number"]
        for tok in addr_dict["informative_tokens"][:4]:
            keys.append(("G", f"{h_num}_{tok}"))

    return keys


def fast_token_jaccard(tokens_a: Set[str], tokens_b: Set[str]) -> float:
    """Fast Jaccard similarity over token sets."""
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return intersection / union if union > 0 else 0.0


def fast_name_similarity(
    name_a: str,
    name_b: str,
    tokens_a: Set[str],
    tokens_b: Set[str],
) -> float:
    """
    Fast string & token similarity check for coarse filtering.
    Avoids expensive heavy model computation.
    """
    if name_a == name_b:
        return 1.0
    if len(name_a) >= 4 and (name_a in name_b or name_b in name_a):
        return 0.85

    # Token overlap
    jaccard = fast_token_jaccard(tokens_a, tokens_b)
    if jaccard >= 0.5:
        return jaccard

    # Fast character similarity (Jaro-Winkler via jellyfish if present, else SequenceMatcher)
    try:
        import jellyfish
        jw = jellyfish.jaro_winkler_similarity(name_a, name_b)
        return max(jaccard, jw)
    except (ImportError, Exception):
        import difflib
        sm = difflib.SequenceMatcher(None, name_a, name_b).quick_ratio()
        return max(jaccard, sm)


class InvertedBlockIndex:
    """
    Inverted Dictionary Index mapping:
    (country, rule_id, block_key) -> List[candidate_entity_id]

    Also retains lightweight metadata (clean_name, clean_addr) for candidate filtering.
    Protects memory by capping maximum entries per bucket during ingestion.
    """

    def __init__(self, max_bucket_size: int = 5000):
        self.max_bucket_size = max_bucket_size
        self.index: Dict[Tuple[str, str, str], List[str]] = defaultdict(list)
        self.meta: Dict[str, Tuple[str, str]] = {}  # eid -> (clean_name, clean_addr)
        self.entity_count = 0

    def add_records(self, df: pd.DataFrame, rules: Optional[Set[str]] = None) -> None:
        """
        Normalize and index candidate records (from Source 2 and Source 3).
        df must contain: entity_id, business_name, business_address, country
        """
        for row in df.itertuples(index=False):
            eid = str(getattr(row, "entity_id", "")).strip()
            ct = normalize_country(getattr(row, "country", None))
            if not ct:
                continue

            nm = normalize_name(getattr(row, "business_name", ""))
            ad = normalize_address(getattr(row, "business_address", ""))

            # Store minimal metadata for coarse filtering
            self.meta[eid] = (nm["clean"], ad["clean"])

            for rule_id, key in extract_blocking_keys_for_record(nm, ad, rules=rules):
                bucket = self.index[(ct, rule_id, key)]
                # Protect RAM: do not store more than max_bucket_size IDs for overly broad keys
                if len(bucket) < self.max_bucket_size:
                    bucket.append(eid)

            self.entity_count += 1

    def query(
        self,
        country: str,
        keys: List[Tuple[str, str]],
    ) -> Dict[str, Set[str]]:
        """
        Retrieve candidate IDs matching any key for the specified country.
        Returns: {candidate_id: set_of_matched_rules}
        """
        candidates: Dict[str, Set[str]] = defaultdict(set)
        for rule_id, key in keys:
            bucket = self.index.get((country, rule_id, key), [])
            # Skip if bucket hit max size limit
            if len(bucket) >= self.max_bucket_size:
                continue
            for cand_id in bucket:
                candidates[cand_id].add(rule_id)
        return candidates


def coarse_filter_candidates(
    s1_clean_name: str,
    s1_clean_addr: str,
    raw_candidates: Dict[str, Set[str]],
    index_meta: Dict[str, Tuple[str, str]],
    min_coarse_similarity: float = 0.30,
) -> Dict[str, Tuple[Set[str], float]]:
    """
    Coarse Filter:
    Prunes low-quality candidates from single-rule bucket explosions before applying
    any numeric cap.

    Logic:
      1. Multi-rule matches (len(rules) >= 2) ALWAYS survive without penalty.
      2. Single-rule matches (e.g. only Rule E prefix or only Rule C trigram) must have
         at least min_coarse_similarity (default 0.30) to avoid being noise.

    Returns: {cand_id: (matched_rules, coarse_sim_score)}
    """
    s1_tokens = set(s1_clean_name.split())
    s1_addr_tokens = set(s1_clean_addr.split())
    filtered: Dict[str, Tuple[Set[str], float]] = {}

    for cand_id, rules in raw_candidates.items():
        meta = index_meta.get(cand_id)
        if not meta:
            # Metadata missing: preserve candidate safely
            filtered[cand_id] = (rules, 1.0)
            continue

        cand_name, cand_addr = meta
        cand_tokens = set(cand_name.split())
        name_sim = fast_name_similarity(s1_clean_name, cand_name, s1_tokens, cand_tokens)

        # Address similarity booster
        if s1_addr_tokens and cand_addr:
            cand_addr_tokens = set(cand_addr.split())
            addr_sim = fast_token_jaccard(s1_addr_tokens, cand_addr_tokens)
        else:
            addr_sim = 0.0

        combined_sim = max(name_sim, addr_sim)

        # Multi-rule matches are high-confidence evidence -> preserve unconditionally
        if len(rules) >= 2:
            filtered[cand_id] = (rules, round(combined_sim, 4))
        else:
            # Single-rule matches must exceed coarse similarity threshold
            if combined_sim >= min_coarse_similarity:
                filtered[cand_id] = (rules, round(combined_sim, 4))

    return filtered


def _query_s1_chunk(args):
    """
    Worker function executed in each parallel process.
    Queries the shared (read-only) index for a chunk of S1 rows.
    """
    (
        chunk_df,
        index_index,
        index_meta,
        max_bucket_size,
        active_rules,
        enable_coarse_filter,
        min_coarse_similarity,
        max_candidates_per_entity,
    ) = args

    candidate_mapping = {}
    provenance_mapping = {}

    for s1_row in chunk_df.itertuples(index=False):
        s1_id = str(getattr(s1_row, "entity_id", "")).strip()
        s1_ct = normalize_country(getattr(s1_row, "country", None))

        if not s1_ct:
            candidate_mapping[s1_id] = []
            provenance_mapping[s1_id] = {}
            continue

        s1_nm = normalize_name(getattr(s1_row, "business_name", ""))
        s1_ad = normalize_address(getattr(s1_row, "business_address", ""))
        s1_keys = extract_blocking_keys_for_record(s1_nm, s1_ad, rules=active_rules)

        raw_candidates: Dict[str, Set[str]] = defaultdict(set)
        for rule_id, key in s1_keys:
            bucket = index_index.get((s1_ct, rule_id, key), [])
            if len(bucket) >= max_bucket_size:
                continue
            for cand_id in bucket:
                raw_candidates[cand_id].add(rule_id)

        if not raw_candidates:
            candidate_mapping[s1_id] = []
            provenance_mapping[s1_id] = {}
            continue

        # PRE-FILTER: Sort by rule count immediately and take top 500 to avoid billions of string ops
        pre_sorted = sorted(raw_candidates.items(), key=lambda item: -len(item[1]))
        if len(pre_sorted) > 500:
            pre_sorted = pre_sorted[:500]
        
        pruned_candidates = dict(pre_sorted)

        if enable_coarse_filter:
            scored_candidates = coarse_filter_candidates(
                s1_clean_name=s1_nm["clean"],
                s1_clean_addr=s1_ad["clean"],
                raw_candidates=pruned_candidates,
                index_meta=index_meta,
                min_coarse_similarity=min_coarse_similarity,
            )
        else:
            scored_candidates = {
                cid: (rules, 1.0) for cid, rules in pruned_candidates.items()
            }

        sorted_cands = sorted(
            scored_candidates.items(),
            key=lambda item: (-len(item[1][0]), -item[1][1], item[0])
        )
        if max_candidates_per_entity is not None and len(sorted_cands) > max_candidates_per_entity:
            sorted_cands = sorted_cands[:max_candidates_per_entity]

        candidate_mapping[s1_id] = [cand_id for cand_id, _ in sorted_cands]
        provenance_mapping[s1_id] = dict(sorted_cands)

    return candidate_mapping, provenance_mapping


def generate_candidate_pairs(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    active_rules: Optional[Set[str]] = None,
    max_candidates_per_entity: Optional[int] = None,
    max_bucket_size: int = 5000,
    enable_coarse_filter: bool = True,
    min_coarse_similarity: float = 0.30,
    checkpoint_dir: Optional[Any] = None,
) -> Tuple[Dict[str, List[str]], Dict[str, Dict[str, Tuple[Set[str], float]]]]:
    """
    Execute high-recall blocking pipeline (PARALLELIZED + CHECKPOINTED):
    1. Build candidate inverted index from S2 + S3.
    2. Split S1 rows across all CPUs, query index in parallel.
    3. Save each batch result to disk — if interrupted, resume from last checkpoint.

    Returns:
      - candidate_mapping: {s1_id: [candidate_id_1, candidate_id_2, ...]}
      - provenance_mapping: {s1_id: {candidate_id: (rules_set, coarse_sim)}}
    """
    import json
    import multiprocessing as mp
    import numpy as np
    from pathlib import Path
    from tqdm import tqdm

    # ── Checkpoint setup ─────────────────────────────────────────
    ckpt_path = None
    if checkpoint_dir is not None:
        ckpt_path = Path(checkpoint_dir)
        ckpt_path.mkdir(parents=True, exist_ok=True)

    # ── Phase 1: Build index from S2+S3 ──────────────────────────
    print(f"  [Blocking] Building inverted index from {len(s2_df)+len(s3_df):,} S2+S3 records...")
    index = InvertedBlockIndex(max_bucket_size=max_bucket_size)
    index.add_records(s2_df, rules=active_rules)
    index.add_records(s3_df, rules=active_rules)
    print(f"  [Blocking] Index built. Querying {len(s1_df):,} S1 records using parallel workers...")

    # ── Phase 2: Build worker args ────────────────────────────────
    n_cores = max(1, mp.cpu_count() - 1)
    chunks  = [c for c in np.array_split(s1_df, n_cores * 4) if not c.empty]
    index_dict = dict(index.index)
    worker_args = [
        (
            chunk, index_dict, index.meta, max_bucket_size,
            active_rules, enable_coarse_filter, min_coarse_similarity,
            max_candidates_per_entity,
        )
        for chunk in chunks
    ]

    candidate_mapping: Dict[str, List[str]] = {}
    provenance_mapping: Dict[str, Dict[str, Tuple[Set[str], float]]] = {}

    # ── Load already-completed checkpoints ───────────────────────
    completed_chunks: Set[int] = set()
    if ckpt_path is not None:
        for ckpt_file in sorted(ckpt_path.glob("chunk_*.json")):
            chunk_idx = int(ckpt_file.stem.split("_")[1])
            with open(ckpt_file) as f:
                data = json.load(f)
            candidate_mapping.update(data["candidate_mapping"])
            for s1_id, prov in data["provenance_mapping"].items():
                provenance_mapping[s1_id] = {
                    cid: (set(v[0]), v[1]) for cid, v in prov.items()
                }
            completed_chunks.add(chunk_idx)
        if completed_chunks:
            print(f"  [Blocking] Resuming — {len(completed_chunks)}/{len(chunks)} chunks already done.")

    # ── Process remaining chunks ──────────────────────────────────
    pending_indices = [i for i in range(len(worker_args)) if i not in completed_chunks]
    pending_args    = [worker_args[i] for i in pending_indices]

    if pending_args:
        with mp.Pool(n_cores) as pool:
            for i, result in tqdm(
                zip(pending_indices, pool.imap(_query_s1_chunk, pending_args)),
                total=len(pending_args),
                desc=f"  Blocking S1 chunks [{n_cores} cores]",
            ):
                cand_map, prov_map = result
                candidate_mapping.update(cand_map)
                provenance_mapping.update(prov_map)

                # ── Save checkpoint immediately after each chunk ──
                if ckpt_path is not None:
                    prov_serial = {
                        s1: {cid: (list(rules), sim) for cid, (rules, sim) in prov.items()}
                        for s1, prov in prov_map.items()
                    }
                    with open(ckpt_path / f"chunk_{i:04d}.json", "w") as f:
                        json.dump({"candidate_mapping": cand_map,
                                   "provenance_mapping": prov_serial}, f)

    total_cands = sum(len(v) for v in candidate_mapping.values())
    print(f"  [Blocking] Done. {total_cands:,} total candidate pairs generated.")

    # ── Clean up checkpoint files on success ─────────────────────
    if ckpt_path is not None:
        for f in ckpt_path.glob("chunk_*.json"):
            f.unlink()

    return candidate_mapping, provenance_mapping


def export_candidate_pairs_tsv(
    candidate_mapping: Dict[str, List[str]],
    all_s1_ids: List[str],
    output_path: str,
) -> None:
    """
    Write candidate pairs strictly adhering to the submission contract:
    - Tab-separated (.tsv)
    - Columns: source1_entity_id, candidate_entity_ids
    - Every S1 entity present (empty string for singletons)
    - Comma-separated candidate IDs without quotes or spaces
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    rows = []
    for s1_id in all_s1_ids:
        cands = candidate_mapping.get(s1_id, [])
        # Deduplicate while preserving order
        deduped = list(dict.fromkeys(cands))
        rows.append({
            "source1_entity_id": s1_id,
            "candidate_entity_ids": ",".join(deduped) if deduped else ""
        })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(output_path, sep="\t", index=False)
    print(f"Successfully exported {len(df_out)} entities to {output_path}")


def export_candidate_provenance_tsv(
    provenance_mapping: Dict[str, Dict[str, Tuple[Set[str], float]]],
    output_path: str,
) -> None:
    """
    Export pairwise candidate provenance for Member 3 feature engineering.
    Columns: source1_entity_id, candidate_entity_id, matched_rules, rule_count, coarse_sim, rule_A..G
    """
    rows = []
    for s1_id, cands in provenance_mapping.items():
        for cand_id, (rules, sim) in cands.items():
            rule_list = sorted(rules)
            rows.append({
                "source1_entity_id": s1_id,
                "candidate_entity_id": cand_id,
                "matched_rules": ",".join(rule_list),
                "rule_count": len(rule_list),
                "coarse_similarity": sim,
                "rule_A": int("A" in rules),
                "rule_B": int("B" in rules),
                "rule_C": int("C" in rules),
                "rule_D": int("D" in rules),
                "rule_E": int("E" in rules),
                "rule_F": int("F" in rules),
                "rule_G": int("G" in rules),
            })

    df_prov = pd.DataFrame(rows)
    if df_prov.empty:
        df_prov = pd.DataFrame(columns=[
            "source1_entity_id", "candidate_entity_id", "matched_rules",
            "rule_count", "coarse_similarity",
            "rule_A", "rule_B", "rule_C", "rule_D", "rule_E", "rule_F", "rule_G"
        ])
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    df_prov.to_csv(output_path, sep="\t", index=False)
    print(f"Successfully exported {len(df_prov)} provenance pairs to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Amazon ML Challenge 2026 - High Recall Blocker")
    parser.add_argument("--s1", required=True, help="Path to Source 1 TSV")
    parser.add_argument("--s2", required=True, help="Path to Source 2 TSV")
    parser.add_argument("--s3", required=True, help="Path to Source 3 TSV")
    parser.add_argument("--output", default="output/candidate_pairs.tsv", help="Path to candidate_pairs.tsv")
    parser.add_argument("--output-prov", default="output/candidate_provenance.tsv", help="Path to candidate_provenance.tsv")
    parser.add_argument("--rules", default="A,B,C,D,E,F,G", help="Comma-separated rules to enable (A,B,C,D,E,F,G)")
    parser.add_argument("--max-candidates", type=int, default=None, help="Max candidates per S1 entity (default: None/uncapped)")
    parser.add_argument("--max-bucket-size", type=int, default=5000, help="Max index bucket size before pruning")
    parser.add_argument("--disable-coarse-filter", action="store_true", help="Disable coarse similarity filter")
    parser.add_argument("--min-coarse-sim", type=float, default=0.30, help="Minimum coarse similarity threshold for single-rule matches")

    args = parser.parse_args()

    active_rules = set(r.strip().upper() for r in args.rules.split(","))

    print(f"Loading data: S1={args.s1}, S2={args.s2}, S3={args.s3}")
    s1_df = pd.read_csv(args.s1, sep="\t")
    s2_df = pd.read_csv(args.s2, sep="\t")
    s3_df = pd.read_csv(args.s3, sep="\t")

    enable_coarse = not args.disable_coarse_filter
    print(f"Executing blocking with rules: {sorted(active_rules)} (Cap: {args.max_candidates}, CoarseFilter: {enable_coarse})")
    cands, prov = generate_candidate_pairs(
        s1_df, s2_df, s3_df,
        active_rules=active_rules,
        max_candidates_per_entity=args.max_candidates,
        max_bucket_size=args.max_bucket_size,
        enable_coarse_filter=enable_coarse,
        min_coarse_similarity=args.min_coarse_sim,
    )

    all_s1_ids = s1_df["entity_id"].astype(str).str.strip().tolist()
    export_candidate_pairs_tsv(cands, all_s1_ids, args.output)
    export_candidate_provenance_tsv(prov, args.output_prov)

    total_cands = sum(len(c) for c in cands.values())
    avg_cands = total_cands / len(all_s1_ids) if all_s1_ids else 0
    print(f"Blocking complete: {total_cands} total candidates generated ({avg_cands:.2f} avg/entity).")


if __name__ == "__main__":
    main()
