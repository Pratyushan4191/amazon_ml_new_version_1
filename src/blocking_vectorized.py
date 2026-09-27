"""
Amazon ML Challenge 2026 - Business Entity Resolution
Module: OPTION 3 - Vectorized Pandas/NumPy Merge-Based Blocking

This is a NEW module. It does not replace src/blocking.py -- that remains
the frozen baseline implementation (see /home/claude/repo_baseline and the
git tag `16a3237` for the original, untouched code).

DESIGN GOAL (see OPTION3_VECTOR_BLOCKING_DESIGN.md for full rationale):
  replace the OLD architecture --
      Python dict/list/set inverted index
      -> multiprocessing task args embedding the whole index
      -> candidate_mapping / provenance_mapping dicts retained for the
         entire dataset
      -> single export at the very end
  -- with --
      compact tabular (long-form DataFrame) blocking index
      -> bounded S1 chunks
      -> vectorized pandas merge for candidate matching
      -> hot-key protection (whole-bucket exclusion, matching OLD semantics
         exactly -- see _filter_hot_buckets)
      -> per-chunk cap + coarse filter (REUSES the original functions from
         src.blocking so scoring semantics are byte-for-byte identical)
      -> immediate parquet checkpoint + memory release
      -> streaming final TSV export (never materializes the full
         candidate_mapping/provenance_mapping dict for the whole dataset)

CANDIDATE SEMANTICS PRESERVED EXACTLY (verified in
tests/test_blocking_vectorized.py by direct comparison against src/blocking.py):
  - All 7 rules (A-G), reusing src.normalization + src.blocking's own
    extract_blocking_keys_for_record so key generation is identical.
  - Hard country gate.
  - max_bucket_size semantics: a (country, rule, key) bucket whose TRUE
    size is >= max_bucket_size is entirely excluded from candidate
    generation for that rule/key (not truncated-and-kept). This exactly
    matches OLD's `query()`: `if len(bucket) >= max_bucket_size: continue`.
    (OLD's `add_records` also stops appending once a bucket reaches
    max_bucket_size, so a bucket's stored length is always <=
    max_bucket_size; the query-time check then excludes buckets that hit
    that cap. Net effect: any key whose TRUE frequency is >= max_bucket_size
    contributes zero candidates. We reproduce this using true frequency
    directly, which is equivalent and does not need a truncate step.)
  - Pre-filter to top 500 candidates by rule-count before coarse filtering
    (matches OLD's `_query_s1_chunk` pre-sort/truncate step).
  - Coarse filter: REUSES src.blocking.coarse_filter_candidates verbatim.
  - Final ordering/cap: sorted by (-rule_count, -coarse_sim, candidate_id),
    capped at max_candidates_per_entity -- identical tie-break to OLD.
"""

from __future__ import annotations

import gc
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any

import numpy as np
import pandas as pd

from src.normalization import normalize_name, normalize_address, normalize_country
from src.blocking import (
    extract_blocking_keys_for_record,
    coarse_filter_candidates,
)
from src.monitoring import MemoryGuard, MemoryBudgetExceeded, ProgressTracker, MemoryEventLog, take_snapshot

ALL_RULES = {"A", "B", "C", "D", "E", "F", "G"}


# ─────────────────────────────────────────────────────────────
# STEP 1: build a compact long-form key table (replaces the Python dict index)
# ─────────────────────────────────────────────────────────────

def _normalize_df_for_keys(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add clean_name / clean_addr / country columns AND a `keys` column
    (list of (rule, key) tuples) to df, reusing the exact same
    normalize_name / normalize_address / extract_blocking_keys_for_record
    functions the OLD implementation uses -- this is what guarantees
    identical candidate semantics.
    """
    out = df[["entity_id", "business_name", "business_address", "country"]].copy()
    out["country"] = out["country"].map(normalize_country)

    def _row_keys(row):
        nm = normalize_name(row["business_name"])
        ad = normalize_address(row["business_address"])
        keys = extract_blocking_keys_for_record(nm, ad)
        return nm["clean"], ad["clean"], keys

    parsed = out.apply(_row_keys, axis=1, result_type="expand")
    out["clean_name"] = parsed[0]
    out["clean_addr"] = parsed[1]
    out["_keys"] = parsed[2]
    return out


def build_index_table(
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    active_rules: Optional[Set[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build the compact tabular blocking index.

    Returns:
      index_table: long-form DataFrame, one row per (entity_id, rule, key),
                   columns [country, rule, key, entity_id] -- dtype
                   'category' for country/rule to keep it compact.
      meta_table:  DataFrame [entity_id, clean_name, clean_addr] indexed by
                   entity_id -- replaces InvertedBlockIndex.meta.
    """
    combined = pd.concat([s2_df, s3_df], ignore_index=True)
    normed = _normalize_df_for_keys(combined)

    if active_rules is not None:
        normed["_keys"] = normed["_keys"].map(
            lambda ks: [(r, k) for r, k in ks if r in active_rules]
        )

    # explode (entity_id, country) x [(rule,key), ...] into long form
    long_rows = normed[["entity_id", "country", "_keys"]].explode("_keys", ignore_index=True)
    long_rows = long_rows.dropna(subset=["_keys"])
    rk = pd.DataFrame(long_rows["_keys"].tolist(), columns=["rule", "key"])
    index_table = pd.concat(
        [long_rows[["entity_id", "country"]].reset_index(drop=True), rk], axis=1
    )
    index_table["country"] = index_table["country"].astype("category")
    index_table["rule"] = index_table["rule"].astype("category")

    meta_table = normed.set_index("entity_id")[["clean_name", "clean_addr"]]
    return index_table, meta_table


def _filter_hot_buckets(index_table: pd.DataFrame, max_bucket_size: int) -> pd.DataFrame:
    """
    Hot-key protection: drop every (country, rule, key) bucket whose TRUE
    size is >= max_bucket_size, BEFORE any merge is attempted. This
    reproduces OLD's query-time `if len(bucket) >= max_bucket_size:
    continue` semantics exactly (see module docstring), and -- critically
    for memory safety -- it means the subsequent S1 merge can never explode
    into a many-to-many join against an ultra-hot key, because that key's
    rows are already gone from the index table.

    BUG FIXED HERE (pre-existing, and a hard crash in production):
    the previous implementation did

        keep_mask = ~index_table.set_index([...]).index.isin(hot_keys)
        return index_table.loc[keep_mask.values]

    `MultiIndex.isin()` returns a plain numpy.ndarray, not a Series, so
    `keep_mask.values` raised
    `AttributeError: 'numpy.ndarray' object has no attribute 'values'`.
    This only fires once a bucket actually reaches `max_bucket_size`, which no
    small fixture ever produced -- but on the real ~10.3M-record dataset some
    bucket always does, so the pipeline died here. It was latent, not
    theoretical.

    It is now expressed as a per-row `groupby(...).transform("size")` instead,
    which additionally avoids materialising a MultiIndex over every row of the
    index table (the old `set_index([...])` built a full copy of the frame's
    index, on the order of the whole table).
    """
    if index_table.empty:
        return index_table

    bucket_size = index_table.groupby(
        ["country", "rule", "key"], observed=True, sort=False
    )["entity_id"].transform("size")

    # Keep rows whose bucket is strictly smaller than the cap -- identical
    # semantics to OLD's `if len(bucket) >= max_bucket_size: continue`.
    keep = (bucket_size < max_bucket_size).to_numpy()
    if keep.all():
        return index_table
    return index_table.loc[keep].reset_index(drop=True)


# ─────────────────────────────────────────────────────────────
# STEP 2: per-chunk vectorized candidate generation
# ─────────────────────────────────────────────────────────────

def _candidates_for_chunk(
    s1_chunk: pd.DataFrame,
    index_table: pd.DataFrame,
    meta_table: pd.DataFrame,
    active_rules: Optional[Set[str]],
    enable_coarse_filter: bool,
    min_coarse_similarity: float,
    max_candidates_per_entity: Optional[int],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Vectorized replacement for `_query_s1_chunk`. Returns:
      cand_df: [source1_entity_id, candidate_entity_id, candidate_source]  (final, capped)
      prov_df: [source1_entity_id, candidate_entity_id, candidate_source,
                matched_rules, rule_count, coarse_similarity]
    covering every S1 id in s1_chunk (including those with zero candidates,
    which simply produce no rows -- the caller is responsible for filling
    in the "every S1 present" contract when writing the final TSV).
    """
    s1_normed = _normalize_df_for_keys(s1_chunk)
    if active_rules is not None:
        s1_normed["_keys"] = s1_normed["_keys"].map(
            lambda ks: [(r, k) for r, k in ks if r in active_rules]
        )
    s1_long = s1_normed[["entity_id", "country", "_keys"]].explode("_keys", ignore_index=True)
    s1_long = s1_long.dropna(subset=["_keys"])
    if s1_long.empty:
        empty_cols_c = ["source1_entity_id", "candidate_entity_id", "candidate_source"]
        empty_cols_p = empty_cols_c + ["matched_rules", "rule_count", "coarse_similarity"]
        return pd.DataFrame(columns=empty_cols_c), pd.DataFrame(columns=empty_cols_p)
    rk = pd.DataFrame(s1_long["_keys"].tolist(), columns=["rule", "key"])
    s1_keys = pd.concat([s1_long[["entity_id", "country"]].reset_index(drop=True), rk], axis=1)
    s1_keys = s1_keys.rename(columns={"entity_id": "source1_entity_id"})

    # VECTORIZED MERGE (the core of Option 3): inner-join S1's exploded
    # keys against the pre-built, hot-key-filtered index table on
    # (country, rule, key). This is the direct replacement for the OLD
    # per-row Python dict lookup loop.
    merged = s1_keys.merge(
        index_table.rename(columns={"entity_id": "candidate_entity_id"}),
        on=["country", "rule", "key"],
        how="inner",
    )
    if merged.empty:
        empty_cols_c = ["source1_entity_id", "candidate_entity_id", "candidate_source"]
        empty_cols_p = empty_cols_c + ["matched_rules", "rule_count", "coarse_similarity"]
        return pd.DataFrame(columns=empty_cols_c), pd.DataFrame(columns=empty_cols_p)

    # Collapse to (s1, candidate) -> set of matched rules, rule_count.
    grouped = (
        merged.groupby(["source1_entity_id", "candidate_entity_id"], observed=True)["rule"]
        .agg(lambda s: tuple(sorted(set(s))))
        .reset_index()
    )
    grouped["rule_count"] = grouped["rule"].map(len)

    # PRE-FILTER: top 500 by rule_count per S1 (matches OLD's `pre_sorted[:500]`).
    grouped = (
        grouped.sort_values(["source1_entity_id", "rule_count"], ascending=[True, False])
        .groupby("source1_entity_id", observed=True, group_keys=False)
        .head(500)
    )

    if enable_coarse_filter:
        prov_parts = []
        s1_meta = s1_normed.set_index("entity_id")[["clean_name", "clean_addr"]]
        for s1_id, grp in grouped.groupby("source1_entity_id", observed=True):
            raw_candidates: Dict[str, Set[str]] = {
                row.candidate_entity_id: set(row.rule) for row in grp.itertuples(index=False)
            }
            s1_clean_name, s1_clean_addr = s1_meta.loc[s1_id, ["clean_name", "clean_addr"]]
            scored = coarse_filter_candidates(
                s1_clean_name=s1_clean_name,
                s1_clean_addr=s1_clean_addr,
                raw_candidates=raw_candidates,
                index_meta={cid: tuple(meta_table.loc[cid]) for cid in raw_candidates
                            if cid in meta_table.index},
                min_coarse_similarity=min_coarse_similarity,
            )
            for cid, (rules, sim) in scored.items():
                prov_parts.append({
                    "source1_entity_id": s1_id,
                    "candidate_entity_id": cid,
                    "matched_rules": ",".join(sorted(rules)),
                    "rule_count": len(rules),
                    "coarse_similarity": sim,
                })
        prov_df = pd.DataFrame(prov_parts, columns=[
            "source1_entity_id", "candidate_entity_id", "matched_rules",
            "rule_count", "coarse_similarity",
        ])
    else:
        prov_df = grouped.rename(columns={"rule": "matched_rules"}).copy()
        prov_df["matched_rules"] = prov_df["matched_rules"].map(lambda r: ",".join(sorted(r)))
        prov_df["coarse_similarity"] = 1.0
        prov_df = prov_df[["source1_entity_id", "candidate_entity_id", "matched_rules",
                            "rule_count", "coarse_similarity"]]

    if prov_df.empty:
        empty_cols_c = ["source1_entity_id", "candidate_entity_id", "candidate_source"]
        empty_cols_p = empty_cols_c + ["matched_rules", "rule_count", "coarse_similarity"]
        return pd.DataFrame(columns=empty_cols_c), pd.DataFrame(columns=empty_cols_p)

    # Final ordering + cap: (-rule_count, -coarse_sim, candidate_id) -- identical tie-break to OLD.
    prov_df = prov_df.sort_values(
        ["source1_entity_id", "rule_count", "coarse_similarity", "candidate_entity_id"],
        ascending=[True, False, False, True],
    )
    if max_candidates_per_entity is not None:
        prov_df = prov_df.groupby("source1_entity_id", observed=True, group_keys=False).head(
            max_candidates_per_entity
        )

    prov_df["candidate_source"] = np.where(
        prov_df["candidate_entity_id"].str.startswith("S2-"), "S2", "S3"
    )
    cand_df = prov_df[["source1_entity_id", "candidate_entity_id", "candidate_source"]].copy()
    prov_df = prov_df[["source1_entity_id", "candidate_entity_id", "candidate_source",
                        "matched_rules", "rule_count", "coarse_similarity"]]
    return cand_df.reset_index(drop=True), prov_df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────
# STEP 3: chunked, checkpointed, streaming driver
# ─────────────────────────────────────────────────────────────

def _chunk_bounds(n: int, chunk_size: int) -> List[Tuple[int, int]]:
    return [(i, min(i + chunk_size, n)) for i in range(0, n, chunk_size)]


class CheckpointConfigError(RuntimeError):
    """
    Raised when a blocking checkpoint cannot be safely resumed under the
    configuration/input of the current run.

    This exists because the failure it guards against is silent. The manifest
    records chunk INDICES, so resuming with a different `chunk_size` re-bounds
    the S1 rows: chunk 7 of the old run covered S1 rows 7000-7999, while chunk
    7 of the new run covers rows 14000-14999. Nothing raises, the pipeline
    completes, and the candidate set is quietly wrong.
    """


def _input_fingerprint(
    s1_df: pd.DataFrame,
    index_table: pd.DataFrame,
    meta_table: Optional[pd.DataFrame] = None,
    impl_version: str = "vectorized_v2.1",
    rule_version: str = "rules_v2",
) -> str:
    """
    Comprehensive, fast, stable digest of the exact inputs and rules a blocking run depends on.

    Covers:
      - S1 entity IDs and content (business_name, country, address)
      - S2 and S3 index keys and entity IDs (built from S2 + S3 source records)
      - Metadata table dimensions and content (if provided)
      - Implementation and rule versions
    """
    # 1. S1 content hash
    s1_parts = [s1_df["entity_id"].astype(str)]
    for col in ("business_name", "country", "business_address"):
        if col in s1_df.columns:
            s1_parts.append(s1_df[col].fillna("").astype(str))
    s1_h = 0
    for part in s1_parts:
        s1_h = (s1_h * 31 + pd.util.hash_pandas_object(part, index=False).sum()) & 0xFFFFFFFFFFFFFFFF

    # 2. S2/S3 index content hash (captures S2 and S3 blocking keys & entities)
    idx_h = 0
    for col in ("rule", "key", "entity_id", "country"):
        if col in index_table.columns:
            idx_h = (idx_h * 31 + pd.util.hash_pandas_object(index_table[col].astype(str), index=False).sum()) & 0xFFFFFFFFFFFFFFFF

    # 3. Meta table hash (if available)
    meta_h = 0
    if meta_table is not None and len(meta_table) > 0:
        for col in ("country", "source"):
            if col in meta_table.columns:
                meta_h = (meta_h * 31 + pd.util.hash_pandas_object(meta_table[col].astype(str), index=False).sum()) & 0xFFFFFFFFFFFFFFFF

    return (
        f"n_s1={len(s1_df)};s1_hash={int(s1_h)};"
        f"n_index={len(index_table)};index_hash={int(idx_h)};"
        f"meta_hash={int(meta_h)};impl={impl_version};rules={rule_version}"
    )


# ─────────────────────────────────────────────────────────────
# PARALLEL CHUNK EXECUTION
# ─────────────────────────────────────────────────────────────
#
# MEASURED (bench/profile_stages.py, 20K S1, 8 cores): stage 3b took 367.6s
# and used exactly one core. The chunk loop is trivially parallel -- each chunk
# is an independent `_candidates_for_chunk` call over disjoint S1 rows against
# a read-only index -- so on a 64-core instance this stage was leaving 63 cores
# idle.
#
# How the index reaches the workers matters as much as the parallelism does.
# `index_table` is large and identical for every chunk, so it is published in
# module globals and the pool is created AFTER they are populated: with the
# `fork` start method each worker then inherits the index copy-on-write and
# costs nothing to set up. Passing it as a task argument instead would pickle
# the whole index once per chunk.
#
# This is also why `_start_method` must stay `fork` for this stage: under
# `spawn` the globals would not be inherited and every worker would need the
# index pickled to it.

_WORKER_S1_DF: Optional[pd.DataFrame] = None
_WORKER_INDEX: Optional[pd.DataFrame] = None
_WORKER_META: Optional[pd.DataFrame] = None
_WORKER_OPTS: Optional[Dict[str, Any]] = None


def _publish_worker_context(
    s1_df: pd.DataFrame,
    index_table: pd.DataFrame,
    meta_table: pd.DataFrame,
    opts: Dict[str, Any],
) -> None:
    """Make the read-only per-chunk inputs available to forked workers."""
    global _WORKER_S1_DF, _WORKER_INDEX, _WORKER_META, _WORKER_OPTS
    _WORKER_S1_DF = s1_df
    _WORKER_INDEX = index_table
    _WORKER_META = meta_table
    _WORKER_OPTS = opts


def _clear_worker_context() -> None:
    global _WORKER_S1_DF, _WORKER_INDEX, _WORKER_META, _WORKER_OPTS
    _WORKER_S1_DF = None
    _WORKER_INDEX = None
    _WORKER_META = None
    _WORKER_OPTS = None


def _blocking_chunk_worker(task: Tuple[int, int, int, Optional[str], Optional[str], bool]):
    """
    Compute candidates for one S1 chunk and checkpoint it.

    Returns only small scalar statistics -- the candidate rows themselves go
    straight to this chunk's own parquet file, so nothing large is pickled
    back to the parent. That keeps the parent's memory flat regardless of how
    many candidates a chunk produces.

    Task tuple: (chunk_idx, start, stop, cand_ckpt_path, prov_ckpt_path, want_prov)
    """
    chunk_idx, start, stop, cand_ckpt_path, prov_ckpt_path, want_prov = task

    if _WORKER_S1_DF is None or _WORKER_INDEX is None or _WORKER_META is None:
        raise RuntimeError(
            "blocking worker context was not published. The pool must be "
            "created with the fork start method after "
            "_publish_worker_context() has been called."
        )
    o = _WORKER_OPTS or {}

    s1_chunk = _WORKER_S1_DF.iloc[start:stop]
    cand_df, prov_df = _candidates_for_chunk(
        s1_chunk, _WORKER_INDEX, _WORKER_META, o.get("active_rules"),
        o.get("enable_coarse_filter", True),
        o.get("min_coarse_similarity", 0.30),
        o.get("max_candidates_per_entity", 200),
    )

    # "Every S1 id present" contract, including zero-candidate singletons.
    all_ids_in_chunk = s1_chunk["entity_id"].astype(str).str.strip()
    cand_by_s1 = (
        cand_df.groupby("source1_entity_id")["candidate_entity_id"]
        .apply(lambda s: ",".join(dict.fromkeys(s)))
    )
    out_rows = pd.DataFrame({
        "source1_entity_id": all_ids_in_chunk.values,
        "candidate_entity_ids": all_ids_in_chunk.map(cand_by_s1).fillna("").values,
    })

    if cand_ckpt_path:
        out_rows.to_parquet(cand_ckpt_path, index=False)
        if prov_ckpt_path is not None and want_prov:
            prov_df.to_parquet(prov_ckpt_path, index=False)

    n_cands = len(cand_df)
    n_rows = len(s1_chunk)
    del s1_chunk, cand_df, prov_df, out_rows, all_ids_in_chunk, cand_by_s1
    gc.collect()
    return chunk_idx, n_cands, n_rows


def _process_chunk_serial(
    chunk_idx: int, start: int, stop: int,
    cand_ckpt_path: Optional[str], prov_ckpt_path: Optional[str], want_prov: bool,
    s1_df: pd.DataFrame, index_table: pd.DataFrame, meta_table: pd.DataFrame,
    opts: Dict[str, Any],
):
    """Single-process path, identical logic to `_blocking_chunk_worker`."""
    return _blocking_chunk_worker(
        (chunk_idx, start, stop, cand_ckpt_path, prov_ckpt_path, want_prov)
    )


def generate_candidate_pairs_vectorized(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    output_candidate_tsv: str,
    output_provenance_tsv: Optional[str] = None,
    active_rules: Optional[Set[str]] = None,
    max_candidates_per_entity: Optional[int] = 200,
    max_bucket_size: int = 5000,
    enable_coarse_filter: bool = True,
    min_coarse_similarity: float = 0.30,
    chunk_size: int = 2000,
    checkpoint_dir: Optional[str] = None,
    resume: bool = True,
    memory_budget_gb: Optional[float] = None,
    stage_label: str = "Blocking",
    workers: Optional[int] = None,
    force_resume_incompatible: bool = False,
    logger=print,
) -> Dict[str, Any]:
    """
    OPTION 3 main entrypoint. Streams candidate_pairs.tsv (and, if
    requested, candidate_provenance.tsv) directly -- it never builds a
    whole-dataset candidate_mapping/provenance_mapping dict in RAM. Peak
    RAM is bounded by (index_table size) + O(chunk_size), not by
    O(total_candidates).

    Returns summary statistics only (counts/timings), not the candidates
    themselves -- callers that need the mapping in memory (e.g. for a
    small-scale old-vs-new validation run) should read it back with
    `load_candidate_mapping_from_tsv`, which is proportional to actual
    candidate count, not to peak processing memory.
    """
    t0 = time.time()
    events = MemoryEventLog()
    guard = MemoryGuard(memory_budget_gb) if memory_budget_gb else None

    ckpt_path = Path(checkpoint_dir) if checkpoint_dir else None
    if ckpt_path:
        ckpt_path.mkdir(parents=True, exist_ok=True)
    manifest_path = ckpt_path / "manifest.json" if ckpt_path else None

    events.record(f"{stage_label}:start", logger=logger)

    logger(f"  [Vectorized Blocking] Building tabular index from "
           f"{len(s2_df) + len(s3_df):,} S2+S3 records...")
    index_table, meta_table = build_index_table(s2_df, s3_df, active_rules=active_rules)
    events.record(f"{stage_label}:index_built", logger=logger)

    index_table = _filter_hot_buckets(index_table, max_bucket_size)
    events.record(f"{stage_label}:hot_keys_filtered", logger=logger)
    logger(f"  [Vectorized Blocking] Index table: {len(index_table):,} rows "
           f"(after hot-key exclusion at max_bucket_size={max_bucket_size:,}), "
           f"{index_table.memory_usage(deep=True).sum() / 1e6:.1f} MB resident.")

    n_s1 = len(s1_df)
    bounds = _chunk_bounds(n_s1, chunk_size)
    n_chunks = len(bounds)

    completed_chunks: Set[int] = set()

    # Everything that must match for a resume to be safe. `chunk_size` is the
    # dangerous one: completed_chunks is a list of chunk INDICES, so changing
    # it re-bounds the S1 rows and the stored chunks no longer correspond to
    # the rows they were computed from. The result is not a crash -- it is a
    # quietly wrong candidate set, which is far worse.
    current_config = {
        "chunk_size": chunk_size,
        "max_bucket_size": max_bucket_size,
        "max_candidates_per_entity": max_candidates_per_entity,
        "enable_coarse_filter": enable_coarse_filter,
        "min_coarse_similarity": round(float(min_coarse_similarity), 6),
        "active_rules": sorted(active_rules) if active_rules else None,
        "impl_version": "vectorized_v2.1",
        "rule_version": "rules_v2",
    }
    input_fingerprint = _input_fingerprint(s1_df, index_table, meta_table)

    if manifest_path and manifest_path.exists() and resume:
        with open(manifest_path) as f:
            manifest = json.load(f)
        completed_chunks = set(manifest.get("completed_chunks", []))

        if completed_chunks:
            prior = manifest.get("config") or {}
            mismatches = []
            for key, value in current_config.items():
                if key in prior and prior[key] != value:
                    mismatches.append(f"{key}: checkpoint={prior[key]!r} current={value!r}")
            prior_fp = manifest.get("input_fingerprint")
            if prior_fp is not None and prior_fp != input_fingerprint:
                mismatches.append(
                    f"input_fingerprint: checkpoint={prior_fp!r} current={input_fingerprint!r}"
                )
            if mismatches and not force_resume_incompatible:
                raise CheckpointConfigError(
                    f"Refusing to resume {stage_label} checkpoint at "
                    f"{ckpt_path}: it was produced under a different "
                    f"configuration, so the stored chunks do not correspond to "
                    f"the rows they would be resumed with.\n  "
                    + "\n  ".join(mismatches)
                    + "\n\nStart a fresh checkpoint (delete the directory, or "
                      "point --checkpoint-dir elsewhere), or pass "
                      "--force-resume-incompatible to override and accept that "
                      "the output may be incomplete or incorrect."
                )
            if mismatches:
                logger(f"  ⚠️  Resuming {stage_label} checkpoint DESPITE a "
                       f"configuration change (--force-resume-incompatible): "
                       + "; ".join(mismatches)
                       + " — output may be incomplete or incorrect.")
            else:
                logger(f"  [Vectorized Blocking] Resuming — {len(completed_chunks)}/{n_chunks} "
                       f"chunks already checkpointed.")
    else:
        manifest = {"completed_chunks": [], "config": current_config,
                    "input_fingerprint": input_fingerprint}

    progress = ProgressTracker(total=n_s1, label="S1")
    progress.processed = sum(
        (bounds[i][1] - bounds[i][0]) for i in completed_chunks if i < len(bounds)
    )
    total_candidates_seen = 0

    from src import parallel
    n_workers = parallel.resolve_workers(workers)
    want_prov = bool(output_provenance_tsv)

    def _ckpt_paths(chunk_idx: int):
        if not ckpt_path:
            return None, None
        return (str(ckpt_path / f"chunk_{chunk_idx:06d}.parquet"),
                str(ckpt_path / f"chunk_{chunk_idx:06d}_prov.parquet"))

    def _already_done(chunk_idx: int) -> bool:
        cand_ckpt, _ = _ckpt_paths(chunk_idx)
        if chunk_idx in completed_chunks and cand_ckpt and os.path.exists(cand_ckpt):
            return True
        return False

    def _commit(chunk_idx: int, n_cands: int, n_rows: int):
        """Record a finished chunk: manifest, counters, progress, logging."""
        nonlocal total_candidates_seen
        completed_chunks.add(chunk_idx)
        if manifest_path:
            manifest["completed_chunks"] = sorted(completed_chunks)
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)
        total_candidates_seen += n_cands
        progress.update(n_rows)
        if chunk_idx % max(1, n_chunks // 20) == 0 or chunk_idx == n_chunks - 1:
            snap = events.record(f"{stage_label}:chunk_{chunk_idx}", echo=False, logger=logger)
            logger("  " + progress.snapshot_line({
                "chunk": f"{chunk_idx+1}/{n_chunks}",
                "cands": f"{total_candidates_seen:,}",
                "mem": f"{snap.line().split('] ')[1]}" if snap else "",
            }))

    def _guard_or_stop(chunk_idx: int):
        if guard is None:
            return
        try:
            guard.check(context={"stage": stage_label, "chunk": chunk_idx,
                                 "processed_s1": progress.processed})
        except MemoryBudgetExceeded:
            if manifest_path:
                with open(manifest_path, "w") as f:
                    json.dump(manifest, f, indent=2)
            logger(f"  🛑 Stopping gracefully at chunk {chunk_idx}/{n_chunks} "
                   f"(checkpoint saved, {len(completed_chunks)} chunks safe on disk).")
            raise

    # Chunks still to do, in order.
    pending = [i for i in range(n_chunks) if not _already_done(i)]

    # Per-chunk settings, published to workers (or used inline). Built once,
    # for BOTH execution paths -- an earlier version of this only built it for
    # the parallel path, which silently reset max_candidates_per_entity to its
    # default on the single-worker fallback.
    opts = {
        "active_rules": active_rules,
        "enable_coarse_filter": enable_coarse_filter,
        "min_coarse_similarity": min_coarse_similarity,
        "max_candidates_per_entity": max_candidates_per_entity,
    }

    # Parallel only pays off when there is more than one chunk to do and a fork
    # is available (the index reaches workers by fork inheritance, not by
    # pickling).
    #
    # NOTE: this deliberately does NOT use parallel.should_run_serial(), which
    # is a *row* budget. It is applied here to a *chunk* count, and the two are
    # not comparable: a single blocking chunk of `chunk_size` S1 rows costs a
    # full merge against the index plus a per-entity coarse-filter pass
    # (measured ~37s at chunk_size=2000 on a 20K S1 set), so even two chunks
    # are far more work than a fork costs. Gating on the row threshold here
    # meant a 10-chunk run stayed single-process -- which is precisely the
    # "one core busy, all the rest idle" case this parallelisation exists to
    # fix.
    start_method = parallel._start_method()
    use_pool = (
        n_workers > 1
        and len(pending) > 1
    )

    if use_pool:
        # Publish the read-only inputs BEFORE creating the pool so forked
        # workers inherit the index copy-on-write instead of receiving a
        # pickled copy per task.
        _publish_worker_context(s1_df, index_table, meta_table, opts)
        try:
            if start_method == "fork":
                pool = parallel.get_pool(workers=n_workers)
            else:
                pool = parallel.get_pool(
                    initializer=_publish_worker_context,
                    initargs=(s1_df, index_table, meta_table, opts),
                    workers=n_workers,
                )
            logger(f"  [Vectorized Blocking] Processing {len(pending)} chunk(s) "
                   f"across {n_workers} worker process(es).")
            tasks = []
            for chunk_idx in pending:
                start, end = bounds[chunk_idx]
                cand_ckpt, prov_ckpt = _ckpt_paths(chunk_idx)
                tasks.append((chunk_idx, start, end, cand_ckpt, prov_ckpt, want_prov))

            stopped_early = False
            for done_count, (chunk_idx, n_cands, n_rows) in enumerate(
                pool.imap(_blocking_chunk_worker, tasks, chunksize=1), start=1
            ):
                _commit(chunk_idx, n_cands, n_rows)
                # The guard can only stop further *dispatch*, not work already
                # in flight; every in-flight chunk still checkpoints cleanly, so
                # a resume picks up from here.
                if done_count % max(1, len(pending) // 20) == 0 or done_count == len(pending):
                    _guard_or_stop(done_count)
        except MemoryBudgetExceeded:
            stopped_early = True
            logger(f"  [Vectorized Blocking] Resumable: rerun to continue from "
                   f"chunk {len(completed_chunks)}/{n_chunks}.")
        finally:
            _clear_worker_context()
            parallel.shutdown_pool()
    else:
        if n_workers > 1 and start_method != "fork":
            logger("  [Vectorized Blocking] Note: start method is not 'fork'; "
                   "running single-process so the index is not pickled per chunk.")
        _publish_worker_context(s1_df, index_table, meta_table, opts)
        try:
            for done_count, chunk_idx in enumerate(pending, start=1):
                _guard_or_stop(chunk_idx)
                start, end = bounds[chunk_idx]
                cand_ckpt, prov_ckpt = _ckpt_paths(chunk_idx)
                _, n_cands, n_rows = _blocking_chunk_worker(
                    (chunk_idx, start, end, cand_ckpt, prov_ckpt, want_prov)
                )
                _commit(chunk_idx, n_cands, n_rows)
        finally:
            _clear_worker_context()

    events.record(f"{stage_label}:chunks_done", logger=logger)

    # ── Streaming final export: read one checkpoint chunk at a time, append,
    #    release. Never loads every chunk into RAM simultaneously. ──
    if ckpt_path:
        _stream_export_tsv(ckpt_path, n_chunks, output_candidate_tsv, kind="candidate")
        if output_provenance_tsv:
            _stream_export_provenance_tsv(ckpt_path, n_chunks, output_provenance_tsv)
    else:
        raise ValueError("checkpoint_dir is required for the streaming vectorized implementation.")

    events.record(f"{stage_label}:exported", logger=logger)

    elapsed = time.time() - t0
    summary = {
        "n_s1": n_s1,
        "n_chunks": n_chunks,
        "chunk_size": chunk_size,
        "total_candidates": total_candidates_seen,
        "elapsed_sec": elapsed,
        "peak_total_pss_mb": events.peak_total_pss(),
        "peak_cgroup_mb": events.peak_cgroup(),
        "output_candidate_tsv": str(output_candidate_tsv),
    }
    if ckpt_path:
        events.to_json(ckpt_path / "memory_events.json")
    logger(f"  [Vectorized Blocking] Done in {elapsed:.1f}s. "
           f"{total_candidates_seen:,} candidate rows written across {n_chunks} chunks.")
    return summary


def _stream_export_tsv(ckpt_path: Path, n_chunks: int, output_path: str, kind: str = "candidate"):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    with open(output_path, "w") as out:
        out.write("source1_entity_id\tcandidate_entity_ids\n")
        for i in range(n_chunks):
            f = ckpt_path / f"chunk_{i:06d}.parquet"
            if not f.exists():
                continue
            part = pd.read_parquet(f)
            part.to_csv(out, sep="\t", index=False, header=False)
            del part
    logger_line = f"Successfully exported (streamed) candidate pairs to {output_path}"
    print(logger_line)


def _stream_export_provenance_tsv(ckpt_path: Path, n_chunks: int, output_path: str):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    cols = ["source1_entity_id", "candidate_entity_id", "matched_rules",
            "rule_count", "coarse_similarity"]
    with open(output_path, "w") as out:
        out.write("\t".join(cols) + "\n")
        for i in range(n_chunks):
            f = ckpt_path / f"chunk_{i:06d}_prov.parquet"
            if not f.exists():
                continue
            part = pd.read_parquet(f)
            if not part.empty:
                part[cols].to_csv(out, sep="\t", index=False, header=False)
            del part
    print(f"Successfully exported (streamed) provenance to {output_path}")


def load_candidate_mapping_from_tsv(path: str) -> Dict[str, List[str]]:
    """
    Read a candidate_pairs.tsv back into the OLD-style
    {s1_id: [candidate_ids]} dict. Intended for small/medium-scale
    validation (old-vs-new comparison, blocking-recall measurement) --
    NOT for production-scale use, since it reintroduces the O(total
    candidates) in-RAM structure the streaming design exists to avoid.
    """
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    mapping = {}
    for _, row in df.iterrows():
        s1 = row["source1_entity_id"]
        cands = row["candidate_entity_ids"]
        mapping[s1] = [c for c in cands.split(",") if c] if cands else []
    return mapping
