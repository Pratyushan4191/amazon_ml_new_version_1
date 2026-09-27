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
    """
    sizes = index_table.groupby(["country", "rule", "key"], observed=True).size()
    hot = sizes[sizes >= max_bucket_size]
    if hot.empty:
        return index_table
    hot_keys = set(hot.index.tolist())
    keep_mask = ~index_table.set_index(["country", "rule", "key"]).index.isin(hot_keys)
    return index_table.loc[keep_mask.values].reset_index(drop=True)


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
    if manifest_path and manifest_path.exists() and resume:
        with open(manifest_path) as f:
            manifest = json.load(f)
        completed_chunks = set(manifest.get("completed_chunks", []))
        if completed_chunks:
            logger(f"  [Vectorized Blocking] Resuming — {len(completed_chunks)}/{n_chunks} "
                   f"chunks already checkpointed.")
    else:
        manifest = {"completed_chunks": [], "config": {
            "chunk_size": chunk_size, "max_bucket_size": max_bucket_size,
            "max_candidates_per_entity": max_candidates_per_entity,
            "enable_coarse_filter": enable_coarse_filter,
            "min_coarse_similarity": min_coarse_similarity,
        }}

    progress = ProgressTracker(total=n_s1, label="S1")
    progress.processed = sum(
        (bounds[i][1] - bounds[i][0]) for i in completed_chunks if i < len(bounds)
    )
    total_candidates_seen = 0

    for chunk_idx, (start, end) in enumerate(bounds):
        chunk_ckpt = ckpt_path / f"chunk_{chunk_idx:06d}.parquet" if ckpt_path else None
        prov_ckpt = ckpt_path / f"chunk_{chunk_idx:06d}_prov.parquet" if ckpt_path else None

        if chunk_idx in completed_chunks and chunk_ckpt and chunk_ckpt.exists():
            continue  # already checkpointed -- do not recompute (resume semantics)

        if guard:
            try:
                guard.check(context={"stage": stage_label, "chunk": chunk_idx,
                                      "processed_s1": progress.processed})
            except MemoryBudgetExceeded as e:
                if manifest_path:
                    with open(manifest_path, "w") as f:
                        json.dump(manifest, f, indent=2)
                logger(f"  🛑 Stopping gracefully at chunk {chunk_idx}/{n_chunks} "
                       f"(checkpoint saved, {len(completed_chunks)} chunks safe on disk).")
                raise

        s1_chunk = s1_df.iloc[start:end]
        cand_df, prov_df = _candidates_for_chunk(
            s1_chunk, index_table, meta_table, active_rules,
            enable_coarse_filter, min_coarse_similarity, max_candidates_per_entity,
        )

        # Include every S1 id in this chunk, even singletons with zero candidates,
        # so the streaming writer can honor the "every S1 present" contract per chunk.
        all_ids_in_chunk = s1_chunk["entity_id"].astype(str).str.strip()
        cand_by_s1 = (
            cand_df.groupby("source1_entity_id")["candidate_entity_id"]
            .apply(lambda s: ",".join(dict.fromkeys(s)))
        )
        out_rows = pd.DataFrame({
            "source1_entity_id": all_ids_in_chunk.values,
            "candidate_entity_ids": all_ids_in_chunk.map(cand_by_s1).fillna("").values,
        })

        if chunk_ckpt:
            out_rows.to_parquet(chunk_ckpt, index=False)
            if prov_ckpt is not None and output_provenance_tsv:
                prov_df.to_parquet(prov_ckpt, index=False)
            completed_chunks.add(chunk_idx)
            manifest["completed_chunks"] = sorted(completed_chunks)
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)

        total_candidates_seen += len(cand_df)
        progress.update(len(s1_chunk))

        # Release chunk-local memory before the next iteration -- this is
        # what bounds RAM by O(chunk_size) instead of O(total_candidates).
        del s1_chunk, cand_df, prov_df, out_rows, all_ids_in_chunk, cand_by_s1
        gc.collect()

        if chunk_idx % max(1, n_chunks // 20) == 0 or chunk_idx == n_chunks - 1:
            snap = events.record(f"{stage_label}:chunk_{chunk_idx}", echo=False, logger=logger)
            logger("  " + progress.snapshot_line({
                "chunk": f"{chunk_idx+1}/{n_chunks}",
                "cands": f"{total_candidates_seen:,}",
                "mem": f"{snap.line().split('] ')[1]}" if snap else "",
            }))

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
