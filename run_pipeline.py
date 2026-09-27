import argparse
import gc
import json
import os
import sys
import time
import pickle
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.blocking import generate_candidate_pairs, export_candidate_pairs_tsv
from src.blocking_vectorized import generate_candidate_pairs_vectorized, load_candidate_mapping_from_tsv
from src.features import (
    fit_tfidf_from_current_records,
    generate_feature_matrix,
    load_source_records,
)
from src.predict import run_inference
from src.threshold import sweep_threshold
from src.train import train_model


# ─────────────────────────────────────────────────────────────
# CHECKPOINT HELPERS
# ─────────────────────────────────────────────────────────────

def _ckpt_done(path: Path) -> bool:
    """Return True if a stage output file already exists and is non-empty."""
    if not path.exists():
        return False
    if path.stat().st_size == 0:
        return False
    return True


def _skip(stage: str, path: Path) -> bool:
    if _ckpt_done(path):
        print(f"  ✅ SKIP {stage} — checkpoint found: {path.name}")
        return True
    return False


def _done(stage: str, path: Path):
    print(f"  ✔  {stage} complete → {path.name}")


# ─────────────────────────────────────────────────────────────
# FEATURE SAVING WITH CHUNK CHECKPOINTING
# ─────────────────────────────────────────────────────────────

def merge_feature_chunks_streaming(chunk_paths, output_path):
    """
    Combine N per-chunk feature parquet files into one final parquet file
    WITHOUT ever holding more than one chunk's rows in memory at once (P0-3
    fix for FINAL_PRODUCTION_READINESS_REPORT.md §9).

    PREVIOUSLY: `all_parts = [pd.read_parquet(f) for f in chunk_paths];
    pd.concat(all_parts)` loaded every chunk for the entire split into RAM
    simultaneously before writing -- defeating the whole point of Stage 4's
    per-chunk checkpointing, and structurally capable of OOMing at large
    candidate counts (the 86M-candidate case) even after Stage 3 is fixed.

    FIX: each chunk is read from disk, written as its own row group into
    `output_path` via pyarrow.parquet.ParquetWriter, and released before
    the next chunk is read. Row order is preserved (chunks are written in
    the exact order given in `chunk_paths`, which callers must supply in
    the original chunk-index order); schema is taken from the first chunk
    on disk and enforced (via `.cast`) on every subsequent chunk, so a
    dtype drift between chunks fails predictably rather than silently
    producing an inconsistent file.

    Returns the total row count written. If `chunk_paths` is empty (e.g. no
    candidates at all for this split), returns 0 without creating
    `output_path` -- callers that need a placeholder file for this case
    (so downstream stages don't hit a missing-file error) should create one
    explicitly, since only the caller knows the correct empty schema.
    """
    import pyarrow.parquet as pq

    total_rows = 0
    writer = None
    schema = None
    try:
        for ckpt_file in chunk_paths:
            ckpt_file = Path(ckpt_file)
            if not ckpt_file.exists():
                continue
            table = pq.read_table(ckpt_file)
            if writer is None:
                schema = table.schema
                writer = pq.ParquetWriter(str(output_path), schema)
            elif table.schema != schema:
                table = table.cast(schema)
            writer.write_table(table)
            total_rows += table.num_rows
            del table
    finally:
        if writer is not None:
            writer.close()
    return total_rows


def save_features(candidate_path, data_dir, split, tfidf, output_path, ckpt_dir,
                   stage4_chunk_size: int = 2_000, workers: int = None):
    """
    Compute and save the feature matrix.
    Uses chunk-level checkpointing — if interrupted mid-way,
    already-completed chunks are reloaded and only remaining
    chunks are processed.

    P0-3 FIX (full Stage-4 audit, not just the final merge):
    PREVIOUSLY this function did, unconditionally, before any chunking:
        candidate_pairs = load_candidate_pairs(candidate_path)   # whole file
        attached        = attach_records(candidate_pairs, sources)  # whole thing
        n_chunks = n_cores * 4
        chunks = split(attached)   # only NOW does anything get chunked
    i.e. the ENTIRE exploded candidate-pair table and the ENTIRE attached
    dataframe were fully materialized in RAM before the loop below ever ran.
    At production scale (~86M candidate pairs) this defeats the point of
    per-chunk checkpointing entirely and is a real OOM risk on its own,
    independent of the final-merge fix that was already in place
    (`merge_feature_chunks_streaming`). "The final merge is streaming" does
    NOT mean "Stage 4 is memory bounded" -- confirmed true here by reading
    the code, exactly as flagged in the takeover brief.

    FIX: `candidate_path` (the raw, un-exploded candidate_pairs.tsv --
    one row per S1 entity) is now read INCREMENTALLY via
    `src.features.iter_exploded_candidate_chunks`, which itself uses
    `pandas.read_csv(..., chunksize=stage4_chunk_size)`. Each yielded chunk
    is only `stage4_chunk_size` raw S1 rows' worth of exploded pairs
    (bounded by `stage4_chunk_size * max_candidates_per_entity` in the
    worst case), gets `attach_records` + `compute_features` applied to just
    that chunk, is written to its own checkpoint parquet, and is then
    released before the next chunk is read from the file. `sources` (the
    S1/S2/S3 lookup tables) are still loaded once, in full -- this is a
    deliberate, bounded design choice, not an oversight: at production
    scale sources are ~10.3M short (name/address/country) rows, several
    orders of magnitude smaller than the ~86M-row candidate-pair table they
    are joined against, and every chunk needs random access into all three
    of them, so keeping them resident is both necessary and cheap relative
    to the actual memory risk (the candidate-pair/attached/feature tables).

    Chunk size is now `stage4_chunk_size`, independent of CPU core count
    (previously `n_chunks = n_cores * 4`, a CPU-derived count with no
    memory justification -- flagged explicitly in the takeover brief). It
    can be tuned separately from the Stage 3 blocking chunk size and from
    core count.
    """
    from src.features import (
        load_source_records,
        iter_exploded_candidate_chunks,
        attach_records,
        compute_features,
        TFIDFSimilarityModel,
        FEATURE_COLS,
    )
    from src.features import RecordLookup
    from tqdm import tqdm

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_meta = ckpt_dir / f"{split}_feature_ckpt_meta.json"

    # Sources are loaded once, in full -- see docstring above for why this
    # is bounded/acceptable and NOT the thing that needs streaming.
    sources = load_source_records(data_dir, split=split)

    # Build the S1/S2/S3 join tables ONCE for the whole stage rather than once
    # per chunk inside attach_records. Previously each chunk re-sliced and
    # re-concatenated the entire S2+S3 record set: at production scale that is
    # ~10.3M records rebuilt >1,100 times per split.
    lookup = RecordLookup.from_sources(sources)
    print(f"  [Features/{split}] Record lookup built once: "
          f"{lookup.nbytes()/1e6:.0f} MB (was rebuilt per chunk).")

    # ── Load existing checkpoint metadata ──
    completed = set()
    if ckpt_meta.exists():
        with open(ckpt_meta) as f:
            completed = set(json.load(f).get("completed_chunks", []))
        print(f"  [Features/{split}] Resuming — {len(completed)} chunk(s) already done "
              f"(chunk_size={stage4_chunk_size} raw S1 rows/chunk).")

    # ── Stream candidate_pairs.tsv in bounded chunks; attach + compute ──
    # features per chunk; never hold more than one chunk's exploded/attached/
    # feature dataframe in memory at a time.
    print(f"  [Features/{split}] Streaming candidate pairs "
          f"({stage4_chunk_size} raw S1 rows/chunk)...")
    n_written_this_run = 0
    for i, exploded_chunk in enumerate(
        tqdm(
            iter_exploded_candidate_chunks(Path(candidate_path), chunk_size=stage4_chunk_size),
            desc=f"  Feature chunks [stream, chunk_size={stage4_chunk_size}]",
        )
    ):
        ckpt_file = ckpt_dir / f"{split}_chunk_{i:06d}.parquet"
        if i in completed:
            # Already computed in a prior interrupted run -- we still had to
            # read (and explode) this chunk from disk to keep chunk indices
            # aligned with the checkpoint, but we skip attach/feature-compute
            # for it, which is the expensive part.
            continue
        chunk = attach_records(exploded_chunk, sources, lookup=lookup)
        del exploded_chunk
        feat = compute_features(chunk, tfidf, workers=workers)
        meta = chunk[["source1_entity_id", "candidate_entity_id", "candidate_source"]].reset_index(drop=True)
        result = pd.concat([meta, feat.reset_index(drop=True)], axis=1)
        result.to_parquet(ckpt_file, index=False)
        completed.add(i)
        n_written_this_run += 1
        with open(ckpt_meta, "w") as f:
            json.dump({"completed_chunks": list(completed)}, f)
        del chunk, feat, meta, result
        gc.collect()

    # ── Merge all chunks into final output (P0-3 fix) ──
    print(f"  [Features/{split}] Streaming {len(completed)} chunks into final parquet "
          f"(one chunk resident in memory at a time)...")
    chunk_paths = [ckpt_dir / f"{split}_chunk_{i:06d}.parquet" for i in sorted(completed)]
    total_rows = merge_feature_chunks_streaming(chunk_paths, output_path)
    if total_rows == 0 and not Path(output_path).exists():
        # No chunks at all for this split (e.g. an empty candidate set) --
        # the streaming merge intentionally creates no file in this case
        # (it doesn't know the schema); write an empty placeholder with the
        # correct schema so downstream stages don't hit a missing-file
        # error. This is a real edge case the old `pd.concat([])`
        # implementation could not handle either (raises ValueError on an
        # empty list), so this is a strict improvement, not a behavior
        # change for any case that used to work.
        from src.features import FEATURE_COLS
        empty_cols = ["source1_entity_id", "candidate_entity_id", "candidate_source"] + list(FEATURE_COLS)
        pd.DataFrame(columns=empty_cols).to_parquet(output_path, index=False)
    print(f"  [Features/{split}] Saved {total_rows:,} rows → {output_path.name}")

    # ── Clean up chunk files to free disk space ──
    for ckpt_file in chunk_paths:
        if ckpt_file.exists():
            ckpt_file.unlink()
    if ckpt_meta.exists():
        ckpt_meta.unlink()

    # Release the worker pool before returning. The pool is deliberately
    # process-wide (so Stage 4 does not pay a fork per chunk), but keeping
    # `cpu_count()` worker processes resident through the training stage costs
    # real memory and idle CPU for nothing -- Stage 5 is the pipeline's memory
    # high-water mark, and every MB the workers hold is a MB it cannot have.
    try:
        from src import parallel
        parallel.shutdown_pool()
    except Exception:
        pass

    return {"output_path": str(output_path), "n_rows": total_rows}


# ─────────────────────────────────────────────────────────────
# BLOCKING DISPATCH — old (frozen baseline) vs vectorized (Option 3)
# ─────────────────────────────────────────────────────────────

def _check_impl_marker(checkpoint_dir, impl):
    """
    P0-2 fix (blocking-implementation half): the 'old' and 'vectorized'
    implementations use different, incompatible checkpoint file formats
    inside the same checkpoint_dir (old: chunk_NNNN.json with an ad-hoc
    format; vectorized: chunk_NNNNNN.parquet + manifest.json). Switching
    --blocking-impl between runs that point at the same checkpoint_dir
    doesn't corrupt the *other* implementation's files (different
    filenames/globs), but it silently throws away all prior progress with
    no warning, which is exactly the kind of silent, surprising behavior
    P0-2 asks to eliminate. Fail loudly instead.
    """
    checkpoint_dir = Path(checkpoint_dir)
    marker_path = checkpoint_dir / "IMPL_MARKER.json"
    if marker_path.exists():
        with open(marker_path) as f:
            prior_impl = json.load(f).get("impl")
        if prior_impl != impl:
            raise RuntimeError(
                "Checkpoint configuration is incompatible with current run.\n"
                f"Checkpoint directory {checkpoint_dir} was created with "
                f"--blocking-impl {prior_impl!r} but current run uses "
                f"--blocking-impl {impl!r}. Delete/restart the checkpoint "
                "directory, or use a different --checkpoint-dir, before "
                "switching blocking implementations."
            )
    else:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        with open(marker_path, "w") as f:
            json.dump({"impl": impl}, f)


def _run_blocking_stage(
    impl, s1_df, s2_df, s3_df, output_candidate_tsv, checkpoint_dir,
    chunk_size, memory_budget_gb, stage_label, force_resume_incompatible=False,
    workers=None,
):
    """
    Dispatch to either the frozen OLD implementation (src.blocking --
    retained only as an explicit baseline/testing option, see P0-1) or the
    NEW Option 3 vectorized/streaming implementation (src/blocking_vectorized,
    now the default -- see P0-1).
    """
    _check_impl_marker(checkpoint_dir, impl)
    all_s1_ids = s1_df["entity_id"].astype(str).str.strip().tolist()
    if impl == "old":
        candidates, _ = generate_candidate_pairs(
            s1_df, s2_df, s3_df,
            max_candidates_per_entity=200,
            max_bucket_size=5000,
            enable_coarse_filter=True,
            min_coarse_similarity=0.30,
            checkpoint_dir=checkpoint_dir,
        )
        export_candidate_pairs_tsv(candidates, all_s1_ids, str(output_candidate_tsv))
    elif impl == "vectorized":
        generate_candidate_pairs_vectorized(
            s1_df, s2_df, s3_df,
            output_candidate_tsv=str(output_candidate_tsv),
            max_candidates_per_entity=200,
            max_bucket_size=5000,
            enable_coarse_filter=True,
            min_coarse_similarity=0.30,
            chunk_size=chunk_size,
            checkpoint_dir=str(checkpoint_dir),
            resume=True,
            force_resume_incompatible=force_resume_incompatible,
            memory_budget_gb=memory_budget_gb,
            stage_label=stage_label,
            workers=workers,
        )
    else:
        raise ValueError(f"Unknown --blocking-impl {impl!r}; expected 'old' or 'vectorized'.")


# ─────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────

def run(
    dataset_dir=PROJECT_ROOT / "dataset",
    output_dir=PROJECT_ROOT / "output",
    blocking_impl: str = "vectorized",
    blocking_chunk_size: int = 2_000,
    stage4_chunk_size: int = 2_000,
    memory_budget_gb: float = None,
    force_resume_incompatible: bool = False,
    workers: int = None,
    deterministic: bool = False,
):
    train_dir     = dataset_dir / "train"
    test_dir      = dataset_dir / "test"
    artifacts_dir = PROJECT_ROOT / "artifacts"
    model_dir     = PROJECT_ROOT / "model"
    ckpt_dir      = PROJECT_ROOT / "checkpoints"

    for d in [artifacts_dir, output_dir, model_dir, ckpt_dir]:
        d.mkdir(parents=True, exist_ok=True)

    ground_truth_path            = train_dir / "train_ground_truth.tsv"
    train_candidate_path         = artifacts_dir / "train_candidate_pairs.tsv"
    train_features_path          = artifacts_dir / "train_features.parquet"
    model_path                   = model_dir / "model.lgb"
    validation_predictions_path  = artifacts_dir / "validation_predictions.parquet"
    threshold_path               = model_dir / "threshold.json"
    test_candidate_path          = output_dir / "candidate_pairs.tsv"
    test_features_path           = artifacts_dir / "test_features.parquet"
    matching_results_path        = output_dir / "matching_results.tsv"
    tfidf_ckpt_path              = artifacts_dir / "tfidf_model.pkl"

    if not ground_truth_path.is_file():
        raise FileNotFoundError(f"Training ground truth not found: {ground_truth_path}")

    start_time = time.time()

    # ── STAGE 1 & 2: Load train sources + fit TF-IDF ──────────────
    print("\n[1/9] Loading Train sources...")
    train_sources = load_source_records(train_dir, split="train")

    print("[2/9] Fitting TF-IDF model...")
    if _ckpt_done(tfidf_ckpt_path):
        print("  ✅ SKIP — loading cached TF-IDF model from checkpoint.")
        with open(tfidf_ckpt_path, "rb") as f:
            tfidf = pickle.load(f)
    else:
        tfidf = fit_tfidf_from_current_records(train_sources)
        with open(tfidf_ckpt_path, "wb") as f:
            pickle.dump(tfidf, f)
        _done("TF-IDF fit", tfidf_ckpt_path)

    # ── STAGE 3: Train Blocking ────────────────────────────────────
    print("\n[3/9] Generating Train Candidate Pairs (Blocking)...")
    print(f"      impl={blocking_impl}")
    if not _skip("Train Blocking", train_candidate_path):
        _run_blocking_stage(
            blocking_impl,
            train_sources["source1"], train_sources["source2"], train_sources["source3"],
            output_candidate_tsv=train_candidate_path,
            checkpoint_dir=ckpt_dir / "blocking_train",
            chunk_size=blocking_chunk_size,
            memory_budget_gb=memory_budget_gb,
            stage_label="Train Blocking",
            force_resume_incompatible=force_resume_incompatible,
            workers=workers,
        )
        _done("Train Blocking", train_candidate_path)

    del train_sources
    gc.collect()

    # ── STAGE 4: Train Features ────────────────────────────────────
    print("\n[4/9] Computing Features for Train Candidates...")
    print("      (ETA: ~15-20 minutes. The progress bar below shows a live estimate)")
    if not _skip("Train Features", train_features_path):
        save_features(
            train_candidate_path, train_dir, "train",
            tfidf, train_features_path, ckpt_dir / "features_train",
            stage4_chunk_size=stage4_chunk_size,
            workers=workers,
        )
        _done("Train Features", train_features_path)
    gc.collect()

    # ── STAGE 5: Train Model ───────────────────────────────────────
    print("\n[5/9] Training LightGBM Model...")
    if not _skip("Model Training", model_path):
        train_model(
            train_features_path,
            ground_truth_path,
            model_path,
            model_dir / "shap.json",
            validation_predictions_path,
            {
                "objective":     "binary",
                "metric":        "binary_logloss",
                "learning_rate": 0.05,
                "num_leaves":    63,
                "verbose":       -1,
                "num_threads":   workers or (os.cpu_count() or 1),
                # LightGBM's multithreaded histogram build is not
                # deterministic by default: two runs on identical input can
                # produce different models (verified -- see docs/PERFORMANCE.md).
                # Off by default because it costs throughput; turn it on when a
                # bit-reproducible submission matters more than speed.
                **({"deterministic": True, "force_row_wise": True} if deterministic else {}),
            },
        )
        _done("Model Training", model_path)

    # ── STAGE 6: Threshold Sweep ───────────────────────────────────
    print("\n[6/9] Sweeping Threshold for F0.5 optimization...")
    if not _skip("Threshold Sweep", threshold_path):
        sweep_threshold(
            validation_predictions_path,
            ground_truth_path,
            threshold_path,
            candidate_pairs_path=train_candidate_path,
        )
        _done("Threshold Sweep", threshold_path)

    # ── STAGE 7 & 8: Load Test Sources + Test Blocking ────────────
    print("\n[7/9] Loading Test sources...")
    test_sources    = load_source_records(test_dir, split="test")
    all_test_s1_ids = test_sources["source1"]["entity_id"].astype(str).str.strip().tolist()

    print("\n[8/9] Generating Test Candidate Pairs (Blocking)...")
    print(f"      impl={blocking_impl}")
    if not _skip("Test Blocking", test_candidate_path):
        _run_blocking_stage(
            blocking_impl,
            test_sources["source1"], test_sources["source2"], test_sources["source3"],
            output_candidate_tsv=test_candidate_path,
            checkpoint_dir=ckpt_dir / "blocking_test",
            chunk_size=blocking_chunk_size,
            memory_budget_gb=memory_budget_gb,
            stage_label="Test Blocking",
            force_resume_incompatible=force_resume_incompatible,
            workers=workers,
        )
        _done("Test Blocking", test_candidate_path)

    del test_sources
    gc.collect()

    # ── STAGE 9: Test Features ─────────────────────────────────────
    print("\n[9/9] Computing Features for Test Candidates...")
    print("      (ETA: ~15-20 minutes. The progress bar below shows a live estimate)")
    if not _skip("Test Features", test_features_path):
        save_features(
            test_candidate_path, test_dir, "test",
            tfidf, test_features_path, ckpt_dir / "features_test",
            stage4_chunk_size=stage4_chunk_size,
            workers=workers,
        )
        _done("Test Features", test_features_path)

    # ── FINAL: Inference ───────────────────────────────────────────
    print("\n[Final] Running Inference on Test Data...")
    if not _skip("Inference", matching_results_path):
        run_inference(
            test_features_path,
            model_path,
            threshold_path,
            matching_results_path,
            all_s1_ids=all_test_s1_ids,
        )
        _done("Inference", matching_results_path)

    elapsed = time.time() - start_time
    hours, rem = divmod(int(elapsed), 3600)
    mins, secs = divmod(rem, 60)
    print(f"\n🎉 PIPELINE COMPLETE in {hours}h {mins}m {secs}s → {output_dir}")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train and run entity resolution on dataset/train and dataset/test."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=PROJECT_ROOT / "dataset",
        help="Directory containing train/ and test/ TSV folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "output",
        help="Directory for candidate_pairs.tsv and matching_results.tsv.",
    )
    parser.add_argument(
        "--blocking-impl",
        choices=["old", "vectorized"],
        default="vectorized",
        help=(
            "Stage 3 blocking implementation. DEFAULT: 'vectorized' -- the "
            "Option 3 chunked/streaming/checkpointed implementation "
            "(src/blocking_vectorized.py), bounded to O(index + chunk) "
            "memory and REQUIRED for the real ~2.2M/~10.3M-record production "
            "dataset. 'old' is the frozen, non-memory-bounded baseline "
            "(src/blocking.py), retained ONLY as an explicit "
            "correctness/testing comparison against the vectorized "
            "implementation -- it keeps a global candidate_mapping/"
            "provenance_mapping dict for the whole dataset and re-pickles "
            "the entire S2+S3 index once per multiprocessing task, which is "
            "what caused the original production OOM. Do not use 'old' on "
            "production-scale data."
        ),
    )
    parser.add_argument(
        "--blocking-chunk-size",
        type=int,
        default=2_000,
        help=(
            "S1 rows processed per chunk in the vectorized blocking "
            "implementation. DEFAULT: 2,000 -- chosen conservatively from "
            "measured peak transient memory of ~0.27-0.36 MB per S1 row in "
            "a chunk on a moderately dense synthetic benchmark (see "
            "MEMORY_REPORT.md); at this default a single chunk's merge step "
            "is estimated at roughly 0.5-1.5 GB peak, versus several GB at "
            "the old 25,000 default. Raise this only after measuring actual "
            "peak memory per chunk on a representative slice of the real "
            "data, ideally together with --memory-budget-gb."
        ),
    )
    parser.add_argument(
        "--stage4-chunk-size",
        type=int,
        default=2_000,
        help=(
            "Raw S1 rows (from candidate_pairs.tsv, BEFORE exploding to "
            "one row per candidate pair) read and processed per chunk in "
            "Stage 4 feature computation. DEFAULT: 2,000. Independent of "
            "--blocking-chunk-size and of CPU core count (P0-3 fix -- the "
            "old code derived chunk count from `n_cores * 4`, which is not "
            "a memory-safe sizing strategy). Worst-case exploded rows per "
            "chunk is roughly this value times Stage 3's "
            "max_candidates_per_entity cap."
        ),
    )
    parser.add_argument(
        "--memory-budget-gb",
        type=float,
        default=None,
        help=(
            "If set, the vectorized blocking stage checks memory against this "
            "budget between chunks and stops gracefully (checkpoint saved, "
            "resumable with --resume/no flag change needed) instead of risking "
            "an OOM kill. Only enforced by --blocking-impl vectorized."
        ),
    )
    parser.add_argument(
        "--force-resume-incompatible",
        action="store_true",
        default=False,
        help=(
            "DANGEROUS. By default, resuming a vectorized-blocking checkpoint "
            "with different config (chunk size, bucket/candidate caps, coarse "
            "filter settings) or different input data raises an error rather "
            "than risk silently skipping or misaligning S1 records (see "
            "CheckpointConfigError in src/blocking_vectorized.py). Pass this "
            "flag to override that check and resume anyway -- only do this if "
            "you understand and accept the risk of incomplete/incorrect output."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=(
            "Worker processes for Stage 3 (blocking) and Stage 4 (feature "
            "computation). DEFAULT: None, meaning ALL logical CPUs. The parent "
            "process is blocked on the worker pool for essentially the whole "
            "of both stages, so it is not competing for a core. Set this (or "
            "the BER_WORKERS environment variable) to leave headroom on a "
            "shared box, or to reduce peak RSS on a memory-constrained "
            "instance -- each worker holds its own copy of the blocking index "
            "and feature slice."
        ),
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=False,
        help=(
            "Force LightGBM into deterministic mode (deterministic=true, "
            "force_row_wise=true). LightGBM's default multithreaded histogram "
            "construction is NOT reproducible: two runs on byte-identical "
            "input can yield different models and, because the sweep often "
            "lands on a high threshold, slightly different predictions. This "
            "flag makes the run bit-reproducible at some cost in speed. Leave "
            "it off unless you specifically need to reproduce a submission."
        ),
    )
    args = parser.parse_args()
    run(
        args.dataset_dir,
        args.output_dir,
        blocking_impl=args.blocking_impl,
        blocking_chunk_size=args.blocking_chunk_size,
        stage4_chunk_size=args.stage4_chunk_size,
        memory_budget_gb=args.memory_budget_gb,
        force_resume_incompatible=args.force_resume_incompatible,
        workers=args.workers,
        deterministic=args.deterministic,
    )


if __name__ == "__main__":
    main()