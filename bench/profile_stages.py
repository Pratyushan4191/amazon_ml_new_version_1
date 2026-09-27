"""
Per-stage wall-time + peak-RSS profiler for the BER pipeline.

Runs each pipeline stage in isolation against a synthetic dataset and reports
time and peak memory, so optimizations can be judged on measurements rather
than intuition. Also records the COW ("copy-on-write") penalty of forking a
process pool from a parent that already holds a large heap -- the mechanism
behind the production OOM.

Usage:
    python bench/profile_stages.py --dataset-dir bench/data --scale-tag 20k
    python bench/profile_stages.py --dataset-dir bench/data --only stage4
"""

import argparse
import gc
import json
import os
import pickle
import resource
import shutil
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS = []


def _rss_mb():
    """Current process RSS in MB (Linux ru_maxrss is in KB)."""
    try:
        with open("/proc/self/statm") as f:
            pages = int(f.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE") / 1e6
    except Exception:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def _peak_rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def stage(name):
    """Decorator: time a stage, record peak RSS delta, print a summary line."""
    def deco(fn):
        def wrapper(*a, **kw):
            gc.collect()
            rss_before = _rss_mb()
            peak_before = _peak_rss_mb()
            t0 = time.time()
            print(f"\n>>> {name} ...", flush=True)
            out = fn(*a, **kw)
            dt = time.time() - t0
            peak_after = _peak_rss_mb()
            rec = {
                "stage": name,
                "sec": round(dt, 2),
                "rss_before_mb": round(rss_before, 1),
                "peak_rss_mb": round(peak_after, 1),
                "peak_delta_mb": round(peak_after - rss_before, 1),
            }
            RESULTS.append(rec)
            print(f"<<< {name}: {dt:.2f}s  peakRSS={peak_after:.0f}MB "
                  f"(+{rec['peak_delta_mb']:.0f}MB)", flush=True)
            return out
        return wrapper
    return deco


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", type=Path, required=True)
    ap.add_argument("--work-dir", type=Path, default=PROJECT_ROOT / "bench" / "work")
    ap.add_argument("--only", default=None,
                    help="comma list: load,tfidf,index,blocking,features,train,threshold,infer")
    ap.add_argument("--blocking-chunk-size", type=int, default=2000)
    ap.add_argument("--stage4-chunk-size", type=int, default=2000)
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()
    only = set(args.only.split(",")) if args.only else None

    def want(s):
        return only is None or s in only

    from src.features import load_source_records, fit_tfidf_from_current_records
    from src.blocking_vectorized import (build_index_table, _filter_hot_buckets,
                                         generate_candidate_pairs_vectorized)
    from src.train import train_model
    from src.threshold import sweep_threshold
    from src.predict import run_inference
    from run_pipeline import save_features

    train_dir = args.dataset_dir / "train"
    test_dir = args.dataset_dir / "test"
    gt = train_dir / "train_ground_truth.tsv"
    if not gt.is_file():
        raise SystemExit(f"missing {gt}")

    W = args.work_dir
    if W.exists():
        shutil.rmtree(W)
    W.mkdir(parents=True)
    ck = W / "ckpt"
    ck.mkdir()
    tfidf_pkl = W / "tfidf.pkl"
    train_cand = W / "train_candidate_pairs.tsv"
    train_feat = W / "train_features.parquet"
    model_p = W / "model.lgb"
    val_pred = W / "val_preds.parquet"
    thr_p = W / "threshold.json"

    # ── Stage 1: load ────────────────────────────────────────────────
    if want("load"):
        @stage("1_load_sources")
        def _load():
            return load_source_records(train_dir, split="train")
        srcs = _load()
    else:
        srcs = load_source_records(train_dir, split="train")

    s1n, s2n, s3n = len(srcs["source1"]), len(srcs["source2"]), len(srcs["source3"])
    print(f"\nDataset: S1={s1n:,} S2={s2n:,} S3={s3n:,}")

    # ── Stage 2: tfidf ───────────────────────────────────────────────
    if want("tfidf"):
        @stage("2_fit_tfidf")
        def _tfidf():
            m = fit_tfidf_from_current_records(srcs)
            with open(tfidf_pkl, "wb") as f:
                pickle.dump(m, f)
            return m
        tfidf = _tfidf()
    else:
        with open(tfidf_pkl, "rb") as f:
            tfidf = pickle.load(f)

    # ── Stage 3a: index build ───────────────────────────────────────
    if want("index"):
        @stage("3a_build_index")
        def _idx():
            it, mt = build_index_table(srcs["source2"], srcs["source3"], active_rules=None)
            it = _filter_hot_buckets(it, 5000)
            print(f"    index_table={len(it):,} rows  "
                  f"{it.memory_usage(deep=True).sum()/1e6:.0f} MB  "
                  f"meta={len(mt):,} rows")
            return it, mt
        index_table, meta_table = _idx()
    else:
        index_table = meta_table = None

    # ── Stage 3b: blocking ──────────────────────────────────────────
    if want("blocking"):
        if index_table is None:
            index_table, meta_table = build_index_table(
                srcs["source2"], srcs["source3"], active_rules=None)
            index_table = _filter_hot_buckets(index_table, 5000)

        @stage("3b_blocking")
        def _blk():
            return generate_candidate_pairs_vectorized(
                srcs["source1"], srcs["source2"], srcs["source3"],
                output_candidate_tsv=str(train_cand),
                active_rules=None, max_candidates_per_entity=200,
                max_bucket_size=5000, enable_coarse_filter=True,
                min_coarse_similarity=0.30,
                chunk_size=args.blocking_chunk_size,
                checkpoint_dir=str(ck / "blocking"),
            )
        _blk()
        print(f"    candidate_pairs.tsv = "
              f"{train_cand.stat().st_size/1e6:.1f} MB")

    # ── Stage 4: features ───────────────────────────────────────────
    if want("features"):
        @stage("4_features")
        def _feat():
            return save_features(train_cand, train_dir, "train", tfidf,
                                train_feat, ck / "feat",
                                stage4_chunk_size=args.stage4_chunk_size)
        _feat()
        print(f"    train_features.parquet = "
              f"{train_feat.stat().st_size/1e6:.1f} MB")

    # ── Stage 5: train ──────────────────────────────────────────────
    if want("train"):
        @stage("5_train")
        def _train():
            return train_model(
                train_feat, gt, model_p, W / "shap.json", val_pred,
                {"objective": "binary", "metric": "binary_logloss",
                 "learning_rate": 0.05, "num_leaves": 63, "verbose": -1,
                 "num_threads": os.cpu_count()},
            )
        _train()

    # ── Stage 6: threshold ──────────────────────────────────────────
    if want("threshold"):
        @stage("6_threshold")
        def _thr():
            return sweep_threshold(val_pred, gt, thr_p,
                                   candidate_pairs_path=train_cand)
        _thr()

    # ── Stage 7: inference ──────────────────────────────────────────
    if want("infer"):
        @stage("7_inference")
        def _inf():
            return run_inference(train_feat, model_p, thr_p,
                                 W / "matching_results.tsv",
                                 all_s1_ids=srcs["source1"]["entity_id"].astype(str).tolist())
        _inf()

    print("\n" + "=" * 72)
    print(f"{'stage':<22}{'sec':>10}{'peakRSS MB':>14}{'delta MB':>12}")
    print("-" * 72)
    for r in RESULTS:
        print(f"{r['stage']:<22}{r['sec']:>10.2f}{r['peak_rss_mb']:>14.0f}"
              f"{r['peak_delta_mb']:>12.0f}")
    print("=" * 72)

    if args.json_out:
        args.json_out.write_text(json.dumps(
            {"dataset": {"s1": s1n, "s2": s2n, "s3": s3n},
             "stages": RESULTS}, indent=2))
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
