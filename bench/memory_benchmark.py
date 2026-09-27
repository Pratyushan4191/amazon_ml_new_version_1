"""
Per-stage peak-memory benchmark.

Runs each pipeline stage in its OWN subprocess and reports that child's true
peak RSS (`ru_maxrss`), so a stage's peak is not contaminated by whatever
earlier stages left in the allocator. `bench/profile_stages.py` reports
in-process peaks and is the right tool for timing; this one is the right tool
for the question "how much RAM does stage N actually need", which is what
determines whether the run survives on a given instance size.

Extrapolation: peak RSS is close to linear in candidate-pair count for stages 3
and 5, so multiplying the measured peak by (production_pairs / measured_pairs)
gives a defensible estimate for a larger dataset. Stage 4 is bounded by
chunk size, not by total pairs, so it should NOT be extrapolated that way.

Usage:
    python bench/memory_benchmark.py --dataset-dir bench/data
    python bench/memory_benchmark.py --dataset-dir bench/data --only train
"""

import argparse
import json
import os

import shutil
import subprocess
import textwrap
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Each entry: name -> python source executed in a fresh subprocess.
STAGE_SNIPPETS = {
    "load": """
        from src.features import load_source_records
        s = load_source_records(D/'train', split='train')
        print('ROWS', len(s['source1']), len(s['source2']), len(s['source3']))
    """,
    "tfidf": """
        import pickle
        from src.features import load_source_records, fit_tfidf_from_current_records
        s = load_source_records(D/'train', split='train')
        m = fit_tfidf_from_current_records(s)
        pickle.dump(m, open(W/'tfidf.pkl','wb'))
        print('OK')
    """,
    "index": """
        from src.features import load_source_records
        from src.blocking_vectorized import build_index_table, _filter_hot_buckets
        s = load_source_records(D/'train', split='train')
        it, mt = build_index_table(s['source2'], s['source3'], active_rules=None)
        it = _filter_hot_buckets(it, 5000)
        print('INDEX', len(it), it.memory_usage(deep=True).sum())
    """,
    "blocking": """
        from src.features import load_source_records
        from src.blocking_vectorized import generate_candidate_pairs_vectorized
        s = load_source_records(D/'train', split='train')
        r = generate_candidate_pairs_vectorized(
            s['source1'], s['source2'], s['source3'],
            output_candidate_tsv=str(W/'cands.tsv'), active_rules=None,
            max_candidates_per_entity=200, max_bucket_size=5000,
            enable_coarse_filter=True, min_coarse_similarity=0.30,
            chunk_size=2000, checkpoint_dir=str(W/'ckpt'), resume=False,
            logger=lambda *a, **k: None)
        print('CANDS', r['total_candidates'])
    """,
    "features": """
        import pickle
        from run_pipeline import save_features
        tf = pickle.load(open(W/'tfidf.pkl','rb'))
        r = save_features(W/'cands.tsv', D/'train', 'train', tf,
                          W/'feats.parquet', W/'fckpt', stage4_chunk_size=2000)
        print('FEATROWS', r['n_rows'])
    """,
    "train": """
        import os, inspect
        from src.train import train_model
        from pathlib import Path
        kw = {}
        # `verbose_mem` only exists in the current implementation; passing it
        # unconditionally would make this harness unable to measure the
        # baseline it is meant to compare against.
        if 'verbose_mem' in inspect.signature(train_model).parameters:
            kw['verbose_mem'] = False
        r = train_model(W/'feats.parquet', D/'train'/'train_ground_truth.tsv',
            W/'model.lgb', W/'shap.json', W/'val.parquet',
            {'objective':'binary','metric':'binary_logloss','learning_rate':0.05,
             'num_leaves':63,'verbose':-1,'num_threads':os.cpu_count()}, **kw)
        print('TRAINROWS', r.get('n_rows', -1))
    """,
    "threshold": """
        from src.threshold import sweep_threshold
        r = sweep_threshold(W/'val.parquet', D/'train'/'train_ground_truth.tsv',
                            W/'thr.json', candidate_pairs_path=W/'cands.tsv')
        print('THR', r)
    """,
    "predict": """
        from src.predict import run_inference
        import pandas as pd
        s1 = pd.read_csv(D/'train'/'train_source1.tsv', sep='\\t', dtype=str,
                        keep_default_na=False)['entity_id'].tolist()
        run_inference(W/'feats.parquet', W/'model.lgb', W/'thr.json',
                      W/'matches.tsv', all_s1_ids=s1)
        print('PRED')
    """,
}

# Rough production scale, for the extrapolation column.
PRODUCTION_PAIRS = 86_000_000


def run_stage(name, dataset_dir, work_dir, timeout=7200):
    work_dir.mkdir(parents=True, exist_ok=True)
    # The child reports its OWN ru_maxrss. Reading
    # getrusage(RUSAGE_CHILDREN).ru_maxrss from the parent does not work: that
    # is a high-water mark across every child the process has ever reaped, so
    # it only ever ratchets upward and every stage after the biggest one
    # reports that same number instead of its own.
    footer = (
        "\nimport resource as _r\n"
        "print('__PEAK_KB__', _r.getrusage(_r.RUSAGE_SELF).ru_maxrss)\n"
    )
    src = "\n".join([
        "import sys, os",
        f"sys.path.insert(0, {str(PROJECT_ROOT)!r})",
        "from pathlib import Path",
        f"D = Path({str(dataset_dir)!r})",
        f"W = Path({str(work_dir)!r})",
        f"os.chdir({str(PROJECT_ROOT)!r})",
        textwrap.dedent(STAGE_SNIPPETS[name]).strip(),
        footer,
        "",
    ])
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, "-c", src],
        capture_output=True, text=True, timeout=timeout,
    )
    dt = time.time() - t0
    if proc.returncode != 0:
        return {
            "stage": name, "ok": False, "sec": round(dt, 1),
            "peak_rss_mb": 0.0,
            "stderr": proc.stderr.strip().splitlines()[-6:],
        }
    peak_kb = 0
    tail = ""
    for line in proc.stdout.strip().splitlines():
        line = line.strip()
        if line.startswith("__PEAK_KB__"):
            peak_kb = int(line.split()[1])
        elif line:
            tail = line
    return {
        "stage": name, "ok": True, "sec": round(dt, 1),
        "peak_rss_mb": round(peak_kb / 1024, 0),
        "output": tail,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", type=Path, required=True)
    ap.add_argument("--work-dir", type=Path, default=PROJECT_ROOT / "bench" / "memwork")
    ap.add_argument("--only", default=None)
    ap.add_argument("--json-out", type=Path, default=None)
    ap.add_argument("--measured-pairs", type=int, default=None,
                    help="candidate pair count, for the extrapolation column")
    ap.add_argument("--keep-work", action="store_true",
                    help="do not wipe the work dir first; required when using "
                         "--only to re-measure a later stage against the "
                         "artifacts an earlier run left behind")
    a = ap.parse_args()

    if a.work_dir.exists():
        if a.keep_work:
            print(f"Reusing existing work dir {a.work_dir} (--keep-work)")
        else:
            shutil.rmtree(a.work_dir)
    a.work_dir.mkdir(parents=True, exist_ok=True)

    order = ["load", "tfidf", "index", "blocking", "features", "train",
             "threshold", "predict"]
    if a.only:
        order = [s for s in order if s in set(a.only.split(","))]

    results = []
    for name in order:
        r = run_stage(name, a.dataset_dir, a.work_dir)
        results.append(r)
        status = "ok " if r["ok"] else "FAIL"
        print(f"{status} {name:10s} {r['sec']:>8.1f}s  peakRSS={r['peak_rss_mb']:>7.0f}MB  "
              f"{r.get('output','')}", flush=True)
        if not r["ok"]:
            for line in r.get("stderr", []):
                print("      " + line)

    scale = None
    if a.measured_pairs:
        scale = PRODUCTION_PAIRS / a.measured_pairs

    print("\n" + "=" * 78)
    print(f"{'stage':<12}{'sec':>9}{'peak RSS MB':>14}" +
          (f"{'~prod @86M pairs':>20}" if scale else ""))
    print("-" * 78)
    for r in results:
        line = f"{r['stage']:<12}{r['sec']:>9.1f}{r['peak_rss_mb']:>14.0f}"
        # Stage 4 is bounded by chunk size, not total pairs -- do not mislead.
        if scale and r["stage"] in ("blocking", "features", "train", "threshold", "predict"):
            line += f"{r['peak_rss_mb'] * scale / 1000:>17.0f} GB"
        print(line)
    if scale:
        print("\nNote: extrapolation is linear in candidate pairs and is only a")
        print("      rough guide. 'features' is chunk-bounded, so its true")
        print("      production peak should be close to the measured value.")
    print("=" * 78)

    if a.json_out:
        a.json_out.write_text(json.dumps(results, indent=2))
        print(f"wrote {a.json_out}")


if __name__ == "__main__":
    main()
