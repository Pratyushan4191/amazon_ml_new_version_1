# archive/ — NON-EXECUTABLE

Nothing in this directory is imported by the production pipeline. There is no
`__init__.py` here, and the project root is the only directory on `sys.path`,
so Python cannot import any module from this tree.

## pre_integration/

Snapshots of files that existed at the repository root alongside a competing
`src/` implementation. On 2026-09-27 these were promoted into `src/` (or, for
the entrypoint, into the project root) and the root copies were moved here.
Each promoted file is byte-identical to its archived copy except where noted.

| Archived file | Promoted to | Note |
|---|---|---|
| `features(1).py` | `src/features.py` | superset: adds `_explode_candidate_pairs`, `iter_exploded_candidate_chunks`, vectorized `attach_records`, `.iloc`-based chunk split |
| `train(1).py` | `src/train.py` | superset: vectorized ground-truth labeling, `del`+`gc.collect()` of dead frames, removed redundant `val_df.copy()` |
| `threshold.py` | `src/threshold.py` | superset: adds `candidate_pairs_path`, widens sweep population to include zero-candidate S1 entities |
| `blocking_vectorized.py` | `src/blocking_vectorized.py` | **was missing from `src/` entirely**; `run_pipeline.py:15` could not import it |
| `monitoring.py` | `src/monitoring.py` | **was missing from `src/` entirely**; imported as `src.monitoring` by `src/blocking_vectorized.py` |
| `normalization.py` | — | byte-identical to `src/normalization.py`; pure duplicate, no promote needed |
| `run_pipeline(2).py` | `run_pipeline.py` | **the entrypoint `run_pipeline.py` had been deleted** (renamed to `run_pipeline(2).py`); project had no runnable entrypoint |
| `FAKE_JaroWinkler_stub__init__.py` | — | **do not restore.** A fake `JaroWinkler` stand-in written because the authoring sandbox had no network to `pip install rapidfuzz`. Its similarity scores are meaningless. It sat at the project root, where it could shadow real imports and silently corrupt any measurement taken while it was present. |
| `newmodel.zip` | — | stale build archive, superseded by the working tree |

## Rule going forward

Every production module lives under `src/` except the entrypoint
`run_pipeline.py`. If you generate an experimental variant, put it in
`archive/<topic>/`, never at the repository root.

## `test_stage4_sandbox_scratch.py`

Was `test_stage4.py` at the repository root. **Not a pytest test** — it is an
authoring scratch script that runs at import time and:

- replaces `pyarrow` and `tqdm` in `sys.modules` with hand-written stubs
  (`sys.modules["pyarrow"] = fake_pyarrow`, `sys.modules["tqdm"] = ...`);
- calls `os.chdir("/home/claude/project")`, a path from the authoring sandbox
  that does not exist on any real machine;
- reads fixtures from `/home/claude/synthtest/`, also nonexistent.

Collected by pytest it would install a fake pyarrow for the *entire session*,
silently breaking every other test that touches parquet. It was archived rather
than repaired. The equivalence it was reaching for — streamed chunked Stage 4
versus whole-file Stage 4 — is now covered properly, with no global stubs, by
`tests/test_feature_equivalence.py`.
