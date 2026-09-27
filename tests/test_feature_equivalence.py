"""
Equivalence test: the parallelised Stage 4 `compute_features` must produce the
same numbers as the pre-optimisation implementation.

The reference implementation below is a verbatim copy of the original
`compute_features` (serial TF-IDF in the parent, three intermediate DataFrames,
row-wise `apply(pd.to_numeric)`, float64 output). It is kept here, in the test
suite, precisely so that "we made it faster" can never quietly become "we made
it different".

Run:  python -m pytest tests/test_feature_equivalence.py -v
"""

import multiprocessing as mp

import numpy as np
import pandas as pd
import pytest

from src.features import (
    FEATURE_COLS,
    _init_feature_worker,
    _process_chunk,
    compute_features,
    fit_tfidf_from_current_records,
    load_source_records,
)


# ─────────────────────────────────────────────────────────────
# Reference implementation (pre-optimisation), verbatim
# ─────────────────────────────────────────────────────────────

def _reference_compute_features(pairs_df, tfidf_model):
    if pairs_df.empty:
        return pd.DataFrame(columns=FEATURE_COLS)

    n_cores = max(1, mp.cpu_count() - 1)
    split_positions = np.array_split(np.arange(len(pairs_df)), n_cores * 4)
    chunks = [pairs_df.iloc[idx] for idx in split_positions]
    valid_chunks = [c for c in chunks if not c.empty]

    with mp.Pool(n_cores) as pool:
        results = list(pool.imap(_process_chunk, valid_chunks))

    name_df = pd.concat([r[0] for r in results], ignore_index=True)
    address_df = pd.concat([r[1] for r in results], ignore_index=True)
    conflict_df = pd.concat([r[2] for r in results], ignore_index=True)

    s1_names = pairs_df["s1_business_name"].fillna("").astype(str).tolist()
    cand_names = pairs_df["candidate_business_name"].fillna("").astype(str).tolist()
    s1_addresses = pairs_df["s1_business_address"].fillna("").astype(str).tolist()
    cand_addresses = pairs_df["candidate_business_address"].fillna("").astype(str).tolist()

    name_df["name_tfidf_cosine"] = tfidf_model.row_cosine(
        tfidf_model.transform_names(s1_names),
        tfidf_model.transform_names(cand_names),
    )
    address_df["addr_tfidf_cosine"] = tfidf_model.row_cosine(
        tfidf_model.transform_addresses(s1_addresses),
        tfidf_model.transform_addresses(cand_addresses),
    )

    result = pd.concat([name_df, address_df, conflict_df], axis=1)
    internal_columns = [col for col in result.columns if col.startswith("_")]
    if internal_columns:
        result = result.drop(columns=internal_columns)
    result = result[FEATURE_COLS]
    result = result.apply(pd.to_numeric, errors="coerce")
    result = result.replace([np.inf, -np.inf], np.nan)
    result = result.fillna(0.0)
    return result


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def sample():
    rng = np.random.default_rng(7)
    n = 900
    words = ["acme", "sunrise", "global", "blue", "delta", "nova", "pvt", "ltd",
             "systems", "solutions", "logistics", "mumbai", "delhi", "street"]

    def w():
        k = rng.integers(1, 4)
        return " ".join(rng.choice(words, size=k))

    s1 = pd.DataFrame({
        "entity_id": [f"S1-{i:05d}" for i in range(n)],
        "business_name": [w() for _ in range(n)],
        "business_address": [w() for _ in range(n)],
        "country": rng.choice(["IN", "US", "GB", ""], size=n),
    })
    cand = s1.sample(n=min(n, 400), random_state=3).copy()
    cand["entity_id"] = [f"S2-{i:05d}" for i in range(len(cand))]
    # perturb so features are not all trivially 1.0
    cand["business_name"] = [w() for _ in range(len(cand))]
    cand["business_address"] = [w() for _ in range(len(cand))]

    pairs = pd.DataFrame({
        "source1_entity_id": s1["entity_id"].values[: len(cand)],
        "candidate_entity_id": cand["entity_id"].values,
        "candidate_source": "S2",
        "s1_business_name": s1["business_name"].values[: len(cand)],
        "s1_business_address": s1["business_address"].values[: len(cand)],
        "s1_country": s1["country"].values[: len(cand)],
        "candidate_business_name": cand["business_name"].values,
        "candidate_business_address": cand["business_address"].values,
        "candidate_country": cand["country"].values,
    })
    sources = {
        "source1": s1.set_index("entity_id"),
        "source2": cand.set_index("entity_id"),
        "source3": cand.iloc[:0].set_index("entity_id"),
    }
    tfidf = fit_tfidf_from_current_records(sources)
    return pairs, tfidf


# ─────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────

def test_matches_reference_exactly(sample):
    """Parallel path must equal the original implementation value-for-value."""
    pairs, tfidf = sample
    ref = _reference_compute_features(pairs, tfidf)
    new = compute_features(pairs, tfidf)
    assert list(new.columns) == list(FEATURE_COLS)
    assert len(new) == len(ref)
    np.testing.assert_allclose(
        new.to_numpy(dtype=np.float64),
        ref.to_numpy(dtype=np.float64),
        rtol=0, atol=1e-6,
    )


def test_serial_path_matches_parallel(sample):
    """The no-fork fallback must agree with the forked path."""
    pairs, tfidf = sample
    par = compute_features(pairs, tfidf, workers=8)
    ser = compute_features(pairs, tfidf, workers=1)
    np.testing.assert_allclose(
        par.to_numpy(dtype=np.float64), ser.to_numpy(dtype=np.float64),
        rtol=0, atol=1e-6,
    )


def test_row_blocking_is_transparent(sample):
    """Different rows_per_task splits must not change any value."""
    pairs, tfidf = sample
    a = compute_features(pairs, tfidf, rows_per_task=20_000)
    b = compute_features(pairs, tfidf, rows_per_task=37)
    c = compute_features(pairs, tfidf, rows_per_task=1)
    np.testing.assert_allclose(a.to_numpy(np.float64), b.to_numpy(np.float64),
                               rtol=0, atol=1e-6)
    np.testing.assert_allclose(a.to_numpy(np.float64), c.to_numpy(np.float64),
                               rtol=0, atol=1e-6)


def test_output_is_float32_and_finite(sample):
    pairs, tfidf = sample
    out = compute_features(pairs, tfidf)
    assert out.dtypes.eq(np.float32).all(), "features must be float32 to halve RAM"
    assert np.isfinite(out.to_numpy()).all(), "no inf/NaN may escape"


def test_empty_input():
    empty = pd.DataFrame(columns=[
        "source1_entity_id", "candidate_entity_id", "candidate_source",
        "s1_business_name", "s1_business_address", "s1_country",
        "candidate_business_name", "candidate_business_address",
        "candidate_country",
    ])
    from src.features import TFIDFSimilarityModel
    out = compute_features(empty, TFIDFSimilarityModel())
    assert out.empty
    assert list(out.columns) == list(FEATURE_COLS)
