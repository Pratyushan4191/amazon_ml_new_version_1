"""
The parallel blocking path must produce byte-identical output to the
single-process path.

The pre-existing fixtures in `test_blocking_vectorized.py` are deliberately
tiny (3 chunks of 100 rows). That is below the threshold at which
`src.parallel` will fork a pool at all, so those tests only ever exercised the
serial fallback -- they could not have caught a divergence introduced by
parallelising the chunk loop. These tests force the pool path on (by dropping
the serial threshold to 0) and compare outputs directly.

Run:  python -m pytest tests/test_blocking_parallel_equivalence.py -v
"""

import shutil

import pandas as pd
import pytest

from src import parallel
from src.blocking_vectorized import (
    build_index_table,
    _filter_hot_buckets,
    generate_candidate_pairs_vectorized,
    load_candidate_mapping_from_tsv,
)


def _synthetic(n_s1=400, n_s2=900, n_s3=900, hot_frac=0.5, seed=11):
    import random
    random.seed(seed)
    countries = ["US", "IN", "DE", ""]

    def nm(i, hot):
        if hot:
            return "GLOBAL LOGISTICS INC"
        return f"{random.choice(['ACME','NORTH','BLUE','ZETA'])} {random.choice(['TRADING','FOODS','SYSTEMS'])} {i}"

    def ad(i):
        return (f"{random.randint(1,999)} {random.choice(['MAIN','OAK','PARK'])} ST, "
                f"{random.choice(['SPRINGFIELD','RIVERTON'])}")

    s1_rows, recs = [], []
    for i in range(n_s1):
        c = random.choice(countries)
        recs.append((nm(i, False), ad(i), c))
        s1_rows.append({"entity_id": f"S1-{i:06d}", "business_name": recs[-1][0],
                        "business_address": recs[-1][1], "country": c})

    s2_rows, s3_rows = [], []
    for i in range(n_s2):
        s2_rows.append({"entity_id": f"S2-{i:06d}",
                        "business_name": nm(i, random.random() < hot_frac),
                        "business_address": ad(i),
                        "country": random.choice(countries)})
    for i in range(n_s3):
        s3_rows.append({"entity_id": f"S3-{i:06d}",
                        "business_name": nm(i, random.random() < hot_frac),
                        "business_address": ad(i),
                        "country": random.choice(countries)})
    return pd.DataFrame(s1_rows), pd.DataFrame(s2_rows), pd.DataFrame(s3_rows)


def _run(tmp_path, s1, s2, s3, workers, tag, **kw):
    out = tmp_path / f"cands_{tag}.tsv"
    ckpt = tmp_path / f"ckpt_{tag}"
    shutil.rmtree(ckpt, ignore_errors=True)
    generate_candidate_pairs_vectorized(
        s1, s2, s3, output_candidate_tsv=str(out),
        chunk_size=50, checkpoint_dir=str(ckpt), resume=False,
        workers=workers, logger=lambda *a, **k: None, **kw,
    )
    return out


@pytest.fixture
def data():
    return _synthetic()


def test_parallel_matches_serial_bytes(tmp_path, data, monkeypatch):
    """Force the pool on, then compare the exported TSV byte-for-byte."""
    s1, s2, s3 = data
    serial_out = _run(tmp_path, s1, s2, s3, workers=1, tag="serial")

    monkeypatch.setattr(parallel, "SERIAL_THRESHOLD", 0)
    par_out = _run(tmp_path, s1, s2, s3, workers=4, tag="par")

    assert serial_out.read_text() == par_out.read_text()


@pytest.fixture
def forced_pool(monkeypatch):
    monkeypatch.setattr(parallel, "SERIAL_THRESHOLD", 0)


@pytest.mark.parametrize("mcpe", [1, 5, 200])
def test_parallel_matches_serial_under_caps(tmp_path, data, forced_pool, mcpe):
    """The per-chunk candidate cap must hold identically on both paths."""
    s1, s2, s3 = data
    kw = {"max_candidates_per_entity": mcpe, "enable_coarse_filter": True,
          "min_coarse_similarity": 0.30, "max_bucket_size": 5000}
    serial = _run(tmp_path, s1, s2, s3, workers=1, tag=f"s{mcpe}", **kw)
    par = _run(tmp_path, s1, s2, s3, workers=4, tag=f"p{mcpe}", **kw)
    assert serial.read_text() == par.read_text()

    m = load_candidate_mapping_from_tsv(str(par))
    assert all(len(v) <= mcpe for v in m.values())


def test_parallel_respects_hot_key_filter(tmp_path, data, forced_pool):
    """With a low bucket cap, hot buckets are dropped -- same on both paths."""
    s1, s2, s3 = data
    kw = {"max_bucket_size": 20, "max_candidates_per_entity": 200}
    serial = _run(tmp_path, s1, s2, s3, workers=1, tag="hots", **kw)
    par = _run(tmp_path, s1, s2, s3, workers=4, tag="hotp", **kw)
    assert serial.read_text() == par.read_text()


def test_every_s1_present_exactly_once(tmp_path, data, forced_pool):
    """The 'every S1 id present' contract must survive parallel execution."""
    s1, s2, s3 = data
    out = _run(tmp_path, s1, s2, s3, workers=4, tag="cover")
    got = pd.read_csv(out, sep="\t", dtype=str, keep_default_na=False)
    assert list(got["source1_entity_id"]) == list(s1["entity_id"].astype(str))


def test_checkpoint_resume_under_pool(tmp_path, data, forced_pool):
    """Killing the run half way and resuming must reproduce the full output."""
    s1, s2, s3 = data
    full = _run(tmp_path, s1, s2, s3, workers=4, tag="full")

    # Simulate a crash: build only the first half of the chunks, then resume.
    ckpt = tmp_path / "ckpt_resume"
    shutil.rmtree(ckpt, ignore_errors=True)
    import src.blocking_vectorized as bv
    real_worker = bv._blocking_chunk_worker
    state = {"n": 0}

    def flaky(task):
        state["n"] += 1
        if state["n"] > 4:
            raise RuntimeError("simulated crash")
        return real_worker(task)

    monkey = pytest.MonkeyPatch()
    monkey.setattr(bv, "_blocking_chunk_worker", flaky)
    with pytest.raises(Exception):
        bv.generate_candidate_pairs_vectorized(
            s1, s2, s3, output_candidate_tsv=str(tmp_path / "partial.tsv"),
            chunk_size=50, checkpoint_dir=str(ckpt), resume=False,
            workers=4, logger=lambda *a, **k: None,
        )
    monkey.undo()

    resumed = tmp_path / "resumed.tsv"
    bv.generate_candidate_pairs_vectorized(
        s1, s2, s3, output_candidate_tsv=str(resumed),
        chunk_size=50, checkpoint_dir=str(ckpt), resume=True,
        workers=4, logger=lambda *a, **k: None,
    )
    assert resumed.read_text() == full.read_text()


def test_hot_bucket_filter_drops_oversized_buckets():
    """
    Regression: `_filter_hot_buckets` used to raise
    `AttributeError: 'numpy.ndarray' object has no attribute 'values'`
    as soon as any bucket reached the cap -- i.e. on every real production
    run, but on no small fixture.
    """
    idx = pd.DataFrame({
        "country": ["US"] * 6,
        "rule": ["A"] * 6,
        "key": ["hot"] * 5 + ["cold"],
        "entity_id": [f"S2-{i}" for i in range(6)],
    })
    idx["country"] = idx["country"].astype("category")
    idx["rule"] = idx["rule"].astype("category")

    out = _filter_hot_buckets(idx, max_bucket_size=5)
    # bucket "hot" has 5 rows -> >= cap -> entirely excluded
    assert list(out["key"]) == ["cold"]

    out2 = _filter_hot_buckets(idx, max_bucket_size=6)
    assert len(out2) == 6, "nothing should be dropped when no bucket reaches the cap"
