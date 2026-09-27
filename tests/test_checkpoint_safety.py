"""
Resume-safety guard for the vectorized blocking checkpoint.

The failure this guards against is silent. The manifest stores chunk INDICES,
so resuming with a different `chunk_size` re-bounds the S1 rows: chunk 7 of the
old run covered S1 rows 7000-7999, chunk 7 of the new run covers 14000-14999.
Nothing raises, the pipeline finishes, and the candidate set is quietly wrong.

`run_pipeline.py --help` advertises `CheckpointConfigError`; before this was
implemented the name appeared in the help text and nowhere else, and
`generate_candidate_pairs_vectorized` did not even accept the
`force_resume_incompatible` argument the entrypoint passed it -- so the
pipeline could not start at all.

Run:  python -m pytest tests/test_checkpoint_safety.py -v
"""

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from src.blocking_vectorized import (
    CheckpointConfigError,
    generate_candidate_pairs_vectorized,
    load_candidate_mapping_from_tsv,
)


def _data(n_s1=120, n_s2=240, n_s3=240, seed=13):
    import random
    random.seed(seed)
    countries = ["US", "IN", "DE"]

    def nm(i):
        return f"{random.choice(['ACME','NORTH','BLUE'])} {random.choice(['TRADING','FOODS'])} {i}"

    s1 = pd.DataFrame([
        {"entity_id": f"S1-{i:06d}", "business_name": nm(i),
         "business_address": f"{i} MAIN ST, SPRINGFIELD",
         "country": random.choice(countries)}
        for i in range(n_s1)
    ])
    s2 = pd.DataFrame([
        {"entity_id": f"S2-{i:06d}", "business_name": nm(i),
         "business_address": f"{i} OAK ST, RIVERTON",
         "country": random.choice(countries)}
        for i in range(n_s2)
    ])
    s3 = pd.DataFrame([
        {"entity_id": f"S3-{i:06d}", "business_name": nm(i),
         "business_address": f"{i} PARK ST, RIVERTON",
         "country": random.choice(countries)}
        for i in range(n_s3)
    ])
    return s1, s2, s3


def _run(s1, s2, s3, out, ckpt, **kw):
    return generate_candidate_pairs_vectorized(
        s1, s2, s3, output_candidate_tsv=str(out),
        chunk_size=kw.pop("chunk_size", 30),
        checkpoint_dir=str(ckpt), resume=kw.pop("resume", True),
        logger=lambda *a, **k: None, **kw,
    )


def test_manifest_records_config_and_fingerprint(tmp_path):
    s1, s2, s3 = _data()
    ck = tmp_path / "ck"
    _run(s1, s2, s3, tmp_path / "o.tsv", ck)
    manifest = json.loads((ck / "manifest.json").read_text())
    assert manifest["config"]["chunk_size"] == 30
    assert "input_fingerprint" in manifest


def test_resume_with_different_chunk_size_is_refused(tmp_path):
    """The dangerous case: chunk indices no longer mean the same S1 rows."""
    s1, s2, s3 = _data()
    ck = tmp_path / "ck"
    _run(s1, s2, s3, tmp_path / "o.tsv", ck, chunk_size=30)

    with pytest.raises(CheckpointConfigError) as exc:
        _run(s1, s2, s3, tmp_path / "o2.tsv", ck, chunk_size=50)
    msg = str(exc.value)
    assert "chunk_size" in msg
    assert "force-resume-incompatible" in msg


def test_force_resume_incompatible_overrides_the_refusal(tmp_path):
    s1, s2, s3 = _data()
    ck = tmp_path / "ck"
    _run(s1, s2, s3, tmp_path / "o.tsv", ck, chunk_size=30)
    # Must not raise.
    _run(s1, s2, s3, tmp_path / "o2.tsv", ck, chunk_size=50,
         force_resume_incompatible=True)


@pytest.mark.parametrize("changed", [
    {"max_candidates_per_entity": 5},
    {"max_bucket_size": 50},
    {"enable_coarse_filter": False},
    {"min_coarse_similarity": 0.9},
])
def test_other_config_changes_are_refused(tmp_path, changed):
    s1, s2, s3 = _data()
    ck = tmp_path / "ck"
    _run(s1, s2, s3, tmp_path / "o.tsv", ck)
    with pytest.raises(CheckpointConfigError):
        _run(s1, s2, s3, tmp_path / "o2.tsv", ck, **changed)


def test_resume_with_different_input_data_is_refused(tmp_path):
    s1, s2, s3 = _data()
    ck = tmp_path / "ck"
    _run(s1, s2, s3, tmp_path / "o.tsv", ck)

    s1b, s2b, s3b = _data(n_s1=180)      # different S1 population
    with pytest.raises(CheckpointConfigError) as exc:
        _run(s1b, s2b, s3b, tmp_path / "o2.tsv", ck)
    assert "input_fingerprint" in str(exc.value)


def test_resume_with_different_s2_content_is_refused(tmp_path):
    s1, s2, s3 = _data()
    ck = tmp_path / "ck"
    _run(s1, s2, s3, tmp_path / "o.tsv", ck)

    s2b = s2.copy()
    s2b.loc[0, "business_name"] = "COMPLETELY DIFFERENT S2 NAME"
    with pytest.raises(CheckpointConfigError) as exc:
        _run(s1, s2b, s3, tmp_path / "o2.tsv", ck)
    assert "input_fingerprint" in str(exc.value)


def test_resume_with_different_s3_content_is_refused(tmp_path):
    s1, s2, s3 = _data()
    ck = tmp_path / "ck"
    _run(s1, s2, s3, tmp_path / "o.tsv", ck)

    s3b = s3.copy()
    s3b.loc[0, "business_name"] = "COMPLETELY DIFFERENT S3 NAME"
    with pytest.raises(CheckpointConfigError) as exc:
        _run(s1, s2, s3b, tmp_path / "o2.tsv", ck)
    assert "input_fingerprint" in str(exc.value)


def test_resume_with_modified_s1_content_same_len_is_refused(tmp_path):
    s1, s2, s3 = _data()
    ck = tmp_path / "ck"
    _run(s1, s2, s3, tmp_path / "o.tsv", ck)

    s1b = s1.copy()
    s1b.loc[0, "business_name"] = "ALTERED BUSINESS NAME IN S1"
    with pytest.raises(CheckpointConfigError) as exc:
        _run(s1b, s2, s3, tmp_path / "o2.tsv", ck)
    assert "input_fingerprint" in str(exc.value)


def test_resume_with_identical_config_is_allowed(tmp_path):
    """The normal interrupted-and-restarted case must still work."""
    s1, s2, s3 = _data()
    ck = tmp_path / "ck"
    full = tmp_path / "full.tsv"
    _run(s1, s2, s3, full, tmp_path / "ck_full", chunk_size=30)

    # Same config, same input -> same output, no refusal.
    again = tmp_path / "again.tsv"
    _run(s1, s2, s3, again, tmp_path / "ck_full", chunk_size=30)
    assert full.read_text() == again.read_text()


def test_partial_checkpoint_then_identical_resume(tmp_path):
    """Truncate the manifest, then resume under the same config."""
    s1, s2, s3 = _data()
    ck = tmp_path / "ck"
    full = tmp_path / "full.tsv"
    _run(s1, s2, s3, full, tmp_path / "ck_ref", chunk_size=30)

    # Produce a partial checkpoint, then rewind the manifest to half done.
    _run(s1, s2, s3, tmp_path / "part.tsv", ck, chunk_size=30)
    man = json.loads((ck / "manifest.json").read_text())
    all_chunks = sorted(man["completed_chunks"])
    man["completed_chunks"] = all_chunks[: len(all_chunks) // 2]
    (ck / "manifest.json").write_text(json.dumps(man))

    resumed = tmp_path / "resumed.tsv"
    _run(s1, s2, s3, resumed, ck, chunk_size=30)
    assert resumed.read_text() == full.read_text()


def test_old_manifest_without_config_still_resumes(tmp_path):
    """A manifest written before this guard existed must not be rejected."""
    s1, s2, s3 = _data()
    ck = tmp_path / "ck"
    _run(s1, s2, s3, tmp_path / "o.tsv", ck, chunk_size=30)
    man = json.loads((ck / "manifest.json").read_text())
    del man["config"]
    del man["input_fingerprint"]
    man["completed_chunks"] = []
    (ck / "manifest.json").write_text(json.dumps(man))
    # No prior config to contradict -> allowed.
    _run(s1, s2, s3, tmp_path / "o2.tsv", ck, chunk_size=30)
