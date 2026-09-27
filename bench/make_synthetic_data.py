"""
Synthetic dataset generator for BER pipeline benchmarking.

Produces a dataset/ tree with the same schema and the same *shape ratios* as
the real Amazon ML 2026 Business Entity Resolution data, at a controllable
scale factor, so per-stage wall time and peak RSS can be measured locally and
extrapolated to production.

Schema (must match docs/BER_Architecture.md):
  dataset/train/train_source1.tsv  entity_id, business_name, business_address, country
  dataset/train/train_source2.tsv  entity_id, business_name, business_address, country
  dataset/train/train_source3.tsv  entity_id, business_name, business_address, country
  dataset/train/train_ground_truth.tsv  source1_entity_id, matched_entity_ids
  dataset/test/test_source{1,2,3}.tsv   same schema

Names are generated from a small vocabulary so that blocking keys actually
collide at a realistic rate -- uniformly random strings would produce ~zero
candidates and make the benchmark meaningless.
"""

import argparse
import random
from pathlib import Path

# Vocabulary sized so Soundex/first-token/char-ngram blocking keys collide
# often enough to produce a realistic candidate density.
FIRST = [
    "acme", "sunrise", "blue", "red", "global", "prime", "apex", "nova",
    "quantum", "delta", "sigma", "omega", "atlas", "vertex", "zenith",
    "crimson", "golden", "silver", "iron", "steel", "eagle", "falcon",
    "pioneer", "summit", "harbor", "meadow", "ridge", "valley",
]
SECOND = [
    "systems", "solutions", "technologies", "logistics", "holdings",
    "enterprises", "industries", "partners", "group", "labs",
    "dynamics", "works", "trading", "consulting", "analytics",
    "motors", "foods", "textiles", "pharma", "energy",
]
STREET = [
    "main", "sec", "park", "elm", "oak", "pine", "maple", "cedar",
    "high", "station", "market", "church", "mill", "river", "hill",
]
SUFFIX = ["ltd", "llc", "inc", "corp", "gmbh", "plc", "co", "pvt", "srl", "bv"]
CITIES = [
    "mumbai", "delhi", "bengaluru", "chennai", "kolkata", "pune",
    "hyderabad", "ahmedabad", "jaipur", "kochi",
]
COUNTRIES = ["IN", "US", "GB", "DE", "SG", "AE", ""]


def _mutate(word, rng):
    """Apply a realistic typo so 'matching' records are not exact duplicates."""
    if not word:
        return word
    r = rng.random()
    if r < 0.35:
        return word
    if r < 0.55 and len(word) > 3:            # drop a char
        i = rng.randrange(len(word))
        return word[:i] + word[i + 1:]
    if r < 0.75:                              # swap adjacent
        i = rng.randrange(len(word) - 1)
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]
    if r < 0.9 and len(word) > 2:             # double a char
        i = rng.randrange(len(word))
        return word[:i] + word[i] + word[i:]
    return word[:max(1, len(word) - 1)] + chr(ord(word[-1]) + 1)


def _make_name(rng):
    f = rng.choice(FIRST)
    s = rng.choice(SECOND)
    if rng.random() < 0.45:
        s = f"{s} {rng.choice(SUFFIX)}"
    return f"{f} {s}"


def _make_addr(rng):
    return (f"{rng.randrange(1, 400)} {rng.choice(STREET)} "
            f"{rng.choice(('street', 'road', 'ave', 'lane'))}, "
            f"{rng.choice(CITIES)}")


def _gen_source(prefix, n, rng):
    rows = []
    for i in range(n):
        rows.append({
            "entity_id": f"{prefix}-{i:07d}",
            "business_name": _make_name(rng),
            "business_address": _make_addr(rng),
            "country": rng.choice(COUNTRIES),
        })
    return rows


def _write_tsv(path: Path, rows, cols):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(str(r[c]) for c in cols) + "\n")


def generate(out_dir: Path, n_s1: int, n_s2: int, n_s3: int, seed: int = 42,
             match_rate: float = 0.35, max_matches: int = 3):
    rng = random.Random(seed)
    cols = ["entity_id", "business_name", "business_address", "country"]
    out_dir.mkdir(parents=True, exist_ok=True)

    for split, s1n, s2n, s3n in (("train", n_s1, n_s2, n_s3), ("test", n_s1, n_s2, n_s3)):
        d = out_dir / split
        s1 = _gen_source("S1", s1n, random.Random(seed + hash(split) % 1000))
        s2 = _gen_source("S2", s2n, random.Random(seed + 7))
        s3 = _gen_source("S3", s3n, random.Random(seed + 13))

        # Ground truth: for a fraction of S1 entities, point at an S2/S3 record
        # and write a *mutated* copy of that record's name/address into the S1
        # row, so there is real fuzzy-match work to do.
        gt = []
        pool = s2 + s3
        for i, r in enumerate(s1):
            if pool and rng.random() < match_rate:
                k = rng.randint(1, max_matches)
                picks = rng.sample(pool, k)
                ids = []
                for tgt in picks:
                    ids.append(tgt["entity_id"])
                    r["business_name"] = " ".join(
                        _mutate(w, rng) for w in tgt["business_name"].split()
                    )
                    r["business_address"] = _mutate(tgt["business_address"], rng)
                    r["country"] = tgt["country"]
                gt.append({"source1_entity_id": r["entity_id"],
                           "matched_entity_ids": ",".join(ids)})

        _write_tsv(d / f"{split}_source1.tsv", s1, cols)
        _write_tsv(d / f"{split}_source2.tsv", s2, cols)
        _write_tsv(d / f"{split}_source3.tsv", s3, cols)
        if split == "train":
            _write_tsv(d / "train_ground_truth.tsv", gt,
                       ["source1_entity_id", "matched_entity_ids"])
        print(f"  {split}: S1={len(s1):,} S2={len(s2):,} S3={len(s3):,} "
              f"gt_rows={len(gt):,}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n-s1", type=int, default=20000)
    p.add_argument("--n-s2", type=int, default=40000)
    p.add_argument("--n-s3", type=int, default=40000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--match-rate", type=float, default=0.35)
    a = p.parse_args()
    print(f"Generating synthetic BER dataset at {a.out} ...")
    generate(a.out, a.n_s1, a.n_s2, a.n_s3, a.seed, a.match_rate)
    print("done.")


if __name__ == "__main__":
    main()
