from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from rapidfuzz.distance import JaroWinkler
from rapidfuzz.fuzz import token_set_ratio, token_sort_ratio

from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as sp

from src.normalization import (
    normalize_name,
    normalize_address,
    normalize_country,
    LEGAL_SUFFIXES,
)


# ============================================================
# CONFIGURATION
# ============================================================

REQUIRED_COLUMNS = [
    "entity_id",
    "business_name",
    "business_address",
    "country",
]

FEATURE_COLS = [
    # Name features
    "name_exact",
    "name_exact_no_suffix",
    "name_sorted_exact",
    "name_edit_dist",
    "name_jaro_winkler",
    "name_jaro_winkler_ns",
    "name_token_jaccard",
    "name_token_set_ratio",
    "name_token_sort_ratio",
    "name_char3gram_jaccard",
    "name_char4gram_jaccard",
    "name_len_diff",
    "name_tfidf_cosine",

    # Address features
    "addr_s1_null",
    "addr_cand_null",
    "addr_both_null",
    "addr_exact",
    "addr_token_jaccard",
    "addr_char3gram_jaccard",
    "addr_house_num_agree",
    "addr_city_agree",
    "addr_pin_agree",
    "addr_tfidf_cosine",

    # Country + conflict
    "country_match",
    "high_name_low_addr",
    "name_sim_x_addr_sim",
    "n_blocking_rules_hit",
    "both_have_address",
    "name_suffix_conflict",
]


# ============================================================
# STAGE 1 — INPUT LOADING
# ============================================================

def load_source_records(
    data_dir: Path,
    split: str = "test",
) -> Dict[str, pd.DataFrame]:
    """
    Load S1, S2 and S3 TSV files.

    Each dataframe is indexed by entity_id.
    """

    if split not in {"train", "test"}:
        raise ValueError("split must be either 'train' or 'test'")

    sources = {}

    for source in ["source1", "source2", "source3"]:
        path = data_dir / f"{split}_{source}.tsv"

        if not path.exists():
            raise FileNotFoundError(
                f"Required source file not found: {path}"
            )

        df = pd.read_csv(path, sep="\t", dtype=str)

        missing = [
            col for col in REQUIRED_COLUMNS
            if col not in df.columns
        ]

        if missing:
            raise ValueError(
                f"{path} is missing columns: {missing}"
            )

        df = df[REQUIRED_COLUMNS].copy()

        df["entity_id"] = df["entity_id"].fillna("").astype(str)
        df["business_name"] = df["business_name"].fillna("")
        df["business_address"] = df["business_address"].fillna("")
        df["country"] = df["country"].fillna("")

        df = df.set_index("entity_id", drop=False)

        sources[source] = df

    return sources


def load_candidate_pairs(
    candidate_path: Path,
) -> pd.DataFrame:
    """
    Expand candidate_pairs.tsv from:

        source1_entity_id    candidate_entity_ids

    into one row per candidate pair.
    """

    if not candidate_path.exists():
        raise FileNotFoundError(
            f"Candidate file not found: {candidate_path}"
        )

    df = pd.read_csv(
        candidate_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    required = [
        "source1_entity_id",
        "candidate_entity_ids",
    ]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Candidate file missing columns: {missing}"
        )

    return _explode_candidate_pairs(df)


def _explode_candidate_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Shared explode logic for a raw candidate-pairs dataframe (one row per S1
    entity, a comma-joined `candidate_entity_ids` string column) -> one row
    per (S1, candidate) pair.

    Factored out of `load_candidate_pairs` (P0-3 Stage-4 streaming fix) so
    that both the whole-file path (`load_candidate_pairs`, still used by
    small-scale/manual callers such as `generate_feature_matrix`'s
    `__main__` demo) and the new bounded-chunk path
    (`iter_exploded_candidate_chunks`, used by `run_pipeline.py:save_features`
    for the real production-scale run) share byte-identical explode/validation
    logic -- there is exactly one place this can drift.

    Vectorized (no per-row Python loop): a real memory/CPU bottleneck at
    production scale; verified byte-identical to a per-row reference
    implementation on tiny fixtures.
    """
    non_empty = df[df["candidate_entity_ids"].astype(str).str.len() > 0]
    if non_empty.empty:
        return pd.DataFrame(
            columns=["source1_entity_id", "candidate_entity_id", "candidate_source"]
        )

    exploded = non_empty.assign(
        candidate_entity_id=non_empty["candidate_entity_ids"].str.split(",")
    ).explode("candidate_entity_id", ignore_index=True)
    exploded["candidate_entity_id"] = exploded["candidate_entity_id"].str.strip()
    exploded = exploded[exploded["candidate_entity_id"] != ""].reset_index(drop=True)

    is_s2 = exploded["candidate_entity_id"].str.startswith("S2-")
    is_s3 = exploded["candidate_entity_id"].str.startswith("S3-")
    unknown = ~(is_s2 | is_s3)
    if unknown.any():
        bad_id = exploded.loc[unknown, "candidate_entity_id"].iloc[0]
        raise ValueError(
            f"Unknown candidate ID prefix: {bad_id}"
        )

    exploded["candidate_source"] = np.where(is_s2, "S2", "S3")

    return exploded[
        ["source1_entity_id", "candidate_entity_id", "candidate_source"]
    ].reset_index(drop=True)


def iter_exploded_candidate_chunks(
    candidate_path: Path,
    chunk_size: int = 2_000,
):
    """
    P0-3 Stage-4 memory fix.

    Read `candidate_pairs.tsv` (one row per S1 entity, comma-joined
    candidate list) INCREMENTALLY via `pandas.read_csv(..., chunksize=...)`,
    `chunk_size` raw S1 rows at a time, and yield one EXPLODED (one row per
    S1/candidate pair) dataframe per raw chunk.

    This is the key difference from `load_candidate_pairs`: that function
    reads the entire file into memory before exploding, which at production
    scale (~86M candidate pairs, per docs/BER_Architecture.md Section 23)
    means the full exploded pair table -- tens of millions of rows -- is
    resident in RAM before Stage 4 even starts computing features. This
    generator instead never has more than `chunk_size` raw S1 rows' worth of
    exploded pairs in memory at once, independent of candidate density per
    entity, and independent of CPU core count (unlike the old
    `n_chunks = n_cores * 4` split, which sized chunks by however many rows
    happened to land in 1/(n_cores*4) of the *already-fully-materialized*
    attached dataframe -- itself only possible after the OOM-prone full load
    this function replaces).

    Chunk sizing is a pure row-count budget on the RAW (pre-explode) file,
    so it is a direct, predictable memory knob: raw rows -> resulting
    exploded-pair count depends on candidates/entity, which is itself capped
    upstream by Stage 3's `max_candidates_per_entity`, so a given
    `chunk_size` has a computable worst-case exploded row count of
    `chunk_size * max_candidates_per_entity`.

    Each yielded chunk has the same columns/dtypes/validation as
    `load_candidate_pairs`'s return value (produced by the same
    `_explode_candidate_pairs` helper), so it is a drop-in per-chunk
    replacement wherever the old code did
    `attach_records(load_candidate_pairs(path), sources)` in one shot.

    A raw chunk whose S1 entities all have empty `candidate_entity_ids`
    (e.g. an early chunk of an input dominated by singletons) explodes to
    zero rows; this generator SKIPS yielding empty chunks (nothing to
    attach/compute features for) rather than yielding a 0-row dataframe --
    callers that need to account for zero-candidate S1 entities in a report
    or population count should do so from the raw candidate_pairs.tsv
    directly (as `threshold.py`'s P1 fix already does), not from this
    generator, which is a features-only intermediate representation.
    """
    if not Path(candidate_path).exists():
        raise FileNotFoundError(f"Candidate file not found: {candidate_path}")

    reader = pd.read_csv(
        candidate_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=chunk_size,
    )

    required = {"source1_entity_id", "candidate_entity_ids"}
    for raw_chunk in reader:
        missing = required - set(raw_chunk.columns)
        if missing:
            raise ValueError(f"Candidate file missing columns: {sorted(missing)}")
        exploded = _explode_candidate_pairs(raw_chunk)
        if exploded.empty:
            continue
        yield exploded


_CAND_RENAME = {
    "business_name": "candidate_business_name",
    "business_address": "candidate_business_address",
    "country": "candidate_country",
}
_S1_RENAME = {
    "business_name": "s1_business_name",
    "business_address": "s1_business_address",
    "country": "s1_country",
}
_ATTACH_COL_ORDER = [
    "source1_entity_id", "candidate_entity_id", "candidate_source",
    "s1_business_name", "s1_business_address", "s1_country",
    "candidate_business_name", "candidate_business_address", "candidate_country",
]


class RecordLookup:
    """
    Pre-built, immutable join tables for `attach_records`.

    `attach_records` is called once per Stage 4 chunk, but both of its merge
    keys are the *full* S1 and S2+S3 record sets -- the chunk only changes
    which rows of those tables are needed, not the tables themselves. Building
    the joined lookup inside every call meant re-slicing and re-concatenating
    the entire S2+S3 table on every chunk: O(|S2| + |S3|) work repeated
    `ceil(|S1| / stage4_chunk_size)` times.

    At production scale that is ~10.3M candidate records reconstructed and
    re-concatenated >1,100 times per split. Constructing this once and reusing
    it turns per-chunk cost into a single pass, and makes it O(chunk) per
    chunk.

    Construct via `RecordLookup.from_sources(sources)`.
    """

    __slots__ = ("s1_cols", "cand_cols")

    def __init__(self, s1_cols: pd.DataFrame, cand_cols: pd.DataFrame):
        self.s1_cols = s1_cols
        self.cand_cols = cand_cols

    @classmethod
    def from_sources(cls, sources: Dict[str, pd.DataFrame]) -> "RecordLookup":
        s1_cols = sources["source1"][
            ["business_name", "business_address", "country"]
        ].rename(columns=_S1_RENAME)
        cand_cols = pd.concat([
            sources["source2"][["business_name", "business_address", "country"]]
            .rename(columns=_CAND_RENAME),
            sources["source3"][["business_name", "business_address", "country"]]
            .rename(columns=_CAND_RENAME),
        ])
        # S2 and S3 entity ids are globally unique (S2-/S3- prefixed), so the
        # concat is already a valid unique join key. Assert it rather than
        # assume: a duplicate here would silently multiply candidate rows.
        if cand_cols.index.has_duplicates:
            dupes = cand_cols.index[cand_cols.index.duplicated()].unique()[:5]
            raise ValueError(
                "Candidate lookup has duplicate entity_ids across S2/S3 "
                f"(e.g. {list(dupes)}); attach_records would emit duplicate "
                "rows per pair."
            )
        return cls(s1_cols, cand_cols)

    def nbytes(self) -> int:
        return int(self.s1_cols.memory_usage(deep=True).sum()
                   + self.cand_cols.memory_usage(deep=True).sum())


def attach_records(
    candidate_pairs: pd.DataFrame,
    sources: Dict[str, pd.DataFrame],
    lookup: Optional[RecordLookup] = None,
) -> pd.DataFrame:
    """
    Attach S1 and candidate record fields to every pair.

    `lookup` is an optional pre-built `RecordLookup`. Pass one when calling
    this in a loop (as Stage 4 does) so the S1/S2/S3 join tables are not
    rebuilt per call; omit it and one is built on the fly, which is correct but
    O(|S1| + |S2| + |S3|) per call.
    """
    if candidate_pairs.empty:
        return pd.DataFrame(columns=_ATTACH_COL_ORDER)

    if lookup is None:
        lookup = RecordLookup.from_sources(sources)

    s1_missing = ~candidate_pairs["source1_entity_id"].isin(sources["source1"].index)
    if s1_missing.any():
        raise KeyError(
            f"S1 entity not found: {candidate_pairs.loc[s1_missing, 'source1_entity_id'].iloc[0]}"
        )

    is_s2 = candidate_pairs["candidate_source"] == "S2"
    s2_missing = ~candidate_pairs.loc[is_s2, "candidate_entity_id"].isin(sources["source2"].index)
    if s2_missing.any():
        raise KeyError(
            f"Candidate entity not found: "
            f"{candidate_pairs.loc[is_s2].loc[s2_missing, 'candidate_entity_id'].iloc[0]}"
        )
    s3_missing = ~candidate_pairs.loc[~is_s2, "candidate_entity_id"].isin(sources["source3"].index)
    if s3_missing.any():
        raise KeyError(
            f"Candidate entity not found: "
            f"{candidate_pairs.loc[~is_s2].loc[s3_missing, 'candidate_entity_id'].iloc[0]}"
        )

    merged = candidate_pairs.merge(
        lookup.s1_cols, left_on="source1_entity_id", right_index=True, how="left"
    )
    merged = merged.merge(
        lookup.cand_cols, left_on="candidate_entity_id", right_index=True, how="left"
    )
    return merged[_ATTACH_COL_ORDER].reset_index(drop=True)


def build_candidate_input(
    candidate_path: Path,
    source_dir: Path,
) -> pd.DataFrame:

    sources = load_source_records(source_dir)

    candidate_pairs = load_candidate_pairs(
        candidate_path
    )

    return attach_records(
        candidate_pairs,
        sources,
    )


# ============================================================
# STRING SIMILARITY HELPERS
# ============================================================

def normalized_edit_distance(
    a: str,
    b: str,
) -> float:
    """
    Levenshtein distance normalized to [0, 1].

    0.0 = identical
    1.0 = maximally different
    """

    a = a or ""
    b = b or ""

    if not a and not b:
        return 0.0

    if not a or not b:
        return 1.0

    previous = list(range(len(b) + 1))

    for i, char_a in enumerate(a, start=1):

        current = [i]

        for j, char_b in enumerate(b, start=1):

            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = (
                previous[j - 1]
                + int(char_a != char_b)
            )

            current.append(
                min(
                    insert_cost,
                    delete_cost,
                    replace_cost,
                )
            )

        previous = current

    distance = previous[-1]

    return distance / max(len(a), len(b))


def jaro_winkler(
    a: str,
    b: str,
) -> float:

    if not a or not b:
        return 0.0

    return float(
        JaroWinkler.similarity(a, b)
    )


def token_jaccard(
    a: str,
    b: str,
) -> float:

    tokens_a = set(a.split())
    tokens_b = set(b.split())

    if not tokens_a and not tokens_b:
        return 0.0

    union = tokens_a | tokens_b

    if not union:
        return 0.0

    return len(tokens_a & tokens_b) / len(union)


def char_ngram_jaccard(
    a: str,
    b: str,
    n: int,
) -> float:

    a = a or ""
    b = b or ""

    if not a or not b:
        return 0.0

    if len(a) < n or len(b) < n:
        return float(a == b)

    grams_a = {
        a[i:i + n]
        for i in range(len(a) - n + 1)
    }

    grams_b = {
        b[i:i + n]
        for i in range(len(b) - n + 1)
    }

    union = grams_a | grams_b

    if not union:
        return 0.0

    return len(grams_a & grams_b) / len(union)


# ============================================================
# SUFFIX HELPERS
# ============================================================

def get_suffix_type(raw_name: str) -> str:
    """
    Identify the terminal legal suffix.

    This is used only for the suffix-conflict feature.
    """

    if not raw_name:
        return ""

    name = str(raw_name).strip().lower()

    normalized_suffixes = sorted(
        [str(x).lower() for x in LEGAL_SUFFIXES],
        key=len,
        reverse=True,
    )

    for suffix in normalized_suffixes:

        if name.endswith(suffix):

            prefix = name[:-len(suffix)].rstrip()

            if prefix:
                return suffix

    return ""


# ============================================================
# NAME FEATURES
# ============================================================

def name_features(
    s1_raw_name: str,
    cand_raw_name: str,
) -> dict:

    s1 = normalize_name(s1_raw_name)
    cand = normalize_name(cand_raw_name)

    s1_norm = s1["clean"]
    cand_norm = cand["clean"]

    s1_no_suffix = s1["no_suffix"]
    cand_no_suffix = cand["no_suffix"]

    s1_sorted = s1["sorted_tokens"]
    cand_sorted = cand["sorted_tokens"]

    return {
        "name_exact":
            int(s1_norm == cand_norm),

        "name_exact_no_suffix":
            int(s1_no_suffix == cand_no_suffix),

        "name_sorted_exact":
            int(s1_sorted == cand_sorted),

        "name_edit_dist":
            normalized_edit_distance(
                s1_norm,
                cand_norm,
            ),

        "name_jaro_winkler":
            jaro_winkler(
                s1_norm,
                cand_norm,
            ),

        "name_jaro_winkler_ns":
            jaro_winkler(
                s1_no_suffix,
                cand_no_suffix,
            ),

        "name_token_jaccard":
            token_jaccard(
                s1_norm,
                cand_norm,
            ),

        "name_token_set_ratio":
            (
                token_set_ratio(
                    s1_norm,
                    cand_norm,
                ) / 100.0
                if s1_norm and cand_norm
                else 0.0
            ),

        "name_token_sort_ratio":
            (
                token_sort_ratio(
                    s1_norm,
                    cand_norm,
                ) / 100.0
                if s1_norm and cand_norm
                else 0.0
            ),

        "name_char3gram_jaccard":
            char_ngram_jaccard(
                s1_norm,
                cand_norm,
                3,
            ),

        "name_char4gram_jaccard":
            char_ngram_jaccard(
                s1_norm,
                cand_norm,
                4,
            ),

        "name_len_diff":
            abs(
                len(s1_norm) - len(cand_norm)
            ) / max(
                len(s1_norm),
                len(cand_norm),
                1,
            ),

        "_s1_no_suffix":
            s1_no_suffix,

        "_cand_no_suffix":
            cand_no_suffix,

        "_s1_suffix":
            get_suffix_type(s1_raw_name),

        "_cand_suffix":
            get_suffix_type(cand_raw_name),
    }


# ============================================================
# ADDRESS FEATURES
# ============================================================

def address_features(
    s1_raw_address: str,
    cand_raw_address: str,
) -> dict:

    s1_is_null = (
        not s1_raw_address
        or not str(s1_raw_address).strip()
    )

    cand_is_null = (
        not cand_raw_address
        or not str(cand_raw_address).strip()
    )

    s1 = normalize_address(s1_raw_address)
    cand = normalize_address(cand_raw_address)

    s1_norm = s1["clean"]
    cand_norm = cand["clean"]

    return {
        "addr_s1_null":
            int(s1_is_null),

        "addr_cand_null":
            int(cand_is_null),

        "addr_both_null":
            int(
                s1_is_null and cand_is_null
            ),

        "addr_exact":
            int(
                s1_norm == cand_norm
            )
            if s1_norm and cand_norm
            else 0,

        "addr_token_jaccard":
            token_jaccard(
                s1_norm,
                cand_norm,
            )
            if s1_norm and cand_norm
            else 0.0,

        "addr_char3gram_jaccard":
            char_ngram_jaccard(
                s1_norm,
                cand_norm,
                3,
            )
            if s1_norm and cand_norm
            else 0.0,

        "addr_house_num_agree":
            int(
                s1["house_number"]
                == cand["house_number"]
            )
            if (
                s1["house_number"]
                and cand["house_number"]
            )
            else 0,

        "addr_city_agree":
            int(
                s1["city"]
                == cand["city"]
            )
            if (
                s1["city"]
                and cand["city"]
            )
            else 0,

        "addr_pin_agree":
            int(
                s1["postal_code"]
                == cand["postal_code"]
            )
            if (
                s1["postal_code"]
                and cand["postal_code"]
            )
            else 0,
    }


# ============================================================
# TF-IDF MODEL
# ============================================================

class TFIDFSimilarityModel:
    """
    Character-level TF-IDF model.

    Architecture specification:
        analyzer = char_wb
        ngram_range = (2, 4)

    Separate vectorizers are used for names and addresses.
    """

    def __init__(
        self,
        max_features: int = 100_000,
    ):

        self.name_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            max_features=max_features,
        )

        self.address_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            max_features=max_features,
        )

        self.fitted = False

    def fit(
        self,
        names,
        addresses,
    ):

        names = [
            x if x else ""
            for x in names
        ]

        addresses = [
            x if x else ""
            for x in addresses
        ]

        self.name_vectorizer.fit(names)

        self.address_vectorizer.fit(addresses)

        self.fitted = True

        return self

    def transform_names(self, texts):

        if not self.fitted:
            raise RuntimeError(
                "TF-IDF model has not been fitted."
            )

        return self.name_vectorizer.transform(
            [
                x if x else ""
                for x in texts
            ]
        )

    def transform_addresses(self, texts):

        if not self.fitted:
            raise RuntimeError(
                "TF-IDF model has not been fitted."
            )

        return self.address_vectorizer.transform(
            [
                x if x else ""
                for x in texts
            ]
        )

    @staticmethod
    def row_cosine(
        matrix_a: sp.csr_matrix,
        matrix_b: sp.csr_matrix,
    ) -> np.ndarray:

        if matrix_a.shape[0] != matrix_b.shape[0]:
            raise ValueError(
                "Matrices must contain the same number of rows."
            )

        dots = (
            matrix_a.multiply(matrix_b)
            .sum(axis=1)
            .A1
        )

        norm_a = np.sqrt(
            matrix_a.multiply(matrix_a)
            .sum(axis=1)
            .A1
        )

        norm_b = np.sqrt(
            matrix_b.multiply(matrix_b)
            .sum(axis=1)
            .A1
        )

        denominator = norm_a * norm_b

        return np.divide(
            dots,
            denominator,
            out=np.zeros_like(
                dots,
                dtype=float,
            ),
            where=denominator > 0,
        )


# ============================================================
# COUNTRY + CONFLICT FEATURES
# ============================================================

def country_and_conflict_features(
    row: pd.Series,
    name_f: dict,
    addr_f: dict,
    n_blocking_rules_hit: int = 0,
) -> dict:

    country_match = int(
        normalize_country(
            row["s1_country"]
        )
        ==
        normalize_country(
            row["candidate_country"]
        )
    )

    high_name_low_addr = int(
        name_f["name_jaro_winkler"] > 0.9
        and addr_f["addr_token_jaccard"] < 0.2
        and not addr_f["addr_cand_null"]
    )

    suffix_conflict = int(
        jaro_winkler(
            name_f["_s1_no_suffix"],
            name_f["_cand_no_suffix"],
        ) > 0.9
        and name_f["_s1_suffix"]
        != name_f["_cand_suffix"]
        and bool(name_f["_s1_suffix"])
        and bool(name_f["_cand_suffix"])
    )

    return {
        "country_match":
            country_match,

        "high_name_low_addr":
            high_name_low_addr,

        "name_sim_x_addr_sim":
            (
                name_f["name_jaro_winkler"]
                *
                addr_f["addr_token_jaccard"]
            ),

        "n_blocking_rules_hit":
            int(n_blocking_rules_hit),

        "both_have_address":
            int(
                not addr_f["addr_s1_null"]
                and not addr_f["addr_cand_null"]
            ),

        "name_suffix_conflict":
            suffix_conflict,
    }


# ============================================================
# FEATURE MATRIX GENERATION
# ============================================================


def _process_chunk(chunk: pd.DataFrame):
    name_list, addr_list, conflict_list = [], [], []
    for _, row in chunk.iterrows():
        nf = name_features(row["s1_business_name"], row["candidate_business_name"])
        af = address_features(row["s1_business_address"], row["candidate_business_address"])
        bc = 0
        if "n_blocking_rules_hit" in row.index:
            try: bc = int(row["n_blocking_rules_hit"])
            except: pass
        cf = country_and_conflict_features(row, nf, af, bc)
        name_list.append(nf)
        addr_list.append(af)
        conflict_list.append(cf)
    return pd.DataFrame(name_list), pd.DataFrame(addr_list), pd.DataFrame(conflict_list)


# ─────────────────────────────────────────────────────────────
# STAGE 4 PARALLELISATION
# ─────────────────────────────────────────────────────────────
#
# MEASURED (bench/profile_stages.py, 20K S1 / 3.5M pairs, 8 cores),
# per 351K-pair feature chunk, before this change:
#
#   pool _process_chunk ...................... 31.3s   (7 workers, parallel)
#   parent-side TF-IDF, SERIAL in the parent .. 33.1s   (ONE core, 51% of chunk)
#     transform_names x2 ...................... 13.2s
#     transform_addresses x2 .................. 18.5s
#     row_cosine x2 ...........................  1.2s
#
# So roughly half of Stage 4 was single-threaded work happening in the parent
# while every worker sat idle waiting on the result queue. The four sparse
# TF-IDF matrices were also materialised for the whole chunk in the parent at
# once (4 x 351K x 4152, ~1GB of float64 indices+data for a single chunk).
#
# The fix: each worker now computes the TF-IDF cosines for its own slice too,
# and returns a small dense float32 block. Consequences:
#   * the 33.1s serial section is divided by the worker count;
#   * the four full-chunk sparse matrices are never all alive at once -- a
#     worker only ever holds the sparse matrices for its own slice;
#   * results cross the process boundary as a dense (n, 29) float32 array
#     (1.4MB for a 12.5K-row slice) instead of three intermediate DataFrames
#     that were then concatenated and run through a row-wise `apply`;
#   * one shared pool is reused across all chunks instead of one pool per chunk.

# Rows per parallel task. Deliberately a row budget, not a multiple of the core
# count: the previous `np.array_split(df, n_cores * 4)` produced 32 slices on an
# 8-core box and 252 slices on a 64-core box, which on a 2,000-raw-row chunk
# meant 252 near-empty tasks whose per-task overhead dominated the real work.
FEATURE_ROWS_PER_TASK = 20_000

_WORKER_TFIDF: Optional["TFIDFSimilarityModel"] = None


def _init_feature_worker(tfidf_model: "TFIDFSimilarityModel"):
    """
    Pool initializer: install the TF-IDF model in the worker exactly once.

    The model is small (vocabulary + idf vector, ~4K terms at production
    settings) so shipping one copy per worker is cheap, and doing it here
    rather than per task means it is pickled N times total instead of once per
    task. Without this, every one of the ~1,100 per-chunk task dispatches at
    production scale would re-pickle the model.
    """
    global _WORKER_TFIDF
    _WORKER_TFIDF = tfidf_model


def _process_chunk_with_tfidf(chunk: pd.DataFrame) -> np.ndarray:
    """
    Compute the complete FEATURE_COLS block for one slice of rows.

    Returns a dense float32 array of shape (len(chunk), len(FEATURE_COLS)) in
    exactly FEATURE_COLS order, with the same values the previous
    (serial-TF-IDF, three-DataFrame) implementation produced.
    """
    name_df, addr_df, conflict_df = _process_chunk(chunk)

    tf = _WORKER_TFIDF
    if tf is None:  # serial fallback path, or pool not initialized
        raise RuntimeError(
            "TF-IDF model not installed in worker. compute_features() must be "
            "called through the pool initializer, not by calling "
            "_process_chunk_with_tfidf directly."
        )

    s1_names = chunk["s1_business_name"].fillna("").astype(str).tolist()
    cand_names = chunk["candidate_business_name"].fillna("").astype(str).tolist()
    s1_addresses = chunk["s1_business_address"].fillna("").astype(str).tolist()
    cand_addresses = chunk["candidate_business_address"].fillna("").astype(str).tolist()

    name_df["name_tfidf_cosine"] = tf.row_cosine(
        tf.transform_names(s1_names),
        tf.transform_names(cand_names),
    )
    addr_df["addr_tfidf_cosine"] = tf.row_cosine(
        tf.transform_addresses(s1_addresses),
        tf.transform_addresses(cand_addresses),
    )

    result = pd.concat([name_df, addr_df, conflict_df], axis=1)

    internal_columns = [c for c in result.columns if c.startswith("_")]
    if internal_columns:
        result = result.drop(columns=internal_columns)
    result = result[FEATURE_COLS]

    arr = result.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    # `replace([inf,-inf], nan).fillna(0.0)` == nan_to_num(nan/posinf/neginf -> 0)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _feature_row_blocks(n_rows: int, rows_per_task: int) -> list:
    """
    Split a row range into (start, stop) blocks of at most `rows_per_task`.

    Uses positional slices and returns plain tuples so the blocks can be
    shipped to workers cheaply; the caller materialises them with `.iloc[]`.
    """
    if n_rows <= 0:
        return []
    step = max(1, rows_per_task)
    return [(i, min(i + step, n_rows)) for i in range(0, n_rows, step)]


def compute_features(
    pairs_df: pd.DataFrame,
    tfidf_model: TFIDFSimilarityModel,
    workers: Optional[int] = None,
    rows_per_task: int = FEATURE_ROWS_PER_TASK,
) -> pd.DataFrame:
    """
    Compute the full feature matrix for `pairs_df`.

    Parallel across all available CPUs (see `src.parallel.resolve_workers`;
    override with BER_WORKERS). Runs inline, without forking, when the input is
    too small for a pool to pay for itself.
    """
    if pairs_df.empty:
        return pd.DataFrame(columns=FEATURE_COLS)

    import numpy as np
    from src import parallel

    n_rows = len(pairs_df)
    n_workers = parallel.resolve_workers(workers)

    blocks = _feature_row_blocks(n_rows, rows_per_task)
    tasks = [pairs_df.iloc[a:b] for a, b in blocks]
    tasks = [t for t in tasks if not t.empty]

    if not tasks:
        return pd.DataFrame(columns=FEATURE_COLS)

    if n_workers <= 1 or len(tasks) == 1:
        # Serial: no fork, no pickle. Same numbers, no pool to pay for.
        _init_feature_worker(tfidf_model)
        blocks_out = [_process_chunk_with_tfidf(t) for t in tasks]
    else:
        pool = parallel.get_pool(
            initializer=_init_feature_worker,
            initargs=(tfidf_model,),
            workers=n_workers,
        )
        blocks_out = list(pool.imap(_process_chunk_with_tfidf, tasks, chunksize=1))

    combined = np.vstack(blocks_out).astype(np.float32, copy=False)
    del blocks_out
    return pd.DataFrame(combined, columns=FEATURE_COLS, dtype=np.float32)


# ============================================================
# TRAINING-CORPUS TF-IDF FITTING
# ============================================================

def fit_tfidf_from_sources(
    source_dir: Path,
) -> TFIDFSimilarityModel:
    """
    Fit TF-IDF using all records available in source_dir.

    For the real challenge, this should be the TRAIN corpus.
    """

    source_files = [
        source_dir / "train_source1.tsv",
        source_dir / "train_source2.tsv",
        source_dir / "train_source3.tsv",
    ]

    names = []
    addresses = []

    for path in source_files:

        if not path.exists():
            continue

        df = pd.read_csv(
            path,
            sep="\t",
            dtype=str,
            usecols=[
                "entity_id",
                "business_name",
                "business_address",
                "country",
            ],
        )

        names.extend(
            df["business_name"]
            .fillna("")
            .astype(str)
            .tolist()
        )

        addresses.extend(
            df["business_address"]
            .fillna("")
            .astype(str)
            .tolist()
        )

    if not names:
        raise FileNotFoundError(
            "No training source files were found for TF-IDF fitting."
        )

    model = TFIDFSimilarityModel()

    model.fit(
        names,
        addresses,
    )

    return model


def fit_tfidf_from_current_records(
    sources: Dict[str, pd.DataFrame],
) -> TFIDFSimilarityModel:
    """
    Dummy-data fallback.

    Used only when the project does not contain training TSVs.
    """

    names = []
    addresses = []

    for df in sources.values():

        names.extend(
            df["business_name"]
            .fillna("")
            .astype(str)
            .tolist()
        )

        addresses.extend(
            df["business_address"]
            .fillna("")
            .astype(str)
            .tolist()
        )

    model = TFIDFSimilarityModel()

    model.fit(
        names,
        addresses,
    )

    return model


# ============================================================
# COMPLETE PIPELINE
# ============================================================

def generate_feature_matrix(
    candidate_path: Path,
    source_dir: Path,
    tfidf_model: TFIDFSimilarityModel,
    split: str = "test",
    chunk_size: int = 2_000,
    workers: Optional[int] = None,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    sources = load_source_records(source_dir, split=split)
    lookup = RecordLookup.from_sources(sources)

    meta_chunks = []
    feat_chunks = []
    for exploded_chunk in iter_exploded_candidate_chunks(Path(candidate_path), chunk_size=chunk_size):
        if exploded_chunk.empty:
            continue
        chunk = attach_records(exploded_chunk, sources, lookup=lookup)
        del exploded_chunk
        feat = compute_features(chunk, tfidf_model, workers=workers)
        meta = chunk[["source1_entity_id", "candidate_entity_id", "candidate_source"]].reset_index(drop=True)
        meta_chunks.append(meta)
        feat_chunks.append(feat)
        del chunk

    if meta_chunks:
        metadata = pd.concat(meta_chunks, ignore_index=True)
        features = pd.concat(feat_chunks, ignore_index=True)
    else:
        metadata = pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id", "candidate_source"])
        features = pd.DataFrame(columns=FEATURE_COLS, dtype=np.float32)

    return metadata, features


# ============================================================
# VALIDATION
# ============================================================

def validate_features(
    features: pd.DataFrame,
) -> None:

    missing = [
        col
        for col in FEATURE_COLS
        if col not in features.columns
    ]

    if missing:
        raise ValueError(
            f"Missing feature columns: {missing}"
        )

    if features[FEATURE_COLS].isna().any().any():
        raise ValueError(
            "Feature matrix contains NaN values."
        )

    if np.isinf(
        features[FEATURE_COLS].to_numpy(
            dtype=float
        )
    ).any():
        raise ValueError(
            "Feature matrix contains infinite values."
        )

    non_numeric = [
        col
        for col in FEATURE_COLS
        if not pd.api.types.is_numeric_dtype(
            features[col]
        )
    ]

    if non_numeric:
        raise ValueError(
            f"Non-numeric feature columns: {non_numeric}"
        )


# ============================================================
# MAIN — DUMMY DATA TEST
# ============================================================

if __name__ == "__main__":

    project_root = (
        Path(__file__)
        .resolve()
        .parents[1]
    )

    candidate_path = (
        project_root
        / "dummy_interfaces"
        / "candidate_pairs.tsv"
    )

    source_dir = (
        project_root
        / "dummy_interfaces"
        / "test"
    )

    print()
    print("=" * 70)
    print("BUSINESS ENTITY RESOLUTION — FEATURE PIPELINE")
    print("=" * 70)

    # --------------------------------------------------------
    # Load sources
    # --------------------------------------------------------

    sources = load_source_records(
        source_dir
    )

    print()
    print("Loaded source records:")

    for source_name, df in sources.items():
        print(
            f"  {source_name}: {len(df)} records"
        )

    # --------------------------------------------------------
    # Candidate pairs
    # --------------------------------------------------------

    candidate_pairs = load_candidate_pairs(
        candidate_path
    )

    print()
    print(
        f"Candidate pairs: {len(candidate_pairs)}"
    )

    # --------------------------------------------------------
    # Attach records
    # --------------------------------------------------------

    attached = attach_records(
        candidate_pairs,
        sources,
    )

    print(
        f"Attached pairs: {len(attached)}"
    )

    # --------------------------------------------------------
    # TF-IDF
    # --------------------------------------------------------

    print()
    print("Fitting TF-IDF model...")

    tfidf_model = fit_tfidf_from_current_records(
        sources
    )

    print(
        "TF-IDF fitted successfully."
    )

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    print()
    print("Computing features...")

    features = compute_features(
        attached,
        tfidf_model,
    )

    validate_features(
        features
    )

    # --------------------------------------------------------
    # Final matrix
    # --------------------------------------------------------

    final_df = pd.concat(
        [
            attached[
                [
                    "source1_entity_id",
                    "candidate_entity_id",
                    "candidate_source",
                ]
            ].reset_index(drop=True),

            features.reset_index(drop=True),
        ],
        axis=1,
    )

    print()
    print("=" * 70)
    print("FEATURE PIPELINE COMPLETED")
    print("=" * 70)

    print()
    print(
        f"Rows: {len(final_df)}"
    )

    print(
        f"Feature columns: {len(FEATURE_COLS)}"
    )

    print()
    print("Feature columns:")

    for i, col in enumerate(
        FEATURE_COLS,
        start=1,
    ):
        print(
            f"  {i:02d}. {col}"
        )

    print()
    print("Feature matrix:")
    print(
        final_df.to_string(
            index=False
        )
    )

    print()
    print(
        "NaN count:",
        int(
            features[FEATURE_COLS]
            .isna()
            .sum()
            .sum()
        ),
    )

    print()
    print(
        "All features are numeric and NaN-safe."
    )