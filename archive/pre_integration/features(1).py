from pathlib import Path
from typing import Dict, Tuple

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


def attach_records(
    candidate_pairs: pd.DataFrame,
    sources: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Attach S1 and candidate record fields to every pair.
    """

    # Vectorized replacement for the original per-row Python loop, which
    # built an unbounded Python list of dicts via one .loc lookup per pair
    # -- verified 26x faster at 3k rows and byte-identical output in this
    # environment (see scratch/test_vectorize2.py). At production scale
    # (~86M pairs per docs/BER_Architecture.md Section 23) the original does
    # not finish in a reasonable time and its growing `rows` list is itself
    # an unbounded-memory pattern.
    if candidate_pairs.empty:
        return pd.DataFrame(columns=[
            "source1_entity_id", "candidate_entity_id", "candidate_source",
            "s1_business_name", "s1_business_address", "s1_country",
            "candidate_business_name", "candidate_business_address", "candidate_country",
        ])

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

    s1_cols = sources["source1"][["business_name", "business_address", "country"]].rename(
        columns={
            "business_name": "s1_business_name",
            "business_address": "s1_business_address",
            "country": "s1_country",
        }
    )
    merged = candidate_pairs.merge(
        s1_cols, left_on="source1_entity_id", right_index=True, how="left"
    )

    rename_cand = {
        "business_name": "candidate_business_name",
        "business_address": "candidate_business_address",
        "country": "candidate_country",
    }
    cand_cols = pd.concat([
        sources["source2"][["business_name", "business_address", "country"]].rename(columns=rename_cand),
        sources["source3"][["business_name", "business_address", "country"]].rename(columns=rename_cand),
    ])
    merged = merged.merge(
        cand_cols, left_on="candidate_entity_id", right_index=True, how="left"
    )

    col_order = [
        "source1_entity_id", "candidate_entity_id", "candidate_source",
        "s1_business_name", "s1_business_address", "s1_country",
        "candidate_business_name", "candidate_business_address", "candidate_country",
    ]
    return merged[col_order].reset_index(drop=True)


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

def compute_features(
    pairs_df: pd.DataFrame,
    tfidf_model: TFIDFSimilarityModel,
) -> pd.DataFrame:
    if pairs_df.empty:
        return pd.DataFrame(columns=FEATURE_COLS)

    import multiprocessing as mp
    import numpy as np
    
    n_cores = max(1, mp.cpu_count() - 1)
    # See Stage 4 audit note in run_pipeline.py: np.array_split on a DataFrame
    # directly is not version-stable across pandas releases. Splitting the row
    # index and using .iloc[] always returns proper DataFrame slices.
    split_positions = np.array_split(np.arange(len(pairs_df)), n_cores * 4)
    chunks = [pairs_df.iloc[idx] for idx in split_positions]

    from tqdm import tqdm
    valid_chunks = [c for c in chunks if not c.empty]
    
    with mp.Pool(n_cores) as pool:
        results = list(tqdm(pool.imap(_process_chunk, valid_chunks), total=len(valid_chunks), desc="Processing Feature Chunks"))
        
    name_df = pd.concat([r[0] for r in results], ignore_index=True) if results else pd.DataFrame()
    address_df = pd.concat([r[1] for r in results], ignore_index=True) if results else pd.DataFrame()
    conflict_df = pd.concat([r[2] for r in results], ignore_index=True) if results else pd.DataFrame()

    # --------------------------------------------------------
    # TF-IDF
    # --------------------------------------------------------

    s1_names = (
        pairs_df["s1_business_name"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    cand_names = (
        pairs_df["candidate_business_name"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    s1_addresses = (
        pairs_df["s1_business_address"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    cand_addresses = (
        pairs_df["candidate_business_address"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    s1_name_matrix = (
        tfidf_model.transform_names(
            s1_names
        )
    )

    cand_name_matrix = (
        tfidf_model.transform_names(
            cand_names
        )
    )

    s1_addr_matrix = (
        tfidf_model.transform_addresses(
            s1_addresses
        )
    )

    cand_addr_matrix = (
        tfidf_model.transform_addresses(
            cand_addresses
        )
    )

    name_tfidf = (
        tfidf_model.row_cosine(
            s1_name_matrix,
            cand_name_matrix,
        )
    )

    addr_tfidf = (
        tfidf_model.row_cosine(
            s1_addr_matrix,
            cand_addr_matrix,
        )
    )

    name_df["name_tfidf_cosine"] = name_tfidf

    address_df["addr_tfidf_cosine"] = addr_tfidf

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    result = pd.concat(
        [
            name_df,
            address_df,
            conflict_df,
        ],
        axis=1,
    )

    # Remove internal helper columns.
    internal_columns = [
        col
        for col in result.columns
        if col.startswith("_")
    ]

    if internal_columns:
        result = result.drop(
            columns=internal_columns
        )

    # Ensure exact feature ordering.
    result = result[FEATURE_COLS]

    # Convert every feature to numeric.
    result = result.apply(
        pd.to_numeric,
        errors="coerce",
    )

    # Explicit NaN protection.
    result = result.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    result = result.fillna(0.0)

    return result


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
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
]:

    sources = load_source_records(source_dir, split=split)

    candidate_pairs = load_candidate_pairs(
        candidate_path
    )

    attached = attach_records(
        candidate_pairs,
        sources,
    )

    features = compute_features(
        attached,
        tfidf_model,
    )

    # Metadata kept separately from ML features.
    metadata = attached[
        [
            "source1_entity_id",
            "candidate_entity_id",
            "candidate_source",
        ]
    ].reset_index(drop=True)

    features = features.reset_index(drop=True)

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