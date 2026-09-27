# Business Entity Resolution — Complete ML Pipeline Architecture

> **MEASURED** (from real files, Phase 0) vs **SPEC** (from challenge doc) vs **TBD** (needs runtime check) are tagged throughout.

---

## 1. Problem Understanding

Three independent, noisy data sources describe overlapping sets of real-world businesses. Source 1 is the deduplicated reference (`S1-` prefix). For each S1 entity, find every matching record in S2 and S3 — across name abbreviations, address noise, script transliterations, and an open-set country label that includes France at test time (zero France records in training).

**Scale** (MEASURED):

| File | Rows |
|------|------|
| train\_source1 | 2,206,821 |
| train\_source2 | 5,034,616 |
| train\_source3 | 5,285,603 |
| train\_ground\_truth | 2,206,821 (one row per S1 entity) |
| test\_source1 | 1,732,544 |
| test\_source2 | 4,887,273 |
| test\_source3 | 5,082,316 |

Raw pair space: `1.73M × (4.89M + 5.08M)` ≈ **1.73 × 10¹³** — blocking is not optional; it is the dominant design decision.

---

## 2. ML Problem Formulation

**Formulation: binary pairwise classification over blocked candidates.**

```
INPUT:  (S1_record, S2_or_S3_record, feature_vector)
OUTPUT: match_probability ∈ [0, 1]
```

**Why this formulation?**

| Alternative | Why rejected |
|---|---|
| Clustering (no reference) | S1 is a deduplicated reference — the problem is link prediction TO it, not clustering |
| Ranking (pairwise) | F0.5 is not a ranking metric; it rewards a precision-aware threshold, not a ranked list |
| Multi-label classification | Output space is ~10M IDs per S1 entity — intractable as a label set |
| Graph embedding | Can't generalize to France with zero training nodes |
| Binary classification over all pairs | Computationally infeasible without blocking (10¹³ pairs) |

Binary classification over blocked candidates solves all of these: it's tractable post-blocking, works with engineered features that generalize to unseen countries, produces calibrated probabilities for threshold optimization, and maps directly to the per-entity F0.5 metric.

**Score → list mapping rule (Phase 13):** For each S1 entity, collect all above-threshold pairs from both S2 and S3, deduplicate, sort (source then ID), join as comma-separated string. Entities with zero above-threshold pairs get empty string (singleton = correct when true).

---

## 3. Data Structure (Schema = FACT; Distributions = MEASURED)

**Schema (SPEC):** All files tab-separated, 4 columns: `entity_id`, `business_name`, `business_address`, `country`.

**MEASURED distributions:**

| Metric | S1 (ref) | S2 | S3 |
|--------|----------|----|----|
| Records (train) | 2,206,821 | 5,034,616 | 5,285,603 |
| Null business\_name | 0% | ~0.00004% | ~0.0002% |
| Null business\_address | 0% | **~3.4%** | **~3.3%** |
| Null country | 0% | 0% | 0% |
| Country dist (train) | US 60%, India 40% | US+India | US+India |
| Country dist (test S1) | **TBD** — includes France | — | — |
| Name char len (S1) | mean=24, p50=24, p95=37, max=105 | TBD | TBD |
| Addr char len (S1) | mean=52, p50=41, p95=103, max=256 | TBD | TBD |
| Name token count (S1) | mean=3.5, p50=4, p95=5, max=16 | TBD | TBD |
| S1 internal duplicates | **0** — confirmed | — | — |

**Ground truth (MEASURED):**

| Statistic | Value |
|-----------|-------|
| Total GT rows | 2,206,821 (= S1 row count: complete coverage) |
| S1 entities with ≥1 match | 2,083,574 (94.4%) |
| Singletons | **123,247 (5.58%)** |
| Mode match count | 3 (530,841 entities) |
| Max match count | 11 |
| Match distribution | 0→123K, 1→119K, 2→375K, 3→531K, 4→484K, 5→322K, 6→165K, 7→64K, 8→19K, 9→4K, 10→534, 11→37 |

**Critical open item resolved — cross-country match check (MEASURED):**
We ran the cross-country check across all 7,638,365 ground-truth pairs. The result is exactly **0 cross-country matches**.
Because true matches *never* cross country labels, we can safely use country as a **hard gate (AND condition)** in our blocking rules to drastically reduce the candidate space, rather than just a soft feature. This gate (`S1_country == Cand_country`) safely handles the unseen `France` label as well, since it relies on string equality, not memorized labels.

---

## 4. Input → Output Contract

```
INPUT:
  dataset/test/test_source1.tsv   (1,732,544 rows, 4 cols, sep="\t")
  dataset/test/test_source2.tsv   (4,887,273 rows, 4 cols, sep="\t")
  dataset/test/test_source3.tsv   (5,082,316 rows, 4 cols, sep="\t")

OUTPUT (both sep="\t"):
  output/matching_results.tsv     → scored on leaderboard
  output/candidate_pairs.tsv      → blocking audit (not scored)
```

**matching\_results.tsv rules (hard rejections if violated):**
- Exactly 1,732,544 rows (one per test S1 entity)
- No duplicate `source1_entity_id`
- `matched_entity_ids` = comma-separated S2-/S3- IDs, or empty string
- All IDs must exist in test S2/S3 files
- No duplicate IDs within one entity's list

**candidate\_pairs.tsv rules:**
- Same structural rules as above
- Must represent the **last** pre-model candidate stage (not raw blocking output)
- `matching_results.tsv` IDs ⊆ `candidate_pairs.tsv` IDs per entity (warns, doesn't hard-reject)

---

## 5. Functional Requirements

| # | Requirement | Source |
|---|-------------|--------|
| FR1 | Read all TSVs with `sep="\t"` | SPEC |
| FR2 | Output one row per test S1 entity | SPEC |
| FR3 | Empty string (not NaN/None) for singletons | SPEC |
| FR4 | Only S2-/S3- IDs from the test set | SPEC |
| FR5 | candidate\_pairs.tsv = last pre-model stage | SPEC |
| FR6 | matching\_results.tsv IDs ⊆ candidate IDs per entity | SPEC (warn) |
| FR7 | Run validate\_submission.py before every upload | SPEC |
| FR8 | Block across US, India, and France (open-set country) | SPEC |
| FR9 | Singleton → empty list, no fallback match | SPEC + METRIC |
| FR10 | Macro-average F0.5 per S1 entity; singleton = 1.0 or 0.0 exactly | SPEC |

---

## 6. Critical Constraints

| Constraint | Implication |
|------------|-------------|
| **Metric: macro F0.5 (β=0.5)** | Precision weighted 2× over recall. A false merge on a singleton scores 0.0 — catastrophic. Tune threshold to favor precision. |
| **Singleton rule: 1.0 or 0.0** | No partial credit. Never force a match below threshold just to avoid empty output. |
| **Open-set country (France unseen)** | Zero France training data. Any country-memorizing component (one-hot, per-country model, country-specific lookup) will fail France. Country = equality signal only. |
| **No external data/APIs** | No geocoding, no business registries, no entity lookup. Static disclosed suffix lists (general domain knowledge) are permitted. |
| **Model: MIT/Apache-2.0, ≤8B params** | Permits small open-weight multilingual embedding models as feature generators. Requires disclosure if used. |
| **Blocking recall ceiling** | Everything missed by blocking is permanently missed. Measure per-rule recall against ground truth before proceeding to model. |

**F0.5 formula (exact — unit-test your scoring code against the worked example):**
```
F0.5 = (1.25 × P × R) / (0.25 × P + R)

Worked example (from spec):
  Predicted: [S2-00047, S2-00193, S3-00812]
  Ground truth: [S2-00047, S3-00812]
  P = 2/3, R = 2/2 = 1.0
  F0.5 = (1.25 × 0.667 × 1.0) / (0.25 × 0.667 + 1.0) = 0.714  ✓
```

```python
def f05_score(pred: set, truth: set) -> float:
    """Per-entity F0.5. Exact singleton rule: 1.0 or 0.0."""
    if not truth:
        return 1.0 if not pred else 0.0   # singleton: exact rule
    if not pred:
        return 0.0
    tp = len(pred & truth)
    p = tp / len(pred)
    r = tp / len(truth)
    if p == 0 and r == 0:
        return 0.0
    return (1.25 * p * r) / (0.25 * p + r)

def macro_f05(pred_dict: dict, truth_dict: dict) -> float:
    """Macro-average over all S1 entities."""
    scores = [f05_score(set(pred_dict.get(k, [])), set(truth_dict.get(k, [])))
              for k in truth_dict]
    return sum(scores) / len(scores)
```

---

## 7. Normalization Architecture

Each transform shown as: **Original → Transformation → Why it helps → Potential information loss → Keep raw?**

### 7.1 Name Normalization Pipeline

```python
def normalize_name(raw: str) -> dict:
    """Returns dict with normalized variants; all retained."""
    if pd.isna(raw):
        return {"raw": "", "norm": "", "sorted_tokens": "", "no_suffix": ""}

    # Step 1: Unicode normalization
    s = unicodedata.normalize("NFKC", raw)
    # "NFKC" covers Devanagari normalization and Latin compatibility chars
    # Why: transliteration variants often differ only in Unicode form
    # Loss: none for ASCII; may collapse some Devanagari distinctions → keep raw

    # Step 2: Lowercase
    s = s.lower()
    # Why: case-insensitive matching; "Corp" ≡ "corp"
    # Loss: acronyms (IBM vs ibm) — still equal after lower, OK

    # Step 3: Punctuation/whitespace normalization
    s = re.sub(r"[^\w\s]", " ", s)    # strip punctuation → space
    s = re.sub(r"\s+", " ", s).strip()
    # Why: "B&B" → "b b", "B & B" → "b   b" → both become "b b"
    # Loss: "&" → "and" separately handled below

    # Step 4: & ↔ "and" normalization (apply BEFORE suffix stripping)
    s_and = re.sub(r"\band\b", "&", s)   # normalize to & form
    # Keep both versions as features

    # Step 5: Legal suffix normalization
    # DISCLOSED static list (general domain knowledge, not entity lookup):
    SUFFIX_MAP = {
        # US forms
        "corporation": "corp", "incorporated": "inc", "limited": "ltd",
        "company": "co", "private": "pvt", "llc": "llc", "llp": "llp",
        "lp": "lp", "plc": "plc",
        # India forms
        "private limited": "pvt ltd", "pvt ltd": "pvt ltd",
        "limited liability partnership": "llp",
        # French forms (DISCLOSED — general domain knowledge)
        "société anonyme": "sa", "société à responsabilité limitée": "sarl",
        "société par actions simplifiée": "sas",
        "société en nom collectif": "snc",
        "entreprise unipersonnelle à responsabilité limitée": "eurl",
        # Transliteration abbreviation pairs
        "प्राइवेट लिमिटेड": "pvt ltd",
        "प्राइवेट": "pvt", "लिमिटेड": "ltd",
    }
    # Apply longest-match suffix stripping
    no_suffix = apply_suffix_map(s, SUFFIX_MAP)

    # Step 6: DBA/trade name handling
    dba_pattern = re.compile(r"\b(dba|d\.b\.a\.|t/a|trading as)\b", re.I)
    parts = dba_pattern.split(s, maxsplit=1)
    legal_name = parts[0].strip() if len(parts) > 1 else s
    trade_name = parts[2].strip() if len(parts) > 2 else ""

    # Step 7: Token sorting (handles word-order transpositions)
    tokens = no_suffix.split()
    sorted_tokens = " ".join(sorted(tokens))
    # Why: "Prime Money" ≡ "Money Prime" after sort
    # Loss: word order carries semantic meaning in natural language → keep unsorted too

    return {
        "raw": raw,
        "norm": s,
        "norm_and": s_and,
        "no_suffix": no_suffix,
        "legal_name": legal_name,
        "trade_name": trade_name,
        "sorted_tokens": sorted_tokens,
    }
```

### 7.2 Address Normalization Pipeline

```python
STREET_ABBR = {
    "street": "st", "road": "rd", "avenue": "ave", "boulevard": "blvd",
    "drive": "dr", "lane": "ln", "court": "ct", "place": "pl",
    "highway": "hwy", "expressway": "expy", "nagar": "ngr", "marg": "marg",
    # India-specific
    "colony": "col", "sector": "sec", "phase": "ph",
}

LANDMARK_PATTERNS = [
    r"\bnear\s+\w+(\s+\w+){0,3}",   # "Near SBI ATM"
    r"\bopposite\s+\w+(\s+\w+){0,2}",
    r"\bbehind\s+\w+(\s+\w+){0,2}",
    r"\bnext\s+to\s+\w+(\s+\w+){0,2}",
    r"\bopp\.?\s+\w+(\s+\w+){0,2}",
]

def normalize_address(raw: str) -> dict:
    if pd.isna(raw):
        return {"raw": "", "norm": "", "no_landmark": "", "is_null": True,
                "has_number": False, "street_token": "", "city_token": ""}

    s = unicodedata.normalize("NFKC", raw).lower()
    s = re.sub(r"\s+", " ", s).strip()

    # Extract and strip landmark phrases (flag, don't delete entirely)
    landmark_found = []
    for pat in LANDMARK_PATTERNS:
        m = re.search(pat, s, re.I)
        if m:
            landmark_found.append(m.group())
            s = s[:m.start()] + " " + s[m.end():]
    s = re.sub(r"\s+", " ", s).strip()

    # Street type abbreviation
    for full, abbr in STREET_ABBR.items():
        s = re.sub(r"\b" + full + r"\b", abbr, s)

    # Extract house/building number (first token that is numeric/alphanumeric)
    tokens = s.split()
    house_number = ""
    if tokens and re.match(r"^\d+[\w\-/]*$", tokens[0]):
        house_number = tokens[0]

    # Pin/ZIP detection
    pin_match = re.search(r"\b\d{5,6}\b", s)
    has_pin = pin_match is not None
    pin_value = pin_match.group() if pin_match else ""

    # City token heuristic: last comma-separated component
    parts = raw.split(",")
    city_token = parts[-2].strip().lower() if len(parts) >= 2 else ""

    return {
        "raw": raw,
        "norm": s,
        "no_landmark": s,  # landmark-stripped
        "landmark_tokens": landmark_found,
        "is_null": False,
        "has_number": bool(house_number),
        "house_number": house_number,
        "has_pin": has_pin,
        "pin": pin_value,
        "city_token": city_token,
        "tokens": tokens,
    }
```

### 7.3 Country Normalization

```python
def normalize_country(raw: str) -> str:
    """Lowercase strip only — never encode as learned category."""
    return str(raw).strip().lower() if pd.notna(raw) else ""
# Used ONLY as an equality signal: s1_country_norm == s2_country_norm → binary feature
```

---

## 8. Blocking Architecture

> **Design principle:** Blocking sets the recall ceiling. Every true match missed here is permanently gone. Measure per-rule and union recall against `train_ground_truth` before touching any model.

### 8.1 Country-in-blocking decision

**Resolved (MEASURED):**
The cross-country check confirmed **0 cross-country matches** across 7.6M ground-truth pairs.
Therefore, `country` is applied as a **hard gate (AND condition)** across all blocking rules (`cand_country == s1_country`). Because this is a string equality check, it safely generalizes to unseen test labels (like France) without memorizing them.

### 8.2 Blocking Rules

```
S1 entity → applied to → all S2 + S3 records (filtered by same country) → union → candidate_pool
```

| Rule | Block Key (S1 side) | Block Key (S2/S3 side) | What it captures | Explosion risk |
|------|--------------------|-----------------------|-----------------|----------------|
| **A** | `sorted_name_tokens[:30]` (no suffix) | same | Token transpositions, suffix variants | Moderate — name collisions in common words |
| **B** | `house_number + street_token_1` | same | Same address, different name format | Low — house numbers are discriminative |
| **C** | `name_3gram_signature` (top-3 sorted char trigrams of normalized name) | same | Typos, transliteration drift, 1–2 char edits | High — 3-grams of short names collide often |
| **D** | `metaphone(first_name_token)` + `city_token` | same | Phonetic variants, transliteration | Moderate |
| **E** | `name_prefix_4` | same | Prefix matches | Low with country gate |

**Implementation:**

```python
# src/blocking.py

def build_block_index(df: pd.DataFrame, side: str) -> dict[str, list[str]]:
    """
    side: 'S1' or 'candidate'
    Returns: { (country_norm, rule, block_key): [entity_id, ...] }
    """
    index = defaultdict(list)
    for _, row in df.iterrows():
        nm = normalize_name(row["business_name"])
        ad = normalize_address(row["business_address"])
        ct = normalize_country(row["country"])
        eid = row["entity_id"]

        keys = []
        # Rule A: sorted token key
        if nm["sorted_tokens"]:
            keys.append(("A", nm["sorted_tokens"][:30]))
        # Rule B: address number + first street token
        if ad["has_number"] and ad["tokens"]:
            street = next((t for t in ad["tokens"][1:] if len(t) > 2), "")
            if street:
                keys.append(("B", f"{ad['house_number']}_{street}"))
        # Rule C: char trigram signature
        tri = sorted_trigram_key(nm["no_suffix"], n=3)
        if tri:
            keys.append(("C", tri))
        # Rule D: phonetic + city
        ph = phonetic_key(nm["norm"].split()[0] if nm["norm"].split() else "")
        city = ad["city_token"][:10]
        if ph and city:
            keys.append(("D", f"{ph}_{city}"))
        # Rule E: prefix
        prefix = nm["norm"][:4]
        if len(prefix) == 4:
            keys.append(("E", prefix))

        for rule, key in keys:
            index[(ct, rule, key)].append(eid)
    return index

def generate_candidates(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    max_candidates_per_entity: int = 200,
) -> pd.DataFrame:
    """
    Returns DataFrame: source1_entity_id, candidate_entity_id, blocking_rules
    This is the pre-model candidate set → written to candidate_pairs.tsv
    """
    cand_index_s2 = build_block_index(s2_df, "candidate")
    cand_index_s3 = build_block_index(s3_df, "candidate")

    records = []
    for _, s1_row in s1_df.iterrows():
        s1_id = s1_row["entity_id"]
        s1_ct = normalize_country(s1_row["country"])

        candidate_map: dict[str, set] = defaultdict(set)  # cand_id → {rules}

        # Generate same keys for S1, look up in S2 and S3 indices using country as primary key
        for rule, key in generate_keys_for_record(s1_row):
            for cand_id in cand_index_s2.get((s1_ct, rule, key), []):
                candidate_map[cand_id].add(rule)
            for cand_id in cand_index_s3.get((s1_ct, rule, key), []):
                candidate_map[cand_id].add(rule)

        # Explosion control: if too many candidates, keep those with most rule hits
        if len(candidate_map) > max_candidates_per_entity:
            candidate_map = dict(
                sorted(candidate_map.items(), key=lambda x: -len(x[1]))
                [:max_candidates_per_entity]
            )

        for cand_id, rules in candidate_map.items():
            records.append({
                "source1_entity_id": s1_id,
                "candidate_entity_id": cand_id,
                "blocking_rules": ",".join(sorted(rules)),
            })

    return pd.DataFrame(records)
```

**Per-rule recall measurement (required gate before proceeding to model):**

```python
def measure_blocking_recall(candidates_df, ground_truth_df) -> dict:
    """Gate: must reach >95% recall before training any model."""
    gt = ground_truth_df.set_index("source1_entity_id")["matched_entity_ids"]

    tp = fp = fn = 0
    per_rule_recall = defaultdict(lambda: {"hit": 0, "miss": 0})

    for s1_id, matches_str in gt.items():
        if pd.isna(matches_str) or not matches_str:
            continue
        truth_set = set(matches_str.split(","))
        cand_rows = candidates_df[candidates_df["source1_entity_id"] == s1_id]
        cand_set = set(cand_rows["candidate_entity_id"])
        hits = truth_set & cand_set
        misses = truth_set - cand_set
        tp += len(hits)
        fn += len(misses)
        # Per-rule attribution
        for _, row in cand_rows.iterrows():
            if row["candidate_entity_id"] in truth_set:
                for r in row["blocking_rules"].split(","):
                    per_rule_recall[r]["hit"] += 1

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    reduction_ratio = 1 - len(candidates_df) / (len(s1_df) * (len(s2_df) + len(s3_df)))
    return {"recall": recall, "reduction_ratio": reduction_ratio, "per_rule": per_rule_recall}
```

> **Gate:** Do not proceed to model training until blocking recall > 95% on the training validation split.

---

## 9. Candidate-Pair Architecture & `candidate_pairs.tsv` Contract

```
[Blocking union] → [Optional coarse similarity filter] → candidate_pairs.tsv → [ML model] → matching_results.tsv
                                                                ↑
                                               THIS is what candidate_pairs.tsv records
```

**Two-stage design:**

| Stage | Purpose | Output |
|-------|---------|--------|
| **Stage 1: Blocking** | Rule-based key matching (Section 8) | Raw union: up to 200 candidates/entity |
| **Stage 2: Coarse filter** (optional) | Lightweight similarity threshold (e.g., Jaro-Winkler on name > 0.7 OR token overlap > 0.3) to cut candidates before expensive feature computation | Pruned candidates: target ~50/entity |
| **→ candidate\_pairs.tsv** | The Stage 2 output — the exact pool the ML model scores | Written here |
| **Stage 3: ML model** | Full feature computation + scoring | matching\_results.tsv |

**Internal working schema (richer than the TSV):**

```python
# Internal candidate pair record
CandidatePair = {
    "source1_entity_id": str,      # S1-XXXXXXXXX
    "candidate_source": str,       # "S2" or "S3"
    "candidate_id": str,           # S2-XXXXXXXXX or S3-XXXXXXXXX
    "blocking_rules": list[str],   # ["A", "C"] — provenance
    "blocking_evidence": str,      # the matching key value
    "coarse_name_sim": float,      # fast Jaro-Winkler score
    "passed_coarse": bool,         # True → in candidate_pairs.tsv
}
```

**Writing `candidate_pairs.tsv`:**

```python
def write_candidate_pairs(candidates_df: pd.DataFrame, s1_ids: list, output_path: str):
    """
    candidates_df: post-coarse-filter (Stage 2 output)
    s1_ids: ALL S1 entity IDs — must include singletons
    """
    grouped = candidates_df.groupby("source1_entity_id")["candidate_id"].apply(list)

    rows = []
    for s1_id in s1_ids:
        if s1_id in grouped.index:
            cands = list(dict.fromkeys(grouped[s1_id]))  # dedupe, preserve order
            rows.append({"source1_entity_id": s1_id, "candidate_entity_ids": ",".join(cands)})
        else:
            rows.append({"source1_entity_id": s1_id, "candidate_entity_ids": ""})

    pd.DataFrame(rows).to_csv(output_path, sep="\t", index=False)
```

---

## 10. Feature Engineering Architecture

```
(S1_record, candidate_record) → feature_vector → model input
```

### 10.1 Name Similarity Features (12 features)

```python
def name_features(s1_norm: dict, cand_norm: dict) -> dict:
    s1n, cn = s1_norm["norm"], cand_norm["norm"]
    s1s, cs = s1_norm["sorted_tokens"], cand_norm["sorted_tokens"]
    s1ns, cns = s1_norm["no_suffix"], cand_norm["no_suffix"]

    return {
        # Exact matches
        "name_exact":              int(s1n == cn),
        "name_exact_no_suffix":    int(s1ns == cns),
        "name_sorted_exact":       int(s1s == cs),

        # Edit distance
        "name_edit_dist":          normalized_edit_distance(s1n, cn),    # [0,1]
        "name_jaro_winkler":       jaro_winkler(s1n, cn),                # [0,1]
        "name_jaro_winkler_ns":    jaro_winkler(s1ns, cns),              # no-suffix variant

        # Token-level
        "name_token_jaccard":      token_jaccard(s1n, cn),
        "name_token_set_ratio":    token_set_ratio(s1n, cn),             # fuzzywuzzy-style
        "name_token_sort_ratio":   token_sort_ratio(s1n, cn),

        # Character n-gram
        "name_char3gram_jaccard":  char_ngram_jaccard(s1n, cn, n=3),
        "name_char4gram_jaccard":  char_ngram_jaccard(s1n, cn, n=4),

        # Length / structure
        "name_len_diff":           abs(len(s1n) - len(cn)) / max(len(s1n), len(cn), 1),

        # TF-IDF cosine (sparse text feature — see Section 10.4)
        "name_tfidf_cosine":       0.0,  # filled by TF-IDF vectorizer at batch time
    }
```

### 10.2 Address Similarity Features (10 features)

```python
def address_features(s1_addr: dict, cand_addr: dict) -> dict:
    s1n, cn = s1_addr["norm"], cand_addr["norm"]

    return {
        # Null indicators (explicit missingness features)
        "addr_s1_null":            int(s1_addr["is_null"]),
        "addr_cand_null":          int(cand_addr["is_null"]),
        "addr_both_null":          int(s1_addr["is_null"] and cand_addr["is_null"]),

        # Exact and token overlap
        "addr_exact":              int(s1n == cn) if s1n and cn else 0,
        "addr_token_jaccard":      token_jaccard(s1n, cn) if s1n and cn else 0.0,
        "addr_char3gram_jaccard":  char_ngram_jaccard(s1n, cn, n=3) if s1n and cn else 0.0,

        # Component-level
        "addr_house_num_agree":    int(s1_addr["house_number"] == cand_addr["house_number"])
                                   if (s1_addr["house_number"] and cand_addr["house_number"]) else 0,
        "addr_city_agree":         int(s1_addr["city_token"] == cand_addr["city_token"])
                                   if (s1_addr["city_token"] and cand_addr["city_token"]) else 0,
        "addr_pin_agree":          int(s1_addr["pin"] == cand_addr["pin"])
                                   if (s1_addr["has_pin"] and cand_addr["has_pin"]) else 0,

        # TF-IDF cosine (sparse text feature)
        "addr_tfidf_cosine":       0.0,  # filled at batch time
    }
```

### 10.3 Country and Conflict Features (6 features)

```python
def country_and_conflict_features(s1_row, cand_row, name_f, addr_f) -> dict:
    return {
        # Country: binary equality ONLY — no learned per-country identity
        "country_match":           int(normalize_country(s1_row["country"])
                                      == normalize_country(cand_row["country"])),

        # Cross-field conflict detection
        "high_name_low_addr":      int(name_f["name_jaro_winkler"] > 0.9
                                      and addr_f["addr_token_jaccard"] < 0.2
                                      and not addr_f["addr_cand_null"]),
        "name_sim_x_addr_sim":     name_f["name_jaro_winkler"] * addr_f["addr_token_jaccard"],

        # Blocking provenance features (how many rules fired)
        "n_blocking_rules_hit":    0,  # filled from candidate provenance
        "both_have_address":       int(not s1_row.get("addr_is_null", False)
                                       and not addr_f["addr_cand_null"]),

        # Suffix conflict (same name, different suffix type)
        "name_suffix_conflict":    int(
            jaro_winkler(name_f.get("s1_legal", ""), name_f.get("c_legal", "")) > 0.9
            and name_f.get("s1_suffix_type", "") != name_f.get("c_suffix_type", "")
        ),
    }
```

### 10.4 TF-IDF Cosine Similarity (sparse text features — "Include TF-IDF" requirement)

```python
# src/features.py — TF-IDF feature generation

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse as sp

class TFIDFSimilarityModel:
    """
    Fit on all business names + addresses in train set.
    At inference: transform both records, compute cosine similarity.
    This is a SPARSE representation — no dense embedding, no external model.
    """
    def __init__(self, analyzer="char_wb", ngram_range=(2, 4), max_features=100_000):
        self.name_vec = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range,
                                        max_features=max_features)
        self.addr_vec = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range,
                                        max_features=max_features)

    def fit(self, all_names: list[str], all_addresses: list[str]):
        self.name_vec.fit(all_names)
        self.addr_vec.fit(all_addresses)

    def transform_names(self, texts: list[str]) -> sp.csr_matrix:
        return self.name_vec.transform(texts)

    def transform_addrs(self, texts: list[str]) -> sp.csr_matrix:
        return self.addr_vec.transform(texts)

    def batch_cosine(self, mat_s1: sp.csr_matrix, mat_cand: sp.csr_matrix) -> np.ndarray:
        """Compute row-wise cosine similarity for matched rows."""
        # Efficient: element-wise dot product of unit-normed rows
        norms_s1   = sp.linalg.norm(mat_s1,   axis=1)
        norms_cand = sp.linalg.norm(mat_cand, axis=1)
        dots = mat_s1.multiply(mat_cand).sum(axis=1).A1
        denom = norms_s1 * norms_cand
        return np.where(denom > 0, dots / denom, 0.0)
```

**Why TF-IDF as sparse feature:**
- Captures sub-word n-gram patterns without an external model
- Character-level (2–4 grams) handles typos, transliterations, abbreviations
- Fits on training corpus → no external data
- One cosine score per pair (name TF-IDF similarity + address TF-IDF similarity = 2 features)
- Complements edit-distance features with global corpus frequency weighting

### 10.5 Optional Dense Embedding Feature

If TF-IDF + string features underperform on validation, add a single cosine similarity feature from a small multilingual model:

```
Model options (MIT/Apache-2.0, ≤8B params, multilingual):
  - sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2  (117M params, Apache-2.0)
  - intfloat/multilingual-e5-small                               (118M params, MIT)
  - BAAI/bge-m3 (up to 1.02B params, MIT)

Disclosure (required in Documentation_template.md):
  "We use [model] solely as a general-purpose multilingual text representation.
   It produces a fixed embedding for any string regardless of business identity.
   This is NOT entity lookup — the model has no knowledge of specific businesses.
   License: [MIT/Apache-2.0]. Parameters: [N]M. Used only as a feature generator
   feeding into a gradient-boosted tree classifier."

Usage pattern:
  - Batch-encode all candidate names at inference (never call per-pair)
  - Cache embeddings by entity_id
  - One cosine similarity feature: embed(s1_name) · embed(cand_name)
  - Same for address if available
```

**Full feature vector: ~32 features** (12 name + 10 address + 6 country/conflict + 2 TF-IDF cosine + [2 optional embedding cosine] + [blocking rule provenance])

---

## 11. Matching Model Architecture

### 11.1 Candidate Model Comparison

| Model | Precision/Recall for ER | Interpretability | Data Requirement | Compute | France Generalization | License/Size |
|-------|------------------------|-----------------|-----------------|---------|----------------------|--------------|
| Rule-based (exact/fuzzy thresholds) | High precision, low recall | Full | None | Minimal | Perfect — no training | N/A |
| Logistic Regression | Moderate P, moderate R | Good | ~100K pairs | Minimal | Good — no country memorization | N/A |
| **Gradient Boosted Trees (LightGBM/XGBoost)** | **High P, tunable R** | **Good (SHAP)** | **~500K pairs** | **Low** | **Good if features are country-agnostic** | **N/A** |
| Random Forest | Similar to GBT | Good | ~500K pairs | Moderate | Good | N/A |
| End-to-end Neural (BERT fine-tune) | High recall, variable P | Low | Millions | High | Risk: memorizes script patterns | ≤8B, MIT/Apache |
| Embedding + GBT hybrid | High on all dims | Good | ~500K pairs | Moderate | Best for cross-script | ≤8B, MIT/Apache |
| Per-country models | High on seen countries | Good | Needs France data | Low | **FAILS — zero France training data → DISQUALIFIED** | N/A |

**Selected: LightGBM over engineered features (including TF-IDF cosine), with optional embedding cosine as an additional feature.**

**Justification:**
1. Dataset is large (2M S1 × 10M cands → after blocking: ~200M pairs → after training split: still millions of labeled pairs). GBT handles this scale efficiently.
2. F0.5 rewards precision — GBT's threshold calibration is transparent and controllable.
3. Features are country-agnostic by design (string similarity, not country identity) → France generalizes.
4. Interpretability via SHAP for debugging false merges.
5. No external model required as baseline; embedding feature is additive if needed.
6. Fully MIT/Apache-2.0 compatible.

```python
# src/train.py

import lightgbm as lgb

FEATURE_COLS = [
    # Name features
    "name_exact", "name_exact_no_suffix", "name_sorted_exact",
    "name_edit_dist", "name_jaro_winkler", "name_jaro_winkler_ns",
    "name_token_jaccard", "name_token_set_ratio", "name_token_sort_ratio",
    "name_char3gram_jaccard", "name_char4gram_jaccard", "name_len_diff",
    "name_tfidf_cosine",
    # Address features
    "addr_s1_null", "addr_cand_null", "addr_both_null",
    "addr_exact", "addr_token_jaccard", "addr_char3gram_jaccard",
    "addr_house_num_agree", "addr_city_agree", "addr_pin_agree",
    "addr_tfidf_cosine",
    # Country + conflict
    "country_match", "high_name_low_addr", "name_sim_x_addr_sim",
    "n_blocking_rules_hit", "both_have_address", "name_suffix_conflict",
    # Optional: embedding cosines
    # "name_embed_cosine", "addr_embed_cosine",
]

def train_model(train_pairs_df: pd.DataFrame) -> lgb.Booster:
    X = train_pairs_df[FEATURE_COLS]
    y = train_pairs_df["label"]   # 1 = match, 0 = non-match

    train_data = lgb.Dataset(X, label=y)
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "n_estimators": 500,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": -1,
        "min_child_samples": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "class_weight": "balanced",  # positives are rare among blocked candidates
        "random_state": 42,
        "n_jobs": -1,
    }
    model = lgb.train(params, train_data, num_boost_round=500,
                      valid_sets=[train_data], callbacks=[lgb.early_stopping(50)])
    return model
```

---

## 12. Threshold Optimization

```
candidate_pairs → features → model.predict_proba() → threshold_sweep → macro_F0.5
```

```python
# src/threshold.py

def sweep_threshold(
    val_pairs_df: pd.DataFrame,
    model: lgb.Booster,
    truth_dict: dict[str, set],
    thresholds: np.ndarray = np.arange(0.1, 0.99, 0.01),
) -> tuple[float, pd.DataFrame]:
    """
    Returns (best_threshold, sweep_results_df)
    Computes macro F0.5 exactly per spec (singleton rule enforced).
    """
    scores = model.predict(val_pairs_df[FEATURE_COLS])
    val_pairs_df = val_pairs_df.copy()
    val_pairs_df["score"] = scores

    results = []
    for t in thresholds:
        pred_dict = {}
        for s1_id, grp in val_pairs_df.groupby("source1_entity_id"):
            above = grp[grp["score"] >= t]["candidate_id"].tolist()
            pred_dict[s1_id] = set(above)

        # Include singletons (S1 entities with no candidates)
        for s1_id in truth_dict:
            if s1_id not in pred_dict:
                pred_dict[s1_id] = set()

        f05 = macro_f05(pred_dict, truth_dict)
        p_avg = np.mean([
            len(pred_dict[k] & truth_dict[k]) / max(len(pred_dict[k]), 1)
            for k in truth_dict
        ])
        r_avg = np.mean([
            len(pred_dict[k] & truth_dict[k]) / max(len(truth_dict[k]), 1)
            for k in truth_dict if truth_dict[k]
        ])
        results.append({"threshold": t, "f05": f05, "precision": p_avg, "recall": r_avg})

    results_df = pd.DataFrame(results)
    best_t = results_df.loc[results_df["f05"].idxmax(), "threshold"]
    return best_t, results_df
```

**Decision guidance:**
- Given 2× precision weight: if two thresholds produce similar F0.5, prefer the higher one (more precision)
- Check per-source (S2 vs S3) threshold separately — they may have different noise profiles
- Inspect false merges at chosen threshold (SHAP explanation helps identify which features drove wrong predictions)
- Do NOT tune against public leaderboard feedback (overfits public split, hurts private split)

---

## 13. Singleton Handling

```python
# Implicit: entities with no candidates stay empty after threshold.
# No fallback matching logic. Ever.

def apply_threshold_and_aggregate(
    scored_pairs: pd.DataFrame,
    all_s1_ids: list[str],
    threshold: float,
) -> dict[str, list[str]]:
    """Returns {s1_id: [matched_ids]} for ALL S1 entities."""
    result = {s1_id: [] for s1_id in all_s1_ids}

    above = scored_pairs[scored_pairs["score"] >= threshold]
    for s1_id, grp in above.groupby("source1_entity_id"):
        result[s1_id] = sorted(grp["candidate_id"].tolist())  # deterministic sort

    return result
    # Entities with zero above-threshold pairs → result[s1_id] = [] → correct singleton
```

**Why no fallback:** The metric makes this a strict calculation. If a true singleton gets a forced match, the entity scores 0.0 instead of 1.0 — a 1.0-point loss. With 5.58% singletons (123K entities), getting these right contributes meaningfully to macro average. A "nearest neighbor always" strategy would lose ~123K × 1.0 ≈ inflating the false-merge rate significantly.

---

## 14. Multi-Match Aggregation

```python
def aggregate_matches(
    match_dict: dict[str, list[str]],
    all_s1_ids: list[str],
) -> pd.DataFrame:
    """
    Produces matching_results.tsv content.
    Deduplicates IDs, sorts deterministically (source prefix then numeric ID).
    """
    rows = []
    for s1_id in all_s1_ids:
        matches = match_dict.get(s1_id, [])
        # Deduplicate (should be clean from threshold step, but be safe)
        seen = dict.fromkeys(matches)   # preserves order, deduplicates
        # Sort: S2- before S3-, then numeric portion ascending
        def sort_key(eid):
            prefix = eid[:2]   # "S2" or "S3"
            num = int(eid[3:]) if eid[3:].isdigit() else 0
            return (prefix, num)
        sorted_matches = sorted(seen.keys(), key=sort_key)
        rows.append({
            "source1_entity_id": s1_id,
            "matched_entity_ids": ",".join(sorted_matches),
        })
    return pd.DataFrame(rows)
```

**Note:** S2-/S3- prefix guarantees uniqueness across sources — no collision handling needed. But keep full prefixes in all internal tables regardless (required in output format, and necessary for attribution).

---

## 15. Region-Aware / Open-Set Country Strategy

| Strategy | Description | France behavior | Recommendation |
|----------|-------------|-----------------|---------------|
| **(A) Single global model, country = hard blocking gate + equality feature** | One model; pairs are only generated if `s1_country == cand_country`. Model uses `country_match` binary feature (always 1.0 for blocked pairs, but kept for consistency). | Generalizes: Since blocking is purely by string equality (`France` == `France`), candidates will be generated correctly without the model needing to know what `France` is. | **RECOMMENDED** |

**Strategy A implementation note:** The cross-country check confirmed that true matches never cross country labels. Therefore, we use country as a hard gate during blocking. For France-France pairs, this will simply retrieve candidates with the exact same string label. The model never needs to know which country it is — only that the strings match. This fully generalizes to any unseen country label.

---

## 16. Training / Validation Architecture

### 16.1 Positive and Negative Pair Construction

```python
# src/train.py

def build_training_pairs(
    s1_df, s2_df, s3_df, ground_truth_df, candidates_df
) -> pd.DataFrame:
    """
    CRITICAL: negatives are drawn from the SAME blocking pipeline used at inference.
    NOT from a random cross-product.
    """
    gt_map = {}
    for _, row in ground_truth_df.iterrows():
        s1_id = row["source1_entity_id"]
        mstr = str(row["matched_entity_ids"]) if pd.notna(row["matched_entity_ids"]) else ""
        gt_map[s1_id] = set(mstr.split(",")) if mstr else set()

    pairs = []
    for _, cand_row in candidates_df.iterrows():
        s1_id = cand_row["source1_entity_id"]
        cand_id = cand_row["candidate_id"]
        label = int(cand_id in gt_map.get(s1_id, set()))
        pairs.append({**cand_row.to_dict(), "label": label})

    df = pd.DataFrame(pairs)

    # Hard negatives: high name similarity but wrong entity
    # These are already included implicitly (blocking brings in false positives)
    # Explicitly upsample hard negatives for training stability:
    easy_neg = df[(df["label"] == 0) & (df["name_jaro_winkler"] < 0.7)]
    hard_neg = df[(df["label"] == 0) & (df["name_jaro_winkler"] >= 0.7)]
    positives = df[df["label"] == 1]

    # Balance: all positives + all hard negatives + sample of easy negatives
    n_easy_sample = len(hard_neg) * 2
    sampled_easy = easy_neg.sample(min(n_easy_sample, len(easy_neg)), random_state=42)
    return pd.concat([positives, hard_neg, sampled_easy]).sample(frac=1, random_state=42)
```

### 16.2 Leakage-Safe Train/Validation Split

```python
def split_by_s1_entity(
    s1_df: pd.DataFrame,
    val_fraction: float = 0.1,
    random_state: int = 42,
) -> tuple[list[str], list[str]]:
    """
    Split on SOURCE 1 ENTITY ID — not on pair rows.
    All pairs for a given S1 entity go entirely into train OR val.
    Stratify by country to ensure both US and India appear in val.
    """
    s1_ids = s1_df["entity_id"].tolist()
    country_map = s1_df.set_index("entity_id")["country"].to_dict()

    from sklearn.model_selection import GroupShuffleSplit
    # Use country as stratification group
    gss = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=random_state)
    countries = [country_map[i] for i in s1_ids]
    train_idx, val_idx = next(gss.split(s1_ids, groups=countries))
    return [s1_ids[i] for i in train_idx], [s1_ids[i] for i in val_idx]
```

**France generalization note:** Zero France entities in training → zero in validation split → France performance is untestable before submission. Public leaderboard is the only pre-final signal. Design features to be country-agnostic and monitor public leaderboard score when France-containing test entities are scored.

---

## 17. Implementability Requirement — Component Specifications

| Component | Purpose | Input | Processing | Output | Dependencies | Validation method |
|-----------|---------|-------|-----------|--------|-------------|------------------|
| `data_loading.py` | Load all TSVs safely | File paths | `pd.read_csv(sep="\t")` | DataFrames | pandas | Check column count == 4, entity_id prefix check |
| `profiling.py` | Measure distributions | All DataFrames | Null rates, country dist, name/addr length stats, S1 dup check | Profile report JSON | pandas, numpy | Manual inspection + assert no S1 dups |
| `normalization.py` | Noise-driven cleaning | Raw records | `normalize_name()`, `normalize_address()`, `normalize_country()` | Enriched DataFrames with norm columns | unicodedata, re | Unit test: "Corporation" → "corp", "Rd" → "rd", landmark stripped |
| `blocking.py` | Candidate set generation (Stage 1) | Normalized S1, S2, S3 | Build inverted indices on 5 rule keys; union lookups | Raw candidates DataFrame | — | Measure blocking recall ≥ 95% against GT |
| `candidate_generation.py` | Coarse filter (Stage 2) → writes `candidate_pairs.tsv` | Raw candidates | Jaro-Winkler name threshold OR token overlap | Pruned candidates DataFrame + TSV | jellyfish/rapidfuzz | Check TSV schema, check `matching_results.tsv` IDs ⊆ candidates per entity |
| `features.py` | Pairwise feature computation | Candidate pairs + normalized records | All similarity functions + TF-IDF cosine | Feature matrix (N × 32) | scikit-learn, rapidfuzz, jellyfish | Check no NaN in feature matrix |
| `train.py` | Model training | Feature matrix + labels | LightGBM fit; hard-negative upsampling; entity-grouped train/val split | `model.lgb` file | lightgbm | Macro F0.5 on held-out val; SHAP inspection of top features |
| `threshold.py` | Threshold sweep and selection | Validation pairs + scores | Sweep 0.01–0.99, compute macro F0.5 per threshold; check per-source | Best threshold scalar | — | Verify F0.5 code against worked example (P=2/3,R=1→0.714) |
| `predict.py` | Inference on test set | Test candidate pairs + model | `model.predict()` + batch TF-IDF cosine | Scored pairs DataFrame | lightgbm | Score distribution sanity check |
| `aggregation.py` | Build output lists | Scored pairs + threshold + all S1 IDs | Apply threshold; dedupe; sort deterministically; empty = singleton | `{s1_id: [cand_ids]}` dict | — | Assert every S1 ID is in result dict; assert no S1- IDs in values |
| `validation.py` | Pre-submission check | Both TSV output files | Run `validate_submission.py` programmatically | PASS/FAIL + issue list | stdlib only | Must reach exit code 0 before any leaderboard upload |

---

## 18. Exact Output Contract (Implementation Checklist)

```python
# Checklist for both output files:

def write_matching_results(match_dict, all_s1_ids, path):
    rows = []
    seen_s1 = set()
    for s1_id in all_s1_ids:
        assert s1_id not in seen_s1, f"Duplicate S1 ID: {s1_id}"
        seen_s1.add(s1_id)
        matches = match_dict.get(s1_id, [])
        # Validate: no S1- IDs, no duplicates
        assert all(m.startswith(("S2-", "S3-")) for m in matches)
        assert len(matches) == len(set(matches)), f"Dupes in {s1_id}"
        rows.append({
            "source1_entity_id": s1_id,
            "matched_entity_ids": ",".join(matches),  # empty string if no matches
        })
    assert len(rows) == len(all_s1_ids)
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False, encoding="utf-8")
```

**Write rule summary:**

| Rule | Violation severity | Internal check |
|------|-------------------|---------------|
| One row per S1 test entity | Hard rejection | `assert len == len(all_s1_ids)` |
| No duplicate `source1_entity_id` | Hard rejection | `seen_s1` set check |
| Empty string for no match | Hard rejection | `"".join([])` naturally produces `""` |
| S2-/S3- IDs only | Hard rejection | prefix assertion |
| IDs must exist in test set | Hard rejection | check against test S2/S3 entity_id sets |
| No duplicate IDs within list | Hard rejection | `len == len(set)` |
| `matching_results` ⊆ `candidate_pairs` | Warning (not rejection) | programmatic subset check in `validation.py` |
| `sep="\t"` | Hard rejection | `to_csv(sep="\t")` |

---

## 19. Detailed Engineering Architecture (Canonical Diagram #1)

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                         BUSINESS ENTITY RESOLUTION PIPELINE — FULL ENGINEERING                      │
└─────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────┐
│  PHASE 0: DATA LOADING           │  data_loading.py
│  ─────────────────────────────   │  pd.read_csv(sep="\t") for ALL files
│  train_s1 (2.2M) │ test_s1 (1.7M)│  Assert: 4 columns, entity_id prefix check
│  train_s2 (5.0M) │ test_s2 (4.9M)│  Assert: S1 has zero duplicate entity_ids (MEASURED ✓)
│  train_s3 (5.3M) │ test_s3 (5.1M)│  Assert: GT has 2.2M rows = S1 row count (MEASURED ✓)
│  train_gt (2.2M) │               │
└────────────┬─────────────────────┘
             │
             ▼
┌──────────────────────────────────┐
│  PHASE 1: PROFILING              │  profiling.py
│  ─────────────────────────────   │  Measure: null rates, country dist, name/addr lengths
│  MEASURED:                       │  Cross-country check → determines hard vs soft country gate
│  Null addr: S2 3.4%, S3 3.3%    │  Run ONCE; save JSON profile; gate before blocking design
│  Singletons: 5.58% (123K)       │
│  Matches: mode=3, max=11         │
│  Cross-country: TBD → soft mode  │
└────────────┬─────────────────────┘
             │
             ▼
┌──────────────────────────────────┐
│  PHASE 2: NORMALIZATION          │  normalization.py
│  ─────────────────────────────   │
│  Names:                          │  normalize_name() → {raw, norm, no_suffix,
│   NFKC unicode → lowercase       │                       sorted_tokens, legal_name,
│   punctuation strip → & ↔ and   │                       trade_name}
│   suffix normalization (SARL,    │
│   Pvt Ltd, Corp, LLC, etc.)      │  normalize_address() → {raw, norm, no_landmark,
│   DBA split → legal + trade      │                          house_number, city_token,
│   token sort (transpositions)    │                          pin, has_null, tokens}
│                                  │
│  Addresses:                      │  normalize_country() → lowercase strip only
│   street abbr (Rd→rd, St→st)    │
│   landmark strip (Near SBI ATM)  │  Retain BOTH raw and normalized fields
│   house number extraction        │  Null address → explicit is_null flag (not silent mismatch)
│   city/pin extraction            │
│   component reorder tolerance    │
└────────────┬─────────────────────┘
             │
             ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│  PHASE 3: BLOCKING (5 complementary rules)                                                       │  blocking.py
│  ──────────────────────────────────────────────────────────────────────────────────────────────  │
│                                                                                                  │
│  S1_entity ──→  Rule A: sorted_name_tokens[:30] (no suffix)  ──→ S2/S3 inverted index lookup   │
│             └── Rule B: house_number + street_token_1         ──→ S2/S3 inverted index lookup   │
│             └── Rule C: top-3 sorted char trigrams of name    ──→ S2/S3 inverted index lookup   │
│             └── Rule D: phonetic(name_tok_1) + city_token     ──→ S2/S3 inverted index lookup   │
│             └── Rule E: name_prefix_4 + country_norm          ──→ S2/S3 inverted index lookup   │
│                                         │                                                        │
│                                         ▼  UNION (deduplicate by candidate_id)                  │
│                               { candidate_id → {rules_fired} }                                  │
│                                                                                                  │
│  MEASURE: blocking recall vs train_gt ≥ 95% (GATE — do not proceed if below)                    │
│  MEASURE: reduction ratio = 1 - (candidates / (S1 × (S2+S3)))  (target: > 99.9%)               │
│                                                                                                  │
│  Country gate: DEFAULT soft mode. If cross-country check = 0: add country gate to rules A/B/E   │
└─────────────────────────────────────┬────────────────────────────────────────────────────────────┘
                                      │  raw candidates (up to 200/entity)
                                      ▼
┌──────────────────────────────────────┐
│  PHASE 4: COARSE FILTER (optional)   │  candidate_generation.py
│  ──────────────────────────────────  │
│  Fast Jaro-Winkler name sim > 0.65   │  Cheap to compute; cuts obvious noise
│  OR token overlap > 0.25            │  Expected: reduces 200 → ~50 candidates/entity
│  OR blocking_rules_hit ≥ 2          │
│                                      │
│  ◄─ THIS OUTPUT IS candidate_pairs.tsv ─►
│     (last pre-model candidate stage)
│     Written with all S1 IDs (empty = no candidates)
└────────────┬─────────────────────────┘
             │  pruned candidates (~50/entity average)
             ▼
┌──────────────────────────────────┐
│  PHASE 5: FEATURE ENGINEERING    │  features.py
│  ─────────────────────────────   │
│  Per candidate pair:             │
│  ┌───────────────────────────┐   │
│  │ Name features (13):       │   │  edit dist, jaro-winkler (raw + no-suffix),
│  │  exact, sorted exact,     │   │  token jaccard, token-set ratio, token-sort ratio
│  │  char 3/4-gram jaccard,   │   │  len diff, TF-IDF cosine (char 2-4gram)
│  │  jaro-winkler, tfidf cos  │   │
│  ├───────────────────────────┤   │
│  │ Address features (10):    │   │  null indicators, exact, token jaccard,
│  │  nulls, exact, token jac, │   │  char 3-gram, house num agree, city agree,
│  │  house num, city, pin,    │   │  pin agree, TF-IDF cosine
│  │  tfidf cosine             │   │
│  ├───────────────────────────┤   │
│  │ Country + conflict (6):   │   │  country_match (binary equality ONLY),
│  │  country_match, conflict, │   │  high_name_low_addr conflict,
│  │  interaction, blocking cnt│   │  name×addr interaction, n_blocking_rules
│  └───────────────────────────┘   │
│  [Optional: +2 embedding cosine] │  multilingual-MiniLM or e5-small (≤8B, MIT/Apache)
│                                  │  batch-encoded by entity_id, never per-pair
└────────────┬─────────────────────┘
             │  feature matrix (N_pairs × 32)
             ▼
┌──────────────────────────────────────────────────────────────┐
│  PHASE 6: MODEL (Training path only)                          │  train.py
│  ──────────────────────────────────────────────────────────── │
│                                                               │
│  Entity-grouped train/val split (10% val, stratified by country)
│                                                               │
│  Positives:  ground-truth matched pairs                       │
│  Easy negs:  blocked non-matches (low name sim)               │
│  Hard negs:  blocked non-matches (high name sim, wrong addr)  │
│  → Upsample hard negs 2×                                      │
│                                                               │
│  LightGBM binary classifier                                   │
│  Hyperparams: lr=0.05, n_leaves=63, subsample=0.8            │
│  Early stopping on val binary logloss                         │
│  → model.lgb (save for inference)                             │
└────────────┬─────────────────────────────────────────────────┘
             │
             ▼
┌──────────────────────────────────┐
│  PHASE 7: THRESHOLD OPTIMIZATION │  threshold.py
│  ─────────────────────────────   │
│  Sweep t ∈ [0.01, 0.99, step=0.01]
│  For each t:                     │
│    → apply to val scored pairs   │
│    → aggregate per S1 entity     │
│    → compute macro F0.5          │
│       (singleton = 1.0 or 0.0)   │
│    → record (t, F0.5, P, R)      │
│  Select t* = argmax F0.5         │
│  Check per-source (S2 vs S3)     │
│  Favor higher t if F0.5 tie      │
└────────────┬─────────────────────┘
             │  best threshold t*
             ▼
┌──────────────────────────────────────────────────────────────┐
│  PHASE 8: INFERENCE (Test path)                               │  predict.py
│  ──────────────────────────────────────────────────────────── │
│  1. Normalize test_s1, test_s2, test_s3                       │
│  2. Build blocking indices on test_s2, test_s3               │
│  3. Generate candidate pairs for test_s1                      │
│  4. Coarse filter → write candidate_pairs.tsv (test version) │
│  5. Compute features for test candidate pairs                 │
│  6. model.predict() → scores                                  │
│  7. Apply t* threshold                                        │
└────────────┬─────────────────────────────────────────────────┘
             │  above-threshold pairs per S1 entity
             ▼
┌──────────────────────────────────┐
│  PHASE 9: AGGREGATION            │  aggregation.py
│  ─────────────────────────────   │
│  For each S1 entity:             │
│    above-threshold S2 ∪ S3 IDs  │
│    deduplicate (S2-/S3- prefixes │
│    guarantee cross-source unique)│
│    sort: S2- then S3-, numeric   │
│    → comma-joined string         │
│    If empty → "" (singleton)     │
│  → match_dict[s1_id] = [ids]     │
└────────────┬─────────────────────┘
             │
             ▼
┌──────────────────────────────────┐
│  PHASE 10: OUTPUT GENERATION     │  validation.py + aggregation.py
│  ─────────────────────────────   │
│  write_matching_results()        │  → output/matching_results.tsv
│  write_candidate_pairs()         │  → output/candidate_pairs.tsv
│  Internal checks:                │
│    all S1 IDs present            │
│    no duplicates                 │
│    no S1- IDs in matches         │
│    match IDs ⊆ candidate IDs    │
└────────────┬─────────────────────┘
             │
             ▼
┌──────────────────────────────────┐
│  PHASE 11: SUBMISSION VALIDATION │  validation.py wraps validate_submission.py
│  ─────────────────────────────   │
│  python3 utils/validate_submission.py \
│    --matching output/matching_results.tsv \
│    --candidate output/candidate_pairs.tsv \
│    --test-dir dataset/test
│  → must exit code 0 before upload│
│  Fix ALL warnings (not just errors)
└──────────────────────────────────┘
             │
             ▼
      🏆 LEADERBOARD UPLOAD
```

---

## 20. Presentation Diagram (Canonical Diagram #2)

```
┌──────────────┐    ┌─────────────────┐    ┌──────────────────────┐    ┌──────────────────────┐
│   RAW DATA   │    │  NORMALIZATION  │    │  MULTI-RULE BLOCKING │    │  CANDIDATE PAIRS     │
│              │───▶│                 │───▶│                      │───▶│  + COARSE FILTER     │
│ S1 (1.7M)   │    │ Name: unicode,  │    │ Rule A: sorted name  │    │                      │
│ S2 (4.9M)   │    │  suffix norm,   │    │ Rule B: addr+house   │    │ candidate_pairs.tsv  │
│ S3 (5.1M)   │    │  token sort,    │    │ Rule C: char trigram │    │ (last pre-model      │
│ test         │    │  DBA split      │    │ Rule D: phonetic+city│    │  candidate stage)    │
└──────────────┘    │ Addr: landmark  │    │ Rule E: prefix+ctry  │    └──────────┬───────────┘
                    │  strip, abbr,   │    │                      │               │
                    │  component norm │    │ UNION → ≥95% recall  │               │
                    │ Country: string │    │ gate before model    │               │
                    │  equality only  │    └──────────────────────┘               │
                    └─────────────────┘                                           │
                                                                                  ▼
┌──────────────────────────┐    ┌──────────────────────────────┐    ┌───────────────────────┐
│  MATCHING MODEL          │    │  PRECISION-AWARE DECISION    │    │  MATCH SETS           │
│                          │◀───│                              │◀───│                       │
│ LightGBM binary classif. │    │ Threshold sweep on val split │    │ Per-entity scoring:   │
│ Features (~32):          │    │ Macro F0.5 (per-entity,      │    │  P, R, F0.5 computed  │
│  name: JW, token Jaccard,│    │  singleton = 1.0/0.0)        │    │  at every threshold   │
│  char ngram, TF-IDF cos  │    │ Select t* = argmax F0.5      │    │                       │
│  addr: null, house, city │    │ Favor precision (2× weight)  │    │ matching_results.tsv  │
│  country: equality only  │    │ No fallback for singletons   │    │ (one row per S1       │
│  conflict: high-sim detec│    └──────────────────────────────┘    │  entity; empty=single)│
└──────────────────────────┘                                        └───────────┬───────────┘
                                                                                │
                                                                                ▼
                                                                   ┌───────────────────────┐
                                                                   │  SUBMISSION           │
                                                                   │  validate_submission  │
                                                                   │  .py → PASS → upload  │
                                                                   └───────────────────────┘
```

---

## 21. File / Folder Structure

```
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv          # scored on leaderboard
│   └── candidate_pairs.tsv           # blocking audit (not scored)
│
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       │   ├── data_loading.py       # sep="\t" everywhere; column assertions
│       │   ├── profiling.py          # distributions, cross-country check, GT stats
│       │   ├── normalization.py      # name/addr/country normalizers + suffix map
│       │   ├── blocking.py           # 5-rule inverted index + recall measurement
│       │   ├── candidate_generation.py  # coarse filter → writes candidate_pairs.tsv
│       │   ├── features.py           # all pairwise features + TF-IDF model + optional embedding
│       │   ├── train.py              # entity-grouped split, hard-neg sampling, LightGBM fit
│       │   ├── predict.py            # test-set inference pipeline
│       │   ├── threshold.py          # macro-F0.5 sweep + f05_score() unit-tested
│       │   ├── aggregation.py        # per-entity match list + write_matching_results()
│       │   └── validation.py         # wraps validate_submission.py; internal checks
│       │
│       ├── README.md                 # exact end-to-end run instructions:
│       │                             #   python3 src/profiling.py
│       │                             #   python3 src/train.py
│       │                             #   python3 src/predict.py
│       │                             #   python3 src/validation.py
│       │
│       └── requirements.txt          # pinned:
│                                     #   pandas==2.2.2
│                                     #   lightgbm==4.5.0
│                                     #   scikit-learn==1.5.1
│                                     #   rapidfuzz==3.9.4
│                                     #   jellyfish==1.0.3
│                                     #   numpy==1.26.4
│                                     #   [sentence-transformers==3.0.0 if embedding used]
│
└── Documentation_template.md         # methodology, blocking strategy, model+features,
                                       # pretrained model license/param disclosure (if used),
                                       # suffix list disclosure (general domain knowledge)
```

---

## 22. Experimentation Plan

Track for every experiment: **macro F0.5, precision, recall, false-merge rate, blocking recall, reduction ratio, candidate count/entity, wall-clock runtime.**

| Step | Experiment | What it tests | Success metric |
|------|-----------|--------------|----------------|
| 1 | Exact normalized name matching only (no model) | Baseline upper bound on precision | F0.5 on val |
| 2 | + Multi-rule blocking + fast string features (JW, token jaccard) | Recall ceiling | Blocking recall ≥ 95% (GATE) |
| 3 | + LightGBM on full feature set | Model adds value over rules | F0.5 > step 1 |
| 4 | + TF-IDF cosine features | Sparse text representation helps | Feature importance of TF-IDF; F0.5 |
| 5 | + Hard-negative upsampling | Reduces false positives | Precision on val |
| 6 | Threshold optimization sweep | Find t* | t* that maximizes macro F0.5 |
| 7 | Per-source (S2 vs S3) threshold | S2/S3 noise profiles differ | F0.5 per source |
| 8 | Ablation: remove `country_match` | Confirm it helps without hurting France | F0.5 stability |
| 9 | (Optional) + embedding cosine feature | Cross-script/transliteration improvement | F0.5 on India subset |
| 10 | (Optional) Coarse filter threshold tuning | Reduce candidate count without recall loss | Blocking recall stable |

---

## 23. Computational Complexity

| Stage | Complexity | Bottleneck | Mitigation |
|-------|-----------|-----------|-----------|
| Raw pair space | O(S1 × (S2+S3)) ≈ 1.73×10¹³ | Infeasible | Blocking reduces by >99.9% |
| Blocking index build | O(S2+S3) | One-time per inference | Parallelizable per rule |
| Candidate generation | O(S1 × k) where k=avg candidates | k~200 → 346M pairs | Explosion control (cap at 200) |
| Coarse filter | O(S1 × k) × O(JW) | k~200, JW O(|s|²) | Single-pass per pair; fast with rapidfuzz C++ backend |
| Feature computation | O(candidates × n_features) | After coarse filter: ~50×S1 pairs | Vectorized pandas; TF-IDF batch transform |
| TF-IDF transform | O(candidates × vocab) | Sparse matrix ops | scipy.sparse; batched |
| **Embedding inference** | O(candidates × model_cost) | **MAIN bottleneck if used** | **Batch by entity_id; cache; use smallest model** |
| LightGBM predict | O(candidates × n_trees × n_leaves) | Fast with compiled C++ | — |
| Total train inference | ~86M pairs (after coarse filter) | Feature computation | Parallelize across S1 batches |
| Total test inference | ~86M pairs | Feature computation | Parallelize across S1 batches |

**Embedding batching pattern:**
```python
# DO NOT:
for pair in candidate_pairs:
    emb = model.encode(pair.s1_name)   # O(N_pairs) model calls

# DO:
unique_s1_names = s1_df["norm_name"].unique()
s1_embeddings = model.encode(unique_s1_names, batch_size=256, show_progress_bar=True)
s1_emb_map = dict(zip(unique_s1_names, s1_embeddings))
# Then lookup by entity_id at feature-computation time
```

---

## 24. Failure / Edge Case Handling

| Edge case | Risk | Handling |
|-----------|------|---------|
| **France entities in test** | Zero training signal → unknown performance | Country as equality feature only; embedding model trained on multilingual corpus provides cross-script generalization |
| **Devanagari script names** | Edit-distance metrics behave differently on multi-byte chars | NFKC normalization + char n-gram similarity at character level (not byte level); multilingual embedding as optional fallback |
| **Null address in S2/S3 (~3.4%)** | Can't compute address features | Explicit `addr_cand_null` binary feature; address features set to 0 (not NaN) when null |
| **DBA names** | Legal name vs. trade name mismatch | Split on DBA/t/a pattern; create legal_name and trade_name variants; block on both |
| **Explosion in blocking** | Common words like "Services", "India" create massive blocks | Cap at 200 candidates/entity; prefer candidates with multiple rule hits |
| **Singleton false merges** | Costs 1.0 on F0.5 per entity | Precision-biased threshold; no fallback; SHAP inspection of FP singletons |
| **S1-S1 self-matches** | Hard rejection by leaderboard | Assertion: all match IDs must start with S2- or S3- |
| **Missing S1 entity in output** | Hard rejection | Build result dict from all S1 IDs before output; fill missing with `""` |
| **TF-IDF vocabulary drift (train vs test)** | Test has France business names not in train corpus | Fit TF-IDF on train+test (no label leakage; fitting on text only) |
| **Very long address strings (max 256 chars)** | Edit distance O(n²) is slow | Use TF-IDF cosine instead of edit distance for long addresses |
| **Comma in entity IDs** | Would corrupt comma-separated ID list | SPEC says IDs never contain commas; assert at load time |
| **Unicode BOM in TSV** | encoding error | `encoding="utf-8-sig"` in read_csv |
| **Whitespace-only business_name** | Empty after normalization | Treat as empty string; skip name-similarity features (set to 0) |

---

## 25. Output Generation (Exact Schemas)

```python
# src/aggregation.py

def generate_outputs(
    scored_test_pairs: pd.DataFrame,   # cols: source1_entity_id, candidate_id, score
    all_test_s1_ids: list[str],        # from test_source1.tsv entity_id column
    pruned_candidates: pd.DataFrame,   # cols: source1_entity_id, candidate_id (post-coarse-filter)
    threshold: float,
    output_dir: str,
):
    # ── matching_results.tsv ──────────────────────────────────────
    match_dict = {s1_id: [] for s1_id in all_test_s1_ids}
    above = scored_test_pairs[scored_test_pairs["score"] >= threshold]
    for s1_id, grp in above.groupby("source1_entity_id"):
        cands = list(dict.fromkeys(  # dedupe preserving order
            sorted(grp["candidate_id"].tolist(),
                   key=lambda x: (x[:2], int(x[3:])))
        ))
        match_dict[s1_id] = cands

    matching_rows = [
        {"source1_entity_id": s1,
         "matched_entity_ids": ",".join(match_dict[s1])}
        for s1 in all_test_s1_ids
    ]
    pd.DataFrame(matching_rows).to_csv(
        f"{output_dir}/matching_results.tsv", sep="\t", index=False, encoding="utf-8"
    )

    # ── candidate_pairs.tsv ───────────────────────────────────────
    cand_dict = {s1_id: [] for s1_id in all_test_s1_ids}
    for s1_id, grp in pruned_candidates.groupby("source1_entity_id"):
        cand_dict[s1_id] = list(dict.fromkeys(grp["candidate_id"].tolist()))

    candidate_rows = [
        {"source1_entity_id": s1,
         "candidate_entity_ids": ",".join(cand_dict[s1])}
        for s1 in all_test_s1_ids
    ]
    pd.DataFrame(candidate_rows).to_csv(
        f"{output_dir}/candidate_pairs.tsv", sep="\t", index=False, encoding="utf-8"
    )
```

---

## 26. Submission Validation

```bash
# Run before EVERY leaderboard upload — no exceptions
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test

# With ID existence check (memory-heavy but thorough):
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test \
    --check-ids
```

```python
# src/validation.py — programmatic pre-submission check

import subprocess, sys

def validate_before_submit(student_resource_dir: str) -> bool:
    result = subprocess.run(
        ["python3", "utils/validate_submission.py",
         "--matching", "output/matching_results.tsv",
         "--candidate", "output/candidate_pairs.tsv",
         "--test-dir", "dataset/test"],
        cwd=student_resource_dir,
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print("SUBMISSION VALIDATION FAILED:", result.stderr)
        return False
    # Also check for warnings — fix them even though they don't hard-fail
    if "WARNING" in result.stdout:
        print("WARNING detected — investigate before submitting")
    return True
```

**Issue severity mapping:**

| Issue type | Validator behavior | Our response |
|-----------|-------------------|-------------|
| Missing row for an S1 entity | Hard FAIL (exit 1) | Fix before any upload |
| Duplicate source1\_entity\_id | Hard FAIL | Fix before any upload |
| S1- ID in matched list | Hard FAIL | Fix before any upload |
| Unknown S2-/S3- ID | Hard FAIL | Fix before any upload (run with --check-ids) |
| Intra-list duplicate IDs | Hard FAIL | Fix before any upload |
| match IDs ⊄ candidate IDs | WARNING only | Still fix — signals pipeline bug |
| ID existence not checked | WARNING (default mode) | Run with --check-ids on final submission |

---

## 27. Implementation Roadmap

```
WEEK 1: Foundation
────────────────────────────────────────────────────────────────────────────────
Step 1:  python3 src/profiling.py
         → Confirm all MEASURED stats; run cross-country check; determine country gate mode
         → Gate: confirm S1 has zero dups; confirm GT = S1 count

Step 2:  python3 src/normalization.py  (unit tests)
         → Test: "Corporation" → "corp"; "Rd" → "rd"; landmark stripped
         → Test: sorted("Prime Money") == sorted("Money Prime")
         → Test: Devanagari "प्राइवेट लिमिटेड" → "pvt ltd"

Step 3:  python3 src/blocking.py --mode train
         → Build blocking indices on train S2/S3
         → Measure per-rule recall + union recall vs train_gt
         → GATE: union recall ≥ 95%  (DO NOT PROCEED if below — fix rules first)
         → Measure reduction ratio (target > 99.9%)

Step 4:  Baseline: exact normalized name match only
         → Compute F0.5 on val split; set baseline target to beat

WEEK 2: Feature Engineering + Model
────────────────────────────────────────────────────────────────────────────────
Step 5:  python3 src/candidate_generation.py --mode train
         → Apply coarse filter; write train candidate_pairs.tsv
         → Confirm coarse filter doesn't drop recall below 95%

Step 6:  python3 src/features.py --mode train
         → Compute 32-feature vector for all training candidate pairs
         → Check: no NaN; feature distributions make sense (name_exact rate ≈ GT match rate)

Step 7:  python3 src/train.py
         → Entity-grouped train/val split
         → Hard-negative upsampling
         → LightGBM fit; inspect feature importance
         → SHAP: confirm top features are name_jaro_winkler, name_token_set_ratio, etc.

Step 8:  python3 src/threshold.py
         → Sweep on val; plot F0.5/P/R vs threshold
         → Select t*; unit-test F0.5 code against worked example
         → Verify t* gives macro F0.5 > baseline (exact match only)

WEEK 3: Refinement + Ablations
────────────────────────────────────────────────────────────────────────────────
Step 9:  TF-IDF cosine features
         → Fit TF-IDF on all names/addresses (train + test text; no label leakage)
         → Add to feature vector; retrain; compare F0.5

Step 10: Hard negative audit
         → Inspect top false positives: what do they look like?
         → Add blocking rule or feature targeting the dominant FP pattern

Step 11: Country ablation (confirm strategy A is correct)
         → Remove country_match feature; retrain; compare F0.5
         → If F0.5 drops: feature helps → keep it
         → If F0.5 stable: feature is noise for seen countries; keep anyway for France safety

Step 12: (Optional) Embedding cosine feature
         → Encode candidate names with multilingual-MiniLM (batch by entity_id)
         → Add cosine as 2 features; retrain; compare
         → Document license + param count in Documentation_template.md

WEEK 4: Test Inference + Submission
────────────────────────────────────────────────────────────────────────────────
Step 13: python3 src/predict.py
         → Full test inference: blocking → coarse filter → features → model → scores

Step 14: python3 src/aggregation.py
         → Apply t*; aggregate per-entity; write matching_results.tsv + candidate_pairs.tsv

Step 15: python3 src/validation.py
         → Run validate_submission.py → must be PASS exit 0
         → Fix any warnings (even if not hard-fail)

Step 16: Write Documentation_template.md
         → Methodology, blocking strategy, model/features
         → Suffix list disclosure (domain knowledge, not entity lookup)
         → Pretrained model disclosure if used (license + param count + why not entity lookup)

Step 17: Package submission zip per Section 21 structure
         → Verify both TSVs present; verify code reproduces outputs from README.md instructions
```

---

## 28. Final Architecture Quality Check

| Checklist item | Status |
|----------------|--------|
| ☑ Schema treated as FACT, not re-guessed | Entity columns: entity\_id, business\_name, business\_address, country — confirmed from real files |
| ☑ Exact macro-F0.5 + singleton 1.0/0.0 rule in threshold code | `f05_score()` function with worked-example unit test (P=2/3,R=1→0.714) |
| ☑ Country handled as open-set equality signal only | `country_match = int(norm(c1) == norm(c2))`; no one-hot, no per-country model |
| ☑ candidate\_pairs.tsv = LAST pre-model candidate stage | Coarse-filter output, not raw blocking union; explicitly stated in Section 9 |
| ☑ Cross-country match check MUST run before finalizing blocking | Cross-country check script written; soft-signal mode as default until confirmed |
| ☑ Pretrained model (if used): ≤8B params, MIT/Apache-2.0, disclosed | multilingual-MiniLM (117M, Apache-2.0) or e5-small (118M, MIT); Documentation\_template.md disclosure required |
| ☑ No external data/API/lookup | No geocoding, no business registries, no internet; static suffix list is general domain knowledge |
| ☑ Singletons: empty list, no fallback | `match_dict = {s1_id: []}` initialized for ALL S1 entities; no nearest-neighbor fallback |
| ☑ Multi-match aggregation: deterministic, deduped, prefixes kept | `sort_key = (prefix, int_id)`; `dict.fromkeys()` for dedup |
| ☑ Leakage-safe entity-grouped train/val split | `GroupShuffleSplit` on S1 entity\_id, stratified by country |
| ☑ validate\_submission.py run before every upload | Wrapped in `validation.py`; must exit 0 |
| ☑ Pipeline diagram drawn exactly ONCE at each granularity | Section 19 (engineering) and Section 20 (presentation) — referenced by number elsewhere |
| ☑ Directly implementable without further guessing | Every block has: Purpose, Input, Processing, Output, pseudocode/function signatures |

---

## 29. What to Build First

```
FIRST BUILD:
  data_loading.py + profiling.py
  → Run cross-country check (the single most critical unknown)
  → Measure country distribution in test_s1 (France fraction)
  → Confirm all MEASURED stats from this document

THEN BUILD:
  normalization.py (with unit tests)
  blocking.py (5 rules)
  → Run blocking recall gate: must reach ≥95% on train_gt before proceeding
  → Iterate on rules until gate passes

THEN BUILD:
  candidate_generation.py (coarse filter → candidate_pairs.tsv)
  features.py (all 32 features including TF-IDF)
  train.py (entity-grouped split, hard-neg sampling, LightGBM)
  threshold.py (macro-F0.5 sweep with worked-example unit test)
  → First model: aim to beat exact-match baseline on val F0.5

THEN BUILD:
  predict.py (full test inference)
  aggregation.py (output generation)
  validation.py (validate_submission.py wrapper)
  Documentation_template.md
  → Run validate_submission.py → PASS → upload

ITERATE:
  Hard-negative audit → blocking rule refinement
  TF-IDF cosine features (add to step 3 feature set)
  Threshold re-optimization
  (Optional) embedding cosine feature if val F0.5 plateaus
  (Optional) per-source threshold (S2 vs S3)
```

---

## 30. Appendix: Disclosed Static Suffix List

The following is a **static list of general legal entity suffixes**, used in normalization (Section 7.1). This is general domain knowledge — not entity-specific lookup. Disclosed here per fair-play rules.

```python
LEGAL_SUFFIX_MAP = {
    # English / US / Commonwealth
    "corporation": "corp", "corp.": "corp",
    "incorporated": "inc", "inc.": "inc",
    "limited": "ltd", "ltd.": "ltd",
    "company": "co", "co.": "co",
    "private": "pvt", "pvt.": "pvt",
    "limited liability company": "llc",
    "limited liability partnership": "llp",
    "limited partnership": "lp",
    "public limited company": "plc",
    # India-specific
    "private limited": "pvt ltd",
    "pvt. ltd.": "pvt ltd",
    "& co.": "co",
    # French (DISCLOSED — general domain knowledge, no entity lookup)
    "société anonyme": "sa",
    "s.a.": "sa",
    "société à responsabilité limitée": "sarl",
    "s.a.r.l.": "sarl",
    "société par actions simplifiée": "sas",
    "s.a.s.": "sas",
    "société en nom collectif": "snc",
    "entreprise individuelle": "ei",
    "entreprise unipersonnelle à responsabilité limitée": "eurl",
    "e.u.r.l.": "eurl",
    # Transliterations in Latin script
    "pvt ltd": "pvt ltd",
    "priv. ltd.": "pvt ltd",
}
# Devanagari forms handled separately via NFKC normalization pass
```
