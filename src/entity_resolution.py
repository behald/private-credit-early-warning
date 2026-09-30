"""Borrower entity resolution across BDC filings.

Normalizes borrower names and matches them to canonical entities using:
1. Name normalization (strip legal suffixes, punctuation, case)
2. Blocking by industry + first character
3. Embedding similarity (sentence-transformers on CPU)
4. Rule-based matching (industry, rate, maturity overlap)
5. Manual override table
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
from pydantic import BaseModel

logger = logging.getLogger(__name__)

LEGAL_SUFFIXES = [
    r"\bllc\b", r"\binc\b", r"\bcorp\b", r"\bcorporation\b",
    r"\bltd\b", r"\blimited\b", r"\blp\b", r"\bco\b",
    r"\bholdings?\b", r"\bgroup\b", r"\benterprise[s]?\b",
    r"\binternational\b", r"\bintl\b", r"\bglobal\b",
    r"\bsolutions?\b", r"\bservices?\b", r"\btechnolog(?:y|ies)\b",
    r"\bpartners?\b", r"\bassociates?\b", r"\bcapital\b",
    r"\bfinancial\b", r"\bmanagement\b", r"\bconsulting\b",
]

LEGAL_PATTERN = re.compile("|".join(LEGAL_SUFFIXES), re.IGNORECASE)

COMMON_ABBREVIATIONS = {
    "intl": "international",
    "tech": "technology",
    "mgmt": "management",
    "svcs": "services",
    "sys": "systems",
    "mfg": "manufacturing",
    "natl": "national",
    "amer": "american",
    "assoc": "associates",
    "ind": "industries",
}


class CanonicalBorrower(BaseModel):
    canonical_id: str
    canonical_name: str
    raw_names: list[str]
    industry: str | None = None
    first_seen_quarter: str | None = None
    last_seen_quarter: str | None = None
    lender_count: int = 0


class MatchResult(BaseModel):
    name_a: str
    name_b: str
    normalized_a: str
    normalized_b: str
    similarity_score: float
    match: bool
    method: str


def normalize_name(name: str) -> str:
    """Normalize a borrower name for matching."""
    s = name.lower().strip()
    s = LEGAL_PATTERN.sub("", s)
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    for abbr, full in COMMON_ABBREVIATIONS.items():
        s = re.sub(rf"\b{abbr}\b", full, s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def generate_canonical_id(name: str) -> str:
    """Generate a stable canonical ID from a normalized name."""
    normalized = normalize_name(name)
    return hashlib.md5(normalized.encode()).hexdigest()[:12]


def load_manual_overrides(path: Path) -> dict[str, str]:
    """Load manual override table: raw_name -> canonical_id."""
    overrides: dict[str, str] = {}
    if not path.exists():
        return overrides
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw = row.get("raw_name", "").strip()
            canonical = row.get("canonical_id", "").strip()
            if raw and canonical:
                overrides[normalize_name(raw)] = canonical
    return overrides


def save_manual_overrides_template(path: Path, sample_names: list[str]) -> None:
    """Create a template CSV for manual labeling of entity pairs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name_a", "name_b", "match"])
        writer.writeheader()
        normalized = [(n, normalize_name(n)) for n in sample_names]
        seen = set()
        for i, (raw_a, norm_a) in enumerate(normalized):
            for j, (raw_b, norm_b) in enumerate(normalized):
                if i >= j:
                    continue
                if norm_a == norm_b:
                    continue
                pair_key = tuple(sorted([norm_a, norm_b]))
                if pair_key in seen:
                    continue
                seen.add(pair_key)
                if len(seen) > 300:
                    break
                writer.writerow({"name_a": raw_a, "name_b": raw_b, "match": ""})


def build_blocks(
    names: list[str],
    industries: list[str | None],
) -> dict[str, list[int]]:
    """Create blocking groups to reduce pairwise comparisons."""
    blocks: dict[str, list[int]] = {}
    for i, (name, industry) in enumerate(zip(names, industries)):
        normalized = normalize_name(name)
        if not normalized:
            continue
        first_char = normalized[0]
        ind = (industry or "unknown").lower()[:20]
        block_key = f"{first_char}|{ind}"
        blocks.setdefault(block_key, []).append(i)
        char_key = f"{first_char}|*"
        blocks.setdefault(char_key, []).append(i)
        first_word = normalized.split()[0] if normalized.split() else ""
        if first_word and len(first_word) > 2:
            word_key = f"w|{first_word}"
            blocks.setdefault(word_key, []).append(i)
    return blocks


def compute_embedding_similarity(
    names: list[str],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 64,
) -> np.ndarray:
    """Compute pairwise cosine similarity using sentence-transformers on CPU."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.warning("sentence-transformers not installed, skipping embedding similarity")
        return np.zeros((len(names), len(names)))

    model = SentenceTransformer(model_name, device="cpu")
    normalized = [normalize_name(n) for n in names]
    embeddings = model.encode(normalized, show_progress_bar=True, batch_size=batch_size)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    normalized_embeddings = embeddings / norms
    similarity = normalized_embeddings @ normalized_embeddings.T
    return similarity


def resolve_entities(
    raw_names: list[str],
    industries: list[str | None],
    manual_overrides_path: Path | None = None,
    similarity_threshold: float = 0.85,
    use_embeddings: bool = True,
) -> dict[str, str]:
    """Resolve raw borrower names to canonical IDs.

    Returns mapping of raw_name -> canonical_id.
    """
    overrides = {}
    if manual_overrides_path:
        overrides = load_manual_overrides(manual_overrides_path)

    unique_names = sorted(set(raw_names))
    name_to_industry: dict[str, str | None] = {}
    for name, ind in zip(raw_names, industries):
        if name not in name_to_industry:
            name_to_industry[name] = ind

    unique_industries = [name_to_industry.get(n) for n in unique_names]
    name_to_canonical: dict[str, str] = {}

    # Step 1: Manual overrides
    for name in unique_names:
        norm = normalize_name(name)
        if norm in overrides:
            name_to_canonical[name] = overrides[norm]

    # Step 2: Exact normalized match
    norm_to_canonical: dict[str, str] = {}
    for name in unique_names:
        if name in name_to_canonical:
            continue
        norm = normalize_name(name)
        if norm in norm_to_canonical:
            name_to_canonical[name] = norm_to_canonical[norm]
        else:
            cid = generate_canonical_id(name)
            norm_to_canonical[norm] = cid
            name_to_canonical[name] = cid

    # Step 3: Embedding similarity within blocks
    if use_embeddings and len(unique_names) > 1:
        unresolved = [n for n in unique_names if n not in name_to_canonical or
                      name_to_canonical.get(n) == generate_canonical_id(n)]
        if len(unresolved) > 1:
            similarity = compute_embedding_similarity(unresolved)
            blocks = build_blocks(
                unresolved,
                [name_to_industry.get(n) for n in unresolved],
            )

            parent: dict[int, int] = {i: i for i in range(len(unresolved))}

            def find(x: int) -> int:
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            def union(x: int, y: int) -> None:
                px, py = find(x), find(y)
                if px != py:
                    parent[px] = py

            for block_indices in blocks.values():
                for i_idx in range(len(block_indices)):
                    for j_idx in range(i_idx + 1, len(block_indices)):
                        i, j = block_indices[i_idx], block_indices[j_idx]
                        if similarity[i, j] >= similarity_threshold:
                            union(i, j)

            clusters: dict[int, list[int]] = {}
            for i in range(len(unresolved)):
                root = find(i)
                clusters.setdefault(root, []).append(i)

            for members in clusters.values():
                canonical = generate_canonical_id(unresolved[members[0]])
                for idx in members:
                    name_to_canonical[unresolved[idx]] = canonical

    result: dict[str, str] = {}
    for name in raw_names:
        result[name] = name_to_canonical.get(name, generate_canonical_id(name))

    return result


def resolve_and_save(
    db_path: Path,
    manual_overrides_path: Path | None = None,
    use_embeddings: bool = True,
) -> dict[str, str]:
    """Run entity resolution on the warehouse and save canonical mappings."""
    con = duckdb.connect(str(db_path))

    rows = con.execute("""
        SELECT DISTINCT borrower_name_raw, industry
        FROM raw_soi_positions
        WHERE borrower_name_raw IS NOT NULL
        ORDER BY borrower_name_raw, industry
    """).fetchall()

    raw_names = [r[0] for r in rows]
    industries = [r[1] for r in rows]

    logger.info(f"Resolving {len(raw_names)} unique borrower names")
    mapping = resolve_entities(
        raw_names, industries,
        manual_overrides_path=manual_overrides_path,
        use_embeddings=use_embeddings,
    )

    con.execute("DROP TABLE IF EXISTS borrower_canonical_map")
    con.execute("""
        CREATE TABLE borrower_canonical_map (
            borrower_name_raw VARCHAR,
            canonical_borrower_id VARCHAR,
            normalized_name VARCHAR
        )
    """)

    for raw, canonical in mapping.items():
        con.execute(
            "INSERT INTO borrower_canonical_map VALUES (?, ?, ?)",
            [raw, canonical, normalize_name(raw)]
        )

    count = con.execute("SELECT COUNT(DISTINCT canonical_borrower_id) FROM borrower_canonical_map").fetchone()[0]
    logger.info(f"Resolved to {count} canonical borrowers")

    con.close()
    return mapping


def evaluate_resolution(
    predictions: dict[str, str],
    labeled_pairs_path: Path,
) -> dict[str, Any]:
    """Evaluate entity resolution against hand-labeled pairs."""
    if not labeled_pairs_path.exists():
        logger.warning(f"No labeled pairs at {labeled_pairs_path}")
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "pairs_evaluated": 0}

    tp = fp = fn = tn = 0
    with open(labeled_pairs_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name_a = row["name_a"]
            name_b = row["name_b"]
            match_label = row.get("match", "").strip()
            if match_label == "":
                continue
            actual_match = int(match_label)

            pred_a = predictions.get(name_a, generate_canonical_id(name_a))
            pred_b = predictions.get(name_b, generate_canonical_id(name_b))
            predicted_match = 1 if pred_a == pred_b else 0

            if actual_match == 1 and predicted_match == 1:
                tp += 1
            elif actual_match == 0 and predicted_match == 1:
                fp += 1
            elif actual_match == 1 and predicted_match == 0:
                fn += 1
            else:
                tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    metrics = {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "pairs_evaluated": tp + fp + fn + tn,
    }

    logger.info(f"Entity resolution evaluation: {metrics}")
    return metrics
