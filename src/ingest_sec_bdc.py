"""Ingest SEC DERA BDC Data Sets (soi.tsv) into the DuckDB warehouse.

Source: https://www.sec.gov/data-research/sec-markets-data/bdc-data-sets

Accepts either the slim Parquet made by filter_bdc.py, or the raw
*_bdc.zip files directly. Writes raw_soi_positions in the schema the rest
of the pipeline expects, plus extra columns (principal, rates, maturity).

Cleaning steps, each one fixes a problem found in the real data:
  1. Column labels: fair value and cost sit under custom labels
     ("Initial fair value of Investment", "Adjusted cost basis").
  2. Comparative periods: each filing repeats prior periods. Keep only
     rows where ddate equals the filing's own period.
  3. Amendments: if a 10-Q/A or 10-K/A exists, keep only the latest
     filing per lender and period.
  4. Lender renames (Owl Rock to Blue Owl): lenders are keyed by CIK and
     given their most recent name.
  5. Subtotal rows and junk: dropped by pattern and by size.
  6. Borrower names: parsed out of long identifier strings.
  7. Undrawn positions: zero or missing cost dropped.

Usage:
    python src/ingest_sec_bdc.py
    python src/ingest_sec_bdc.py --source "C:/Users/you/Downloads"
"""

from __future__ import annotations

import argparse
import warnings
import hashlib
import io
import logging
import re
import zipfile
from pathlib import Path

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", message="This pattern is interpreted as a regular expression")

DEFAULT_SOURCE = Path("data/raw/soi_all_bdcs.parquet")
DB_PATH = Path("data/processed/warehouse.duckdb")

# Candidate source columns for each field, first match wins.
# The SEC headers use display labels, and some are mislabelled, so we
# list the label we actually observed first.
COLUMN_CANDIDATES = {
    "fair_value": ["Initial fair value of Investment", "Investment Owned, Fair Value"],
    "cost": ["Adjusted cost basis", "Investment Owned, Cost"],
    "principal": ["Investment Owned, Balance, Principal Amount"],
    "identifier": ["Investment, Identifier Axis"],
    "issuer_name": ["Investment, Issuer Name Axis", "Investment, Issuer Name [Extensible Enumeration]"],
    "interest_rate": ["Investment Interest Rate"],
    "spread": ["Investment, Basis Spread, Variable Rate"],
    "pik_rate": ["Investment, Interest Rate, Paid in Kind"],
    "maturity_date": ["Investment Maturity Date"],
    "pct_net_assets": ["Investment Owned, Net Assets, Percentage"],
    "industry_axis": ["Industry Sector Axis"],
    "industry_enum": ["Investment, Industry Sector [Extensible Enumeration]"],
    "type_axis": ["Investment Type Axis"],
    "type_enum": ["Investment, Type [Extensible Enumeration]"],
}

# Legal suffixes that usually end a company name inside an identifier
SUFFIX_RE = re.compile(
    r"\b(Inc|LLC|L\.L\.C|Corp|Corporation|Ltd|Limited|L\.P|LP|LLP|Co|Company|"
    r"plc|PLC|GmbH|S\.A|SA|S\.A\.R\.L|B\.V|BV|N\.V|AB|AS|SAS|Pty|ULC|LTD)\b\.?",
)

# Text that marks the start of the tranche description after the name
TRANCHE_RE = re.compile(
    r"\b(first|second|1st|2nd|senior|junior|subordinated|unitranche|one stop|term loan|"
    r"revolv|delayed draw|ddtl|preferred|common|warrant|equity|units?\b|class [a-z]\b|"
    r"series [a-z0-9]\b|note|bond|loan|debt|secured|unsecured|lp interest|membership|"
    r"incremental|amendment|tranche|add-on)",
    re.IGNORECASE,
)

# Leading category words to strip before looking for the name
PREFIX_PATTERNS = [
    r"^investments?\b[\s:-]*",
    r"^(non-?controlled|controlled|affiliated|non-?affiliated)[\w\s/,-]*?investments?[\s,:-]*",
    r"^(debt|equity) (investments?|securities)[\s,:-]*",
    r"^(us|u\.s\.) corporate debt[\s|,-]*",
    r"^(first|second|1st|2nd) lien[\w\s/-]*?(debt|loans?|secured)?[\s,:-]*",
    r"^senior secured[\w\s-]*?(loans?|debt)?[\s,:-]*",
]

# Markers that an identifier is a subtotal or not a borrower at all
NON_POSITION_RE = re.compile(
    r"(^total\b|\btotal investments\b|cash (and cash )?equivalents|money market|"
    r"treasury bill|other cash|forward (foreign )?currency|interest rate swap|"
    r"^investment [a-z ,&]+ - ?-?\d+(\.\d+)?%$|^[a-z ,&/-]+ - ?-?\d+(\.\d+)?%$)",
    re.IGNORECASE,
)

PIK_RE = re.compile(r"\bPIK\b|paid[- ]in[- ]kind", re.IGNORECASE)
NON_ACCRUAL_RE = re.compile(r"non[- ]?accrual", re.IGNORECASE)
EQUITY_RE = re.compile(
    r"\b(common|preferred|equity|warrants?|units?|membership interest|lp interest|shares)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------- loading

def _read_soi_from_zip(path: Path) -> pd.DataFrame | None:
    with zipfile.ZipFile(path) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith("soi.tsv")]
        if not names:
            logger.warning(f"No soi.tsv in {path.name}")
            return None
        raw = zf.read(names[0])
    for enc in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(raw), sep="\t", dtype=str, encoding=enc,
                               on_bad_lines="skip", low_memory=False)
        except UnicodeDecodeError:
            continue
    return None


def load_source(source: Path) -> pd.DataFrame:
    """Load from a Parquet/CSV file, or from a folder of *_bdc.zip files."""
    if source.is_dir():
        zips = sorted(source.glob("*_bdc.zip"))
        if not zips:
            raise FileNotFoundError(f"No *_bdc.zip files in {source}")
        frames = []
        for z in zips:
            logger.info(f"Reading {z.name}")
            df = _read_soi_from_zip(z)
            if df is not None:
                df["source_file"] = z.name
                frames.append(df)
        return pd.concat(frames, ignore_index=True)
    if source.suffix == ".parquet":
        return pd.read_parquet(source)
    return pd.read_csv(source, dtype=str, low_memory=False)


def _pick(df: pd.DataFrame, field: str) -> pd.Series:
    """Return the first candidate column that has data, else an empty series."""
    for col in COLUMN_CANDIDATES[field]:
        if col in df.columns and df[col].notna().any():
            return df[col]
    return pd.Series(pd.NA, index=df.index, dtype="object")


# ---------------------------------------------------------------- parsing

def _decamel_member(value: object) -> str | None:
    """'http://x.com/2024#FirstLienSecuredTermLoanMember' -> 'First Lien Secured Term Loan'."""
    if not isinstance(value, str) or not value.strip():
        return None
    s = value.split("#")[-1]
    s = re.sub(r"\s*\[?Member\]?$", "", s)
    s = re.sub(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", " ", s)
    return re.sub(r"\s+", " ", s).strip() or None


def parse_borrower(identifier: str) -> str | None:
    """Pull the borrower name out of a free text SOI identifier.

    Handles the common layouts seen across BDCs:
      'Pluralsight, LLC, First Lien Term Loan'
      'Debt Investments | Adelaide Borrower, LLC | First Lien ...'
      '... Senior Secured Debt - 186.10% Pluralsight, Inc Industry ...'
      '... Company JTM Foods, LLC, Industry Food Products ...'
      'Shiftkey, LLC-Senior Secured First Lien Term Loan Due 6/21/2027'
    """
    if not isinstance(identifier, str):
        return None
    s = re.sub(r"\s+", " ", identifier).strip()
    if not s:
        return None

    # Explicit labels used by some BDCs
    m = re.search(r"(?:^|[,\-]\s*)(?:Company|Issuer Name(?: By)?|Portfolio Company)\s+([A-Z0-9].+?)(?:,\s*(?:Industry|Maturity|Type)\b|\s+(?:Industry|Maturity)\b|$)", s)
    if m:
        return _tidy(m.group(1))

    # Pipe separated layouts: pick the segment that looks like a company
    if "|" in s:
        parts = [p.strip() for p in s.split("|") if p.strip()]
        for p in parts:
            if SUFFIX_RE.search(p) and not TRANCHE_RE.match(p):
                return _tidy(p)
        if len(parts) >= 3:
            return _tidy(parts[2])

    # Cut away leading percentages like '... - 186.10% Pluralsight, Inc ...'
    first_suffix = SUFFIX_RE.search(s)
    head_end = first_suffix.start() if first_suffix else len(s)
    pct = list(re.finditer(r"-?\d[\d,]*\.?\d*%\s*", s[:head_end]))
    if pct:
        s = s[pct[-1].end():]

    for pat in PREFIX_PATTERNS:
        s = re.sub(pat, "", s, flags=re.IGNORECASE)

    # Name ends at the first legal suffix, if there is one near the start
    m = SUFFIX_RE.search(s)
    if m and m.start() < 90:
        name = s[: m.end()]
        # Keep a following '(dba ...)' out, but keep 'Holdings, LLC' style names whole
        return _tidy(name)

    # Otherwise cut at the first tranche word or separator
    cut = len(s)
    t = TRANCHE_RE.search(s)
    if t and t.start() > 2:
        cut = min(cut, t.start())
    for sep in (" - ", ",", "(", " Due ", " Industry "):
        i = s.find(sep)
        if 2 < i < cut:
            cut = i
    return _tidy(s[:cut])


CATEGORY_WORDS_RE = re.compile(
    r"\b(debt|loans?|securities|investments?|secured|lien|portfolio|senior|"
    r"non-?controlled|affiliated|united states|equity)\b",
    re.IGNORECASE,
)


def strip_category_prefixes(names: pd.Series, min_followers: int = 10) -> pd.Series:
    """Remove leading industry or category phrases, learned from the data.

    Some BDCs write 'Internet Software and Services Pluralsight, Inc.'.
    A leading phrase is treated as a category when many different names
    follow it AND it either shows up alone as a heading row or contains
    category words like 'Debt' or 'Secured'. That keeps real name starts
    like 'American' or 'Project' safe.
    """
    unique = pd.Series(names.dropna().unique())
    standalone = set(unique.str.lower().str.strip())
    followers: dict[str, set[str]] = {}
    for name in unique:
        words = name.split()
        for k in range(1, min(6, len(words))):
            head = " ".join(words[:k]).strip(" ,-").lower()
            tail = " ".join(words[k:])
            followers.setdefault(head, set()).add(tail)
    prefixes = {
        h for h, tails in followers.items()
        if len(tails) >= min_followers
        and not SUFFIX_RE.search(h)
        and (h in standalone or CATEGORY_WORDS_RE.search(h))
    }
    ordered = sorted(prefixes, key=len, reverse=True)
    logger.info(f"Learned {len(ordered)} category prefixes, e.g. {ordered[:5]}")

    def strip(name: object) -> object:
        if not isinstance(name, str):
            return name
        changed = True
        while changed:
            changed = False
            low = name.lower()
            for pre in ordered:
                if low.startswith(pre + " ") or low.startswith(pre + ","):
                    name = name[len(pre):].lstrip(" ,-")
                    changed = True
                    break
        return name

    cleaned = names.map(strip)
    # Heading rows that are only a category phrase are not borrowers
    is_heading = cleaned.str.lower().str.strip().isin(prefixes) | cleaned.str.contains(
        r"^non-?controlled|^controlled|^affiliated", case=False, na=False)
    return cleaned.mask(is_heading)


def _tidy(name: str) -> str | None:
    name = re.sub(r"\s+\d+$", "", name.strip())   # trailing tranche numbers: 'Plasma Buyer LLC 3'
    name = name.strip(" ,;:-|")
    if len(name) < 2 or not re.search(r"[A-Za-z]{2}", name) or re.match(r"^\d+/\d+", name):
        return None
    return name[:120]


# ---------------------------------------------------------------- cleaning

def clean(df: pd.DataFrame) -> pd.DataFrame:
    n0 = len(df)
    out = pd.DataFrame({
        "cik": df["cik"].astype(str).str.lstrip("0"),
        "lender_name_raw": df["name"],
        "accession_number": df["adsh"],
        "form": df["form"],
        "filing_date": pd.to_datetime(df["filed"], errors="coerce"),
        "period": pd.to_datetime(df["period"].astype(str), errors="coerce"),
        "period_end": pd.to_datetime(df["ddate"].astype(str), errors="coerce"),
        "identifier": _pick(df, "identifier"),
        "issuer_name": _pick(df, "issuer_name"),
        "fair_value": pd.to_numeric(_pick(df, "fair_value"), errors="coerce"),
        "cost": pd.to_numeric(_pick(df, "cost"), errors="coerce"),
        "principal": pd.to_numeric(_pick(df, "principal"), errors="coerce"),
        "interest_rate": pd.to_numeric(_pick(df, "interest_rate"), errors="coerce"),
        "spread": pd.to_numeric(_pick(df, "spread"), errors="coerce"),
        "pik_rate": pd.to_numeric(_pick(df, "pik_rate"), errors="coerce"),
        "maturity_date": pd.to_datetime(_pick(df, "maturity_date"), errors="coerce"),
        "pct_net_assets": pd.to_numeric(_pick(df, "pct_net_assets"), errors="coerce"),
        "industry": _pick(df, "industry_axis").fillna(_pick(df, "industry_enum")).map(_decamel_member),
        "type_label": _pick(df, "type_axis").fillna(_pick(df, "type_enum")).map(_decamel_member),
    })

    # 2. Own period only
    out = out[out["period_end"] == out["period"]]
    logger.info(f"Own period rows: {len(out):,} of {n0:,}")

    # 3. Latest filing per lender and period (handles amendments)
    latest = out.groupby(["cik", "period_end"])["filing_date"].transform("max")
    out = out[out["filing_date"] == latest]
    keep_adsh = out.groupby(["cik", "period_end"])["accession_number"].transform("max")
    out = out[out["accession_number"] == keep_adsh]

    # Need an identifier and numbers
    out = out.dropna(subset=["identifier", "fair_value", "cost"])
    out = out.drop_duplicates(["cik", "period_end", "identifier", "cost", "fair_value"])
    logger.info(f"After dedup: {len(out):,}")

    # 5. Non positions and subtotals by pattern
    bad = out["identifier"].str.contains(NON_POSITION_RE, na=False)
    out = out[~bad]

    # 7. Undrawn or written off to zero cost
    out = out[out["cost"] > 0]

    # 5b. Subtotals by size: a single line far bigger than the rest of the book
    book = out.groupby(["cik", "period_end"])["fair_value"].transform("sum")
    out = out[out["fair_value"] <= 0.25 * book]
    logger.info(f"After removing subtotals and zero cost: {len(out):,}")

    # 4. Lender name = most recent name for the CIK
    latest_name = (out.sort_values("filing_date").groupby("cik")["lender_name_raw"].last())
    out["bdc_name"] = out["cik"].map(latest_name)

    # 6. Borrower names
    # Prefer the tagged issuer name when a BDC provides it, else parse the identifier
    tagged = out["issuer_name"].map(_decamel_member)
    parsed = strip_category_prefixes(out["identifier"].map(parse_borrower))
    out["borrower_name_raw"] = tagged.fillna(parsed)
    out = out.dropna(subset=["borrower_name_raw"])

    # Flags and asset class
    text = out["identifier"].fillna("") + " " + out["type_label"].fillna("")
    out["pik_flag"] = (text.str.contains(PIK_RE) | (out["pik_rate"].fillna(0) > 0)).astype(int)
    out["non_accrual_flag"] = text.str.contains(NON_ACCRUAL_RE).astype(int)
    is_equity = text.str.contains(EQUITY_RE) & out["principal"].isna()
    out["asset_class"] = is_equity.map({True: "equity", False: "debt"})

    # investment_type keeps tokens the existing panel SQL looks for
    out["investment_type"] = (
        out["type_label"].fillna(out["asset_class"])
        + out["pik_flag"].map({1: " | PIK", 0: ""})
        + out["non_accrual_flag"].map({1: " | NonAccrual", 0: ""})
    )

    # Schema fields the pipeline already uses
    q = out["period_end"].dt.quarter
    out["fiscal_year"] = out["period_end"].dt.year
    out["fiscal_period"] = "Q" + q.astype(str)
    out["frame"] = "CY" + out["fiscal_year"].astype(str) + "Q" + q.astype(str) + "I"
    out["investment_id"] = [
        hashlib.md5(f"{c}|{i}".encode()).hexdigest()[:16]
        for c, i in zip(out["cik"], out["identifier"])
    ]

    cols = [
        "cik", "bdc_name", "accession_number", "form", "filing_date", "period_end",
        "fiscal_year", "fiscal_period", "frame", "investment_id", "borrower_name_raw",
        "identifier", "fair_value", "cost", "principal", "industry", "investment_type",
        "asset_class", "pik_flag", "non_accrual_flag", "interest_rate", "spread",
        "pik_rate", "maturity_date", "pct_net_assets",
    ]
    return out[cols].reset_index(drop=True)


def write_warehouse(df: pd.DataFrame, db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    con.execute("DROP TABLE IF EXISTS raw_soi_positions")
    con.register("df_view", df)
    con.execute("CREATE TABLE raw_soi_positions AS SELECT * FROM df_view")
    n = con.execute("SELECT COUNT(*) FROM raw_soi_positions").fetchone()[0]
    con.close()
    logger.info(f"Wrote {n:,} rows to raw_soi_positions in {db_path}")


def summarize(df: pd.DataFrame) -> None:
    print(f"\nPositions: {len(df):,}")
    print(f"Lenders (CIKs): {df['cik'].nunique()}")
    print(f"Distinct borrower names: {df['borrower_name_raw'].nunique():,}")
    print(f"Quarters: {sorted(df['period_end'].dt.date.unique())}")
    print(f"Debt share: {(df['asset_class'] == 'debt').mean():.1%}, "
          f"PIK flagged: {df['pik_flag'].mean():.1%}, "
          f"non accrual flagged: {df['non_accrual_flag'].mean():.1%}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--source", default=str(DEFAULT_SOURCE),
                   help="Parquet/CSV from filter_bdc.py, or a folder of *_bdc.zip files")
    p.add_argument("--db", default=str(DB_PATH))
    args = p.parse_args()

    raw = load_source(Path(args.source))
    logger.info(f"Loaded {len(raw):,} raw rows")
    clean_df = clean(raw)
    write_warehouse(clean_df, Path(args.db))
    summarize(clean_df)


if __name__ == "__main__":
    main()
