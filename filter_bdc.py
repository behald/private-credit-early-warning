"""Shrink SEC BDC Data Set zips into one small file, keeping ALL lenders.

Reads soi.tsv from every *_bdc.zip in your Downloads folder, keeps only
the columns the radar needs, and writes one compressed Parquet file.

Usage (from the project folder):
    python filter_bdc.py
    python filter_bdc.py --input "C:/Users/you/Downloads"
"""

import argparse
import io
import re
import zipfile
from pathlib import Path

import pandas as pd

# Exact metadata columns to keep (from the SEC BDC readme)
META_COLUMNS = {"adsh", "cik", "name", "form", "period", "ddate", "filed", "qtrs", "url"}

# Keep any other column whose simplified name contains one of these words.
# Simplified = lowercase with spaces, commas and brackets removed,
# so "Investment Owned, Fair Value" becomes "investmentownedfairvalue".
KEYWORDS = [
    "identifier",      # borrower / investment name
    "industry",        # sector
    "investmenttype",  # first lien, PIK, non accrual text often lives here
    "affiliation",     # affiliated vs non affiliated
    "fairvalue",
    "cost",
    "interestrate",
    "spread",
    "maturity",
    "pik",
    "accrual",
    "principal",
    "netassets",
    "issuername",      # tagged borrower name, cleaner than parsing when present
    "nonincome",       # non income producing flag
    "acquisitiondate",
]


def simplify(col: str) -> str:
    return re.sub(r"[^a-z0-9]", "", col.lower())


def pick_columns(columns: list[str]) -> list[str]:
    keep = []
    for c in columns:
        s = simplify(c)
        if c.lower() in META_COLUMNS or any(k in s for k in KEYWORDS):
            keep.append(c)
    return keep


def read_soi(zf: zipfile.ZipFile) -> pd.DataFrame | None:
    soi_names = [n for n in zf.namelist() if n.lower().endswith("soi.tsv")]
    if not soi_names:
        print(f"  no soi.tsv found. Files inside: {zf.namelist()}")
        return None
    raw = zf.read(soi_names[0])
    for enc in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(
                io.BytesIO(raw), sep="\t", dtype=str, encoding=enc,
                on_bad_lines="warn", low_memory=False,
            )
        except UnicodeDecodeError:
            continue
    return None


def save(out: pd.DataFrame, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        path = out_dir / "soi_all_bdcs.parquet"
        out.to_parquet(path, compression="zstd", index=False)
    except ImportError:
        print("pyarrow not installed, saving compressed CSV instead "
              "(run: pip install pyarrow  for a smaller file)")
        path = out_dir / "soi_all_bdcs.csv.gz"
        out.to_csv(path, index=False, compression="gzip")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(Path.home() / "Downloads"))
    parser.add_argument("--output-dir", default="data/raw")
    args = parser.parse_args()

    zips = sorted(Path(args.input).glob("*_bdc.zip"))
    if not zips:
        print(f"No *_bdc.zip files found in {args.input}")
        return

    frames = []
    all_columns: set[str] = set()
    for zp in zips:
        print(f"Reading {zp.name}")
        with zipfile.ZipFile(zp) as zf:
            df = read_soi(zf)
        if df is None:
            continue
        all_columns.update(df.columns)
        keep = pick_columns(list(df.columns))
        print(f"  {len(df):,} rows, keeping {len(keep)} of {len(df.columns)} columns")
        slim = df[keep].copy()
        slim["source_file"] = zp.name
        frames.append(slim)

    if not frames:
        print("Nothing loaded. Check the output above.")
        return

    out = pd.concat(frames, ignore_index=True)
    out_dir = Path(args.output_dir)
    path = save(out, out_dir)

    # Column lists so the new ingest code can match names exactly,
    # and so we can see if anything useful was dropped
    kept = list(out.columns)
    dropped = sorted(all_columns - set(kept))
    (out_dir / "soi_columns.txt").write_text(
        "KEPT\n" + "\n".join(kept) + "\n\nDROPPED\n" + "\n".join(dropped)
    )

    size_mb = path.stat().st_size / 1e6
    print(f"\nSaved {len(out):,} rows to {path} ({size_mb:.1f} MB)")
    print(f"Distinct lenders: {out['name'].nunique() if 'name' in out else 'unknown'}")
    if "name" in out:
        print("\nTop 15 lenders by row count:")
        print(out["name"].value_counts().head(15).to_string())

    # Quick Pluralsight check across text columns
    text = out.drop(columns="source_file").astype(str)
    hit = text.apply(lambda col: col.str.contains("pluralsight", case=False, na=False)).any(axis=1)
    print(f"\nRows mentioning Pluralsight: {hit.sum()}")
    if hit.any() and "name" in out:
        print("Lenders holding it:")
        print(out.loc[hit, "name"].value_counts().to_string())


if __name__ == "__main__":
    main()
