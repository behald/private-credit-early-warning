"""SEC EDGAR BDC Schedule of Investments ingestion.

Pulls XBRL-tagged SOI data from SEC EDGAR for listed BDCs.
Respects SEC rate limits (max 10 req/s) and User-Agent requirements.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)

SEC_BASE = "https://data.sec.gov"
SEC_SUBMISSIONS = f"{SEC_BASE}/submissions"
SEC_COMPANYFACTS = f"{SEC_BASE}/api/xbrl/companyfacts"

KNOWN_BDCS = [
    {"name": "Ares Capital Corporation", "cik": "0001287750", "ticker": "ARCC"},
    {"name": "Blue Owl Capital Corporation", "cik": "0001544206", "ticker": "OBDC"},
    {"name": "FS KKR Capital Corp", "cik": "0001379785", "ticker": "FSK"},
    {"name": "Golub Capital BDC Inc", "cik": "0001572694", "ticker": "GBDC"},
    {"name": "Owl Rock Technology Finance Corp", "cik": "0001758583", "ticker": "ORTF"},
    {"name": "Main Street Capital Corporation", "cik": "0001396440", "ticker": "MAIN"},
    {"name": "Prospect Capital Corporation", "cik": "0001287032", "ticker": "PSEC"},
    {"name": "Hercules Capital Inc", "cik": "0001280361", "ticker": "HTGC"},
    {"name": "PennantPark Floating Rate Capital", "cik": "0001396446", "ticker": "PFLT"},
    {"name": "Gladstone Investment Corporation", "cik": "0001273931", "ticker": "GAIN"},
    {"name": "TCP Capital Corp", "cik": "0001534254", "ticker": "TCPC"},
    {"name": "New Mountain Finance Corporation", "cik": "0001497645", "ticker": "NMFC"},
    {"name": "Goldman Sachs BDC Inc", "cik": "0001572090", "ticker": "GSBD"},
    {"name": "Blackstone Secured Lending Fund", "cik": "0001655050", "ticker": "BXSL"},
    {"name": "Morgan Stanley Direct Lending Fund", "cik": "0001743340", "ticker": "MSDL"},
    {"name": "Blue Owl Capital Corporation II", "cik": "0001826470", "ticker": "OBDE"},
    {"name": "Sixth Street Specialty Lending", "cik": "0001538263", "ticker": "TPVG"},
    {"name": "Trinity Capital Inc", "cik": "0001631761", "ticker": "TRIN"},
    {"name": "Oaktree Specialty Lending Corp", "cik": "0001492901", "ticker": "OCSL"},
    {"name": "Barings BDC Inc", "cik": "0001379380", "ticker": "BBDC"},
    {"name": "Saratoga Investment Corp", "cik": "0001377936", "ticker": "SAR"},
    {"name": "Gladstone Capital Corporation", "cik": "0001273931", "ticker": "GLAD"},
    {"name": "WhiteHorse Finance Inc", "cik": "0001552198", "ticker": "WHF"},
    {"name": "Crescent Capital BDC Inc", "cik": "0001705338", "ticker": "CCAP"},
    {"name": "SLR Investment Corp", "cik": "0001409446", "ticker": "SLRC"},
    {"name": "Carlyle Secured Lending Inc", "cik": "0001655888", "ticker": "CGBD"},
    {"name": "BlackRock TCP Capital Corp", "cik": "0001534254", "ticker": "TCPC"},
    {"name": "Runway Growth Finance Corp", "cik": "0001661181", "ticker": "RWAY"},
    {"name": "MidCap Financial Investment Corp", "cik": "0001396033", "ticker": "MFIC"},
    {"name": "Owl Rock Core Income Corp", "cik": "0001756592", "ticker": "ORCC"},
]

# SOI-related XBRL concepts
SOI_CONCEPTS = [
    "InvestmentIdentifierAxis",
    "InvestmentOwnedAtFairValue",
    "InvestmentOwnedAtCost",
    "InvestmentOwnedPercentOfNetAssets",
    "InvestmentInterestRate",
    "InvestmentMaturityDate",
]

MIN_REQUEST_INTERVAL = 0.11  # SEC: max 10 req/s


def get_user_agent() -> str:
    import os
    ua = os.getenv("SEC_USER_AGENT")
    if not ua:
        raise ValueError(
            "SEC_USER_AGENT env var required. "
            "Set it to 'Your Name your@email.com' in .env"
        )
    return ua


class EdgarClient:
    """Rate-limited SEC EDGAR API client."""

    def __init__(self, user_agent: str):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": user_agent,
            "Accept": "application/json",
        })
        self._last_request_time = 0.0

    def _throttle(self) -> None:
        elapsed = time.time() - self._last_request_time
        if elapsed < MIN_REQUEST_INTERVAL:
            time.sleep(MIN_REQUEST_INTERVAL - elapsed)
        self._last_request_time = time.time()

    def get(self, url: str) -> dict[str, Any]:
        self._throttle()
        resp = self.session.get(url)
        resp.raise_for_status()
        return resp.json()

    def get_submissions(self, cik: str) -> dict[str, Any]:
        cik_padded = cik.lstrip("0").zfill(10)
        url = f"{SEC_SUBMISSIONS}/CIK{cik_padded}.json"
        return self.get(url)

    def get_company_facts(self, cik: str) -> dict[str, Any]:
        cik_padded = cik.lstrip("0").zfill(10)
        url = f"{SEC_COMPANYFACTS}/CIK{cik_padded}.json"
        return self.get(url)


def extract_recent_filings(
    submissions: dict[str, Any],
    form_types: tuple[str, ...] = ("10-K", "10-Q"),
    max_quarters: int = 12,
) -> list[dict[str, Any]]:
    """Extract the most recent 10-K and 10-Q filings."""
    recent = submissions.get("filings", {}).get("recent", {})
    if not recent:
        return []

    forms = recent.get("form", [])
    accessions = recent.get("accessionNumber", [])
    dates = recent.get("filingDate", [])
    primary_docs = recent.get("primaryDocument", [])

    filings = []
    for i, form in enumerate(forms):
        if form in form_types and len(filings) < max_quarters:
            filings.append({
                "form": form,
                "accessionNumber": accessions[i],
                "filingDate": dates[i],
                "primaryDocument": primary_docs[i] if i < len(primary_docs) else None,
            })
    return filings


def extract_soi_from_facts(
    facts: dict[str, Any],
    cik: str,
    bdc_name: str,
) -> list[dict[str, Any]]:
    """Extract Schedule of Investments data from XBRL company facts."""
    records = []
    us_gaap = facts.get("facts", {}).get("us-gaap", {})

    fair_value_data = us_gaap.get("InvestmentOwnedAtFairValue", {})
    cost_data = us_gaap.get("InvestmentOwnedAtCost", {})

    fv_units = fair_value_data.get("units", {}).get("USD", [])
    cost_units = cost_data.get("units", {}).get("USD", [])

    cost_by_key: dict[str, float] = {}
    for entry in cost_units:
        key = f"{entry.get('accn', '')}|{entry.get('frame', '')}|{entry.get('fy', '')}|{entry.get('fp', '')}"
        dims = _extract_dimensions(entry)
        if dims.get("investment_id"):
            dim_key = f"{key}|{dims['investment_id']}"
            cost_by_key[dim_key] = entry.get("val", 0)

    for entry in fv_units:
        frame = entry.get("frame", "")
        fy = entry.get("fy")
        fp = entry.get("fp")
        accn = entry.get("accn", "")
        val = entry.get("val", 0)
        filed = entry.get("filed", "")
        end_date = entry.get("end", "")

        dims = _extract_dimensions(entry)
        investment_id = dims.get("investment_id", "")

        if not investment_id:
            continue

        key = f"{accn}|{frame}|{fy}|{fp}|{investment_id}"
        cost_val = cost_by_key.get(key, None)

        records.append({
            "cik": cik,
            "bdc_name": bdc_name,
            "accession_number": accn,
            "filing_date": filed,
            "period_end": end_date,
            "fiscal_year": fy,
            "fiscal_period": fp,
            "frame": frame,
            "investment_id": investment_id,
            "borrower_name_raw": dims.get("borrower_name", investment_id),
            "fair_value": val,
            "cost": cost_val,
            "industry": dims.get("industry"),
            "investment_type": dims.get("investment_type"),
        })

    return records


def _extract_dimensions(entry: dict[str, Any]) -> dict[str, Any]:
    """Extract XBRL dimension info from a fact entry."""
    dims: dict[str, Any] = {}
    # Dimensions come from the segment/dimension members
    # The InvestmentIdentifierAxis member label is typically the borrower
    for key, value in entry.items():
        if "dimension" in key.lower() or "member" in key.lower():
            if "InvestmentIdentifier" in str(value) or "Investment" in key:
                dims["investment_id"] = str(value)
            if "Industry" in str(value):
                dims["industry"] = str(value)
            if "InvestmentType" in str(value):
                dims["investment_type"] = str(value)
    # Often the investment_id IS the borrower name in XBRL
    if "investment_id" in dims:
        dims["borrower_name"] = dims["investment_id"].split(":")[-1].replace("Member", "")
    return dims


def ingest_bdc(
    client: EdgarClient,
    bdc: dict[str, str],
    output_dir: Path,
    max_quarters: int = 12,
) -> int:
    """Ingest SOI data for one BDC. Returns record count."""
    cik = bdc["cik"]
    name = bdc["name"]
    ticker = bdc.get("ticker", "")

    bdc_dir = output_dir / ticker.lower()
    bdc_dir.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded (idempotent)
    facts_file = bdc_dir / "company_facts.json"
    if facts_file.exists():
        logger.info(f"Already downloaded: {name} ({ticker}), loading from cache")
        facts = json.loads(facts_file.read_text())
    else:
        try:
            logger.info(f"Fetching company facts for {name} ({ticker}, CIK={cik})")
            facts = client.get_company_facts(cik)
            facts_file.write_text(json.dumps(facts, indent=2))
        except requests.HTTPError as e:
            logger.warning(f"Failed to fetch {name}: {e}")
            return 0

    # Also fetch submissions for filing metadata
    subs_file = bdc_dir / "submissions.json"
    if subs_file.exists():
        submissions = json.loads(subs_file.read_text())
    else:
        try:
            submissions = client.get_submissions(cik)
            subs_file.write_text(json.dumps(submissions, indent=2))
        except requests.HTTPError as e:
            logger.warning(f"Failed to fetch submissions for {name}: {e}")
            submissions = {}

    # Extract SOI records
    records = extract_soi_from_facts(facts, cik, name)

    if records:
        records_file = bdc_dir / "soi_records.json"
        records_file.write_text(json.dumps(records, indent=2))
        logger.info(f"Extracted {len(records)} SOI records for {name}")

    # Extract filing list
    filings = extract_recent_filings(submissions, max_quarters=max_quarters)
    if filings:
        filings_file = bdc_dir / "filings.json"
        filings_file.write_text(json.dumps(filings, indent=2))

    return len(records)


def create_sample_fixture(output_dir: Path, sample_dir: Path) -> None:
    """Create a small sample fixture for tests from downloaded data."""
    sample_dir.mkdir(parents=True, exist_ok=True)

    all_records = []
    sample_bdcs = []
    ticker_dirs = sorted(output_dir.iterdir())[:3]  # First 3 BDCs

    for bdc_dir in ticker_dirs:
        if not bdc_dir.is_dir():
            continue
        records_file = bdc_dir / "soi_records.json"
        if records_file.exists():
            records = json.loads(records_file.read_text())
            # Take first 50 records per BDC
            sample = records[:50]
            all_records.extend(sample)
            sample_bdcs.append(bdc_dir.name)

    if all_records:
        sample_file = sample_dir / "soi_sample.json"
        sample_file.write_text(json.dumps(all_records, indent=2))
        meta = {
            "source": "SEC EDGAR XBRL company facts",
            "bdcs_included": sample_bdcs,
            "record_count": len(all_records),
            "note": "Sampled subset for CI/testing. Not synthetic.",
        }
        (sample_dir / "sample_meta.json").write_text(json.dumps(meta, indent=2))
        logger.info(f"Created sample fixture with {len(all_records)} records from {sample_bdcs}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest BDC SOI data from SEC EDGAR")
    parser.add_argument("--sample", action="store_true", help="Only ingest 3 BDCs for sample")
    parser.add_argument("--max-quarters", type=int, default=12)
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--sample-dir", type=Path, default=Path("data/sample"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    user_agent = get_user_agent()
    client = EdgarClient(user_agent)

    bdcs = KNOWN_BDCS[:3] if args.sample else KNOWN_BDCS
    total_records = 0

    for bdc in bdcs:
        count = ingest_bdc(client, bdc, args.output_dir, args.max_quarters)
        total_records += count

    logger.info(f"Total: {total_records} SOI records from {len(bdcs)} BDCs")

    create_sample_fixture(args.output_dir, args.sample_dir)


if __name__ == "__main__":
    main()
