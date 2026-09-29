"""Tests for SEC EDGAR ingestion."""

import json
from pathlib import Path

import pytest

from src.ingest import (
    EdgarClient,
    extract_recent_filings,
    extract_soi_from_facts,
    KNOWN_BDCS,
)


def test_known_bdcs_has_minimum_count():
    assert len(KNOWN_BDCS) >= 30


def test_known_bdcs_have_required_fields():
    for bdc in KNOWN_BDCS:
        assert "name" in bdc
        assert "cik" in bdc
        assert "ticker" in bdc


def test_extract_recent_filings_filters_form_type():
    submissions = {
        "filings": {
            "recent": {
                "form": ["10-K", "10-Q", "8-K", "10-Q", "DEF 14A"],
                "accessionNumber": ["a1", "a2", "a3", "a4", "a5"],
                "filingDate": ["2024-03-15", "2024-02-15", "2024-01-15", "2023-11-15", "2023-10-15"],
                "primaryDocument": ["d1.htm", "d2.htm", "d3.htm", "d4.htm", "d5.htm"],
            }
        }
    }
    filings = extract_recent_filings(submissions, max_quarters=12)
    assert len(filings) == 3
    assert all(f["form"] in ("10-K", "10-Q") for f in filings)


def test_extract_recent_filings_respects_max_quarters():
    submissions = {
        "filings": {
            "recent": {
                "form": ["10-Q"] * 20,
                "accessionNumber": [f"a{i}" for i in range(20)],
                "filingDate": [f"2024-01-{i+1:02d}" for i in range(20)],
                "primaryDocument": [f"d{i}.htm" for i in range(20)],
            }
        }
    }
    filings = extract_recent_filings(submissions, max_quarters=8)
    assert len(filings) == 8


def test_extract_recent_filings_empty():
    filings = extract_recent_filings({})
    assert filings == []


def test_extract_soi_from_facts_empty():
    facts = {"facts": {"us-gaap": {}}}
    records = extract_soi_from_facts(facts, "0001", "Test BDC")
    assert records == []


@pytest.fixture
def sample_fixture_path():
    path = Path("data/sample/soi_sample.json")
    if path.exists():
        return path
    return None


def test_sample_fixture_valid(sample_fixture_path):
    if sample_fixture_path is None:
        pytest.skip("No sample fixture yet")
    data = json.loads(sample_fixture_path.read_text())
    assert isinstance(data, list)
    assert len(data) > 0
    required_fields = ["cik", "bdc_name", "borrower_name_raw", "fair_value"]
    for record in data[:5]:
        for field in required_fields:
            assert field in record, f"Missing {field} in sample record"
