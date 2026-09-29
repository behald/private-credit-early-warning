"""Tests for real SEC BDC ingestion helpers."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ingest_sec_bdc import parse_borrower, strip_category_prefixes  # noqa: E402
from newly_distressed import manager_of  # noqa: E402


def test_parse_simple_layout():
    assert parse_borrower("Pluralsight, LLC, First Lien Term Loan") == "Pluralsight, LLC"


def test_parse_percentage_layout():
    s = ("Investment Debt Investment - 208.88% United States 198.50% 1st Lien/Senior "
         "Secured Debt - 186.10% Pluralsight, Inc Industry Professional Services")
    assert parse_borrower(s) == "Pluralsight, Inc"


def test_parse_pipe_layout():
    s = "Debt Investments | Adelaide Borrower, LLC |First Lien Senior Secured Revolving Loan"
    assert parse_borrower(s) == "Adelaide Borrower, LLC"


def test_parse_labelled_layout():
    s = ("Non-Controlled/Non-Affiliated Investments, Senior Secured First Lien Loans, "
         "Company JTM Foods, LLC, Industry Food Products")
    assert parse_borrower(s) == "JTM Foods, LLC"


def test_learned_prefix_is_stripped():
    names = [f"Internet Software and Services Company{i}, Inc." for i in range(12)]
    names += ["Internet Software and Services"]
    out = strip_category_prefixes(pd.Series(names))
    assert out.iloc[0] == "Company0, Inc."
    assert pd.isna(out.iloc[-1])  # heading row is not a borrower


def test_manager_grouping():
    assert manager_of("OWL ROCK CAPITAL CORP") == manager_of("BLUE OWL CREDIT INCOME CORP.")
    assert manager_of("NORTH HAVEN PRIVATE INCOME FUND LLC") == "morgan_stanley"
    assert manager_of("ARES CAPITAL CORP") == "ares"
