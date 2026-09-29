"""Tests for borrower entity resolution."""

import pytest

from src.entity_resolution import (
    normalize_name,
    resolve_entities,
    build_blocks,
)


class TestNormalizeName:
    def test_strips_legal_suffixes(self):
        assert normalize_name("Acme Corp") == "acme"
        assert normalize_name("Acme Corporation") == "acme"
        assert normalize_name("Acme LLC") == "acme"
        assert normalize_name("Acme Inc") == "acme"
        assert normalize_name("Acme Ltd") == "acme"

    def test_lowercase(self):
        assert normalize_name("ACME WIDGETS") == "acme widgets"

    def test_strips_punctuation(self):
        assert normalize_name("Acme, Inc.") == "acme"
        assert normalize_name("O'Brien Holdings LLC") == "obrien"

    def test_collapses_whitespace(self):
        assert normalize_name("Acme   Widget   Co") == "acme widget"

    def test_empty_string(self):
        assert normalize_name("") == ""

    def test_preserves_meaningful_words(self):
        result = normalize_name("Sunrise Medical Devices")
        assert "sunrise" in result
        assert "medical" in result
        assert "devices" in result


class TestBuildBlocks:
    def test_creates_blocks(self):
        names = ["Acme Corp", "Alpha LLC", "Beta Inc"]
        industries = ["tech", "tech", "finance"]
        blocks = build_blocks(names, industries)
        assert len(blocks) > 0
        # 'a' names in tech should be in same block
        tech_a_block = blocks.get("a|tech", [])
        assert len(tech_a_block) == 2

    def test_handles_none_industry(self):
        names = ["Acme"]
        industries = [None]
        blocks = build_blocks(names, industries)
        assert "a|unknown" in blocks


class TestResolveEntities:
    def test_exact_match_after_normalization(self):
        names = ["Acme Corp", "Acme Corporation", "Beta LLC"]
        industries = [None, None, None]
        result = resolve_entities(names, industries, use_embeddings=False)
        assert result["Acme Corp"] == result["Acme Corporation"]
        assert result["Acme Corp"] != result["Beta LLC"]

    def test_distinct_entities_stay_separate(self):
        names = ["Sunrise Medical", "Moonlight Financial"]
        industries = [None, None]
        result = resolve_entities(names, industries, use_embeddings=False)
        assert result["Sunrise Medical"] != result["Moonlight Financial"]

    def test_all_names_get_canonical_id(self):
        names = ["A Corp", "B Inc", "C LLC"]
        industries = [None, None, None]
        result = resolve_entities(names, industries, use_embeddings=False)
        for name in names:
            assert name in result
            assert result[name] != ""
