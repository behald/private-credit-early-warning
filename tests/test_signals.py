"""Tests for signal computation."""

import pytest

from src.signals import compute_mark


class TestComputeMark:
    def test_normal_mark(self):
        assert compute_mark(950000, 1000000) == 0.95

    def test_par_mark(self):
        assert compute_mark(1000000, 1000000) == 1.0

    def test_zero_cost(self):
        assert compute_mark(100, 0) is None

    def test_none_cost(self):
        assert compute_mark(100, None) is None

    def test_above_par(self):
        mark = compute_mark(1050000, 1000000)
        assert mark == 1.05

    def test_distressed(self):
        mark = compute_mark(500000, 1000000)
        assert mark == 0.5
