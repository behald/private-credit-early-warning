"""Tests for early warning model."""

import numpy as np
import pytest

from src.model import _precision_at_k


class TestPrecisionAtK:
    def test_perfect_ranking(self):
        y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
        y_scores = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1, 0.05, 0.04, 0.03, 0.02])
        assert _precision_at_k(y_true, y_scores, k=3) == 1.0

    def test_worst_ranking(self):
        y_true = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1])
        y_scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
        assert _precision_at_k(y_true, y_scores, k=3) == 0.0

    def test_k_larger_than_array(self):
        y_true = np.array([1, 0, 1])
        y_scores = np.array([0.9, 0.5, 0.8])
        result = _precision_at_k(y_true, y_scores, k=10)
        assert 0 <= result <= 1
