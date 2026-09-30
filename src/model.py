"""Early warning prediction model.

Predicts which positions become non-accrual or fall below a mark threshold
in the next quarter. Uses time-based splits to prevent leakage.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
)

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    "mark",
    "mark_drift",
    "pik_flag",
    "non_accrual_flag",
    "num_lenders",
    "mark_dispersion",
    "lender_mark_vs_avg",
    "pik_migration",
    "par_migration",
    "avg_mark_across_lenders",
]

TARGET_COLUMN = "target_distress_next_q"


def prepare_model_data(
    con: duckdb.DuckDBPyConnection,
    mark_threshold: float = 0.80,
) -> pd.DataFrame:
    """Prepare features and target for modeling.

    Target: position becomes non-accrual OR mark falls below threshold in next quarter.
    Only uses features from current and prior quarters (no leakage).
    """
    df = con.execute("""
        WITH with_target AS (
            SELECT s.*,
                LEAD(CASE
                    WHEN non_accrual_flag = 1 THEN 1
                    WHEN mark < $1 THEN 1
                    ELSE 0
                END) OVER (
                    PARTITION BY canonical_borrower_id, lender_cik
                    ORDER BY quarter
                ) AS target_distress_next_q
            FROM mart_signals s
        )
        SELECT * FROM with_target
        WHERE target_distress_next_q IS NOT NULL
    """, [mark_threshold]).fetchdf()

    return df


def time_based_split(
    df: pd.DataFrame,
    train_end_q: str,
    val_q: str,
    test_q: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    """Split data by quarter for time-based validation."""
    train = df[df["quarter"] <= train_end_q]
    val = df[df["quarter"] == val_q]
    test = df[df["quarter"] == test_q] if test_q else None
    return train, val, test


def train_baseline(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> dict[str, Any]:
    """Train logistic regression baseline on mark level alone."""
    X_train = train_df[["mark"]].fillna(0)
    y_train = train_df[TARGET_COLUMN]
    X_val = val_df[["mark"]].fillna(0)
    y_val = val_df[TARGET_COLUMN]

    model = LogisticRegression(class_weight="balanced", random_state=42)
    model.fit(X_train, y_train)

    val_proba = model.predict_proba(X_val)[:, 1]
    pr_auc = average_precision_score(y_val, val_proba)

    precision, recall, thresholds = precision_recall_curve(y_val, val_proba)
    p_at_50 = _precision_at_k(y_val.values, val_proba, k=50)

    return {
        "model_type": "logistic_regression_baseline",
        "features": ["mark"],
        "pr_auc": round(float(pr_auc), 4),
        "precision_at_50": round(float(p_at_50), 4),
        "val_size": len(val_df),
        "val_positive_rate": round(float(y_val.mean()), 4),
    }


def train_main_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    model_dir: Path,
) -> dict[str, Any]:
    """Train LightGBM model with all features."""
    try:
        import lightgbm as lgb
    except ImportError:
        logger.error("lightgbm not installed")
        return {"error": "lightgbm not installed"}

    features = [c for c in FEATURE_COLUMNS if c in train_df.columns]

    X_train = train_df[features].fillna(0)
    y_train = train_df[TARGET_COLUMN]
    X_val = val_df[features].fillna(0)
    y_val = val_df[TARGET_COLUMN]

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    params = {
        "objective": "binary",
        "metric": "average_precision",
        "is_unbalance": True,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": 6,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "seed": 42,
        "num_threads": 1,
    }

    callbacks = [lgb.log_evaluation(50), lgb.early_stopping(20)]

    model = lgb.train(
        params,
        train_data,
        num_boost_round=300,
        valid_sets=[val_data],
        callbacks=callbacks,
    )

    val_proba = model.predict(X_val)
    pr_auc = average_precision_score(y_val, val_proba)
    p_at_50 = _precision_at_k(y_val.values, val_proba, k=50)

    # Feature importance
    importance = dict(zip(features, model.feature_importance(importance_type="gain").tolist()))
    sorted_importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    # Save model
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_dir / "lightgbm_model.txt"))

    # Save metrics
    metrics = {
        "model_type": "lightgbm",
        "features": features,
        "pr_auc": round(float(pr_auc), 4),
        "precision_at_50": round(float(p_at_50), 4),
        "val_size": len(val_df),
        "val_positive_rate": round(float(y_val.mean()), 4),
        "feature_importance": sorted_importance,
    }
    (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    return metrics


def _precision_at_k(y_true: np.ndarray, y_scores: np.ndarray, k: int = 50) -> float:
    """Precision at top-k predictions."""
    if len(y_true) < k:
        k = len(y_true)
    top_k_indices = np.argsort(y_scores)[-k:]
    return float(y_true[top_k_indices].sum() / k)
