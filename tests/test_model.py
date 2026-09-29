"""Invariant tests for the churn dataset and the model pipeline.

These guard the contracts the README relies on: a clean binary target, a
pipeline that emits calibrated-shape probabilities, and evaluation metrics
that stay inside their mathematical bounds and beat a random baseline.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import generate_data  # noqa: E402
import train  # noqa: E402

NUM = train.NUM
CAT = train.CAT


@pytest.fixture(scope="module")
def data():
    return generate_data.generate(n=4000)


@pytest.fixture(scope="module")
def fitted(data):
    X, y = data[NUM + CAT], data["churned"]
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42)
    pipe = train.build_pipeline().fit(Xtr, ytr)
    return pipe, Xte, yte


def test_schema(data):
    expected = {"tenure_months", "monthly_charges", "contract",
                "support_calls", "internet_service", "churned"}
    assert expected.issubset(data.columns)


def test_target_is_binary(data):
    assert set(data["churned"].unique()) <= {0, 1}


def test_churn_rate_realistic(data):
    # Signals are tuned for a plausible telecom churn base rate.
    assert 0.10 < data["churned"].mean() < 0.45


def test_value_ranges(data):
    assert (data["tenure_months"] >= 0).all()
    assert (data["monthly_charges"] > 0).all()
    assert (data["support_calls"] >= 0).all()


def test_proba_shape_and_bounds(fitted):
    pipe, Xte, _ = fitted
    proba = pipe.predict_proba(Xte)[:, 1]
    assert proba.shape == (len(Xte),)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_metrics_bounded_and_better_than_random(fitted):
    pipe, Xte, yte = fitted
    m = train.evaluate(pipe, Xte, yte)
    assert 0.5 < m["roc_auc"] <= 1.0          # beats a coin flip
    for key in ("accuracy", "churn_precision", "churn_recall", "churn_f1"):
        assert 0.0 <= m[key] <= 1.0


def test_confusion_matrix_totals(fitted):
    pipe, Xte, yte = fitted
    m = train.evaluate(pipe, Xte, yte)
    cm = np.array(m["confusion_matrix"])
    assert cm.shape == (2, 2)
    assert cm.sum() == len(yte)


def test_drivers_ranked(fitted):
    pipe, Xte, yte = fitted
    m = train.evaluate(pipe, Xte, yte)
    imps = [d["importance"] for d in m["top_drivers"]]
    assert imps == sorted(imps, reverse=True)     # descending
    assert abs(sum(pipe.named_steps["clf"].feature_importances_) - 1.0) < 1e-6
