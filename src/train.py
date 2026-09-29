"""Train and evaluate the churn classifier, then produce an executive-grade
model report: metrics JSON, ranked churn drivers, and a one-page dashboard.

Run after ``generate_data.py``. Model artefacts land in ``outputs/``; the
committed copies used by the README live in ``docs/``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "churn.csv"
OUT = ROOT / "outputs"
DOCS = ROOT / "docs"

NUM = ["tenure_months", "monthly_charges", "support_calls"]
CAT = ["contract", "internet_service"]

# ---- House style -----------------------------------------------------------
INK = "#0f172a"
GRID = "#e2e8f0"
ACCENT = "#1E90FF"
ACCENT_2 = "#00C2A8"
PALETTE = ["#1E90FF", "#00C2A8", "#F5A524", "#7C5CFF", "#F2647C"]

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": GRID,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.titlecolor": INK,
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
})

# Friendly labels for the encoded feature names.
_PRETTY = {
    "num__monthly_charges": "Monthly charges",
    "num__tenure_months": "Tenure (months)",
    "num__support_calls": "Support calls",
    "cat__contract_Month-to-month": "Contract: month-to-month",
    "cat__contract_One year": "Contract: one year",
    "cat__contract_Two year": "Contract: two year",
    "cat__internet_service_Fiber": "Internet: fiber",
    "cat__internet_service_DSL": "Internet: DSL",
    "cat__internet_service_None": "Internet: none",
}


def build_pipeline() -> Pipeline:
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
    ])
    clf = RandomForestClassifier(n_estimators=300, max_depth=12,
                                 class_weight="balanced", random_state=42)
    return Pipeline([("pre", pre), ("clf", clf)])


def evaluate(pipe: Pipeline, Xte: pd.DataFrame, yte: pd.Series) -> dict:
    """Compute the metric set that the README and dashboard both read from."""
    proba = pipe.predict_proba(Xte)[:, 1]
    pred = (proba >= 0.5).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(
        yte, pred, labels=[1], average=None)
    cm = confusion_matrix(yte, pred)
    names = pipe.named_steps["pre"].get_feature_names_out()
    imps = pipe.named_steps["clf"].feature_importances_
    drivers = sorted(zip(names, imps), key=lambda t: -t[1])
    return {
        "roc_auc": round(float(roc_auc_score(yte, proba)), 3),
        "accuracy": round(float((pred == yte).mean()), 3),
        "churn_precision": round(float(prec[0]), 3),
        "churn_recall": round(float(rec[0]), 3),
        "churn_f1": round(float(f1[0]), 3),
        "test_churn_rate": round(float(yte.mean()), 3),
        "n_train": int(len(Xte) * 3),  # 75/25 split
        "n_test": int(len(Xte)),
        "confusion_matrix": cm.tolist(),
        "top_drivers": [
            {"feature": _PRETTY.get(n, n), "importance": round(float(i), 3)}
            for n, i in drivers[:6]
        ],
        "_proba": proba,          # kept in-memory for plots, stripped before JSON
        "_pred": pred,
        "_y": yte.to_numpy(),
        "_roc": roc_curve(yte, proba),
        "_drivers": drivers[:6],
    }


def dashboard(m: dict) -> None:
    """One-page model report: ROC, confusion matrix, drivers, score separation."""
    OUT.mkdir(exist_ok=True)
    DOCS.mkdir(exist_ok=True)
    fpr, tpr, _ = m["_roc"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Telecom Churn — Model Report", fontsize=16, fontweight="bold",
                 color=INK, x=0.5, y=0.99)

    # 1) ROC curve
    ax = axes[0, 0]
    ax.plot(fpr, tpr, color=ACCENT, lw=2.4,
            label=f"RandomForest (AUC = {m['roc_auc']:.3f})")
    ax.fill_between(fpr, tpr, alpha=0.10, color=ACCENT)
    ax.plot([0, 1], [0, 1], color="#94a3b8", lw=1.2, ls="--", label="Random")
    ax.set_title("ROC Curve")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right", frameon=False, fontsize=9)

    # 2) Confusion matrix
    ax = axes[0, 1]
    cm = np.array(m["confusion_matrix"])
    ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion Matrix (threshold 0.50)")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred: stay", "Pred: churn"])
    ax.set_yticklabels(["Actual: stay", "Actual: churn"])
    ax.grid(False)
    vmax = cm.max()
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > vmax * 0.6 else INK,
                    fontsize=13, fontweight="bold")

    # 3) Top churn drivers
    ax = axes[1, 0]
    labels = [_PRETTY.get(n, n) for n, _ in m["_drivers"]][::-1]
    vals = [i for _, i in m["_drivers"]][::-1]
    ax.barh(labels, vals, color=PALETTE[0])
    ax.set_title("Top Churn Drivers (feature importance)")
    ax.set_xlabel("Importance")
    ax.grid(axis="y", visible=False)

    # 4) Score separation: predicted probability by true class
    ax = axes[1, 1]
    proba, y = m["_proba"], m["_y"]
    ax.hist(proba[y == 0], bins=30, alpha=0.75, color=ACCENT_2, label="Stayed")
    ax.hist(proba[y == 1], bins=30, alpha=0.65, color=PALETTE[4], label="Churned")
    ax.axvline(0.5, color="#94a3b8", ls="--", lw=1.2)
    ax.set_title("Predicted Churn Probability by Actual Class")
    ax.set_xlabel("Predicted probability of churn")
    ax.set_ylabel("Customers")
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.legend(frameon=False, fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    for target in (OUT / "dashboard.png", DOCS / "dashboard.png"):
        fig.savefig(target, dpi=130)
    plt.close(fig)

    # Standalone ROC (handy for slides / embedding elsewhere)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color=ACCENT, lw=2.4, label=f"AUC = {m['roc_auc']:.3f}")
    ax.fill_between(fpr, tpr, alpha=0.10, color=ACCENT)
    ax.plot([0, 1], [0, 1], color="#94a3b8", lw=1.2, ls="--")
    ax.set_title("Churn Model — ROC Curve")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    for target in (OUT / "roc_curve.png", DOCS / "roc_curve.png"):
        fig.savefig(target, dpi=120)
    plt.close(fig)


def main() -> None:
    if not DATA.exists():
        raise SystemExit("Run: python src/generate_data.py first")
    df = pd.read_csv(DATA)
    X, y = df[NUM + CAT], df["churned"]
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42)
    pipe = build_pipeline().fit(Xtr, ytr)

    proba = pipe.predict_proba(Xte)[:, 1]
    pred = (proba >= 0.5).astype(int)
    print("=== Churn Model Evaluation ===")
    print(classification_report(yte, pred, digits=3))
    print(f"ROC-AUC: {roc_auc_score(yte, proba):.3f}")

    m = evaluate(pipe, Xte, yte)
    dashboard(m)

    print("\nTop drivers of churn:")
    for d in m["top_drivers"]:
        print(f"  {d['feature']:<28} {d['importance']:.3f}")

    # Strip the in-memory plotting arrays before serialising.
    snapshot = {k: v for k, v in m.items() if not k.startswith("_")}
    OUT.mkdir(exist_ok=True); DOCS.mkdir(exist_ok=True)
    for target in (OUT / "metrics.json", DOCS / "metrics.json"):
        target.write_text(json.dumps(snapshot, indent=2))
    print(f"\nDashboard + ROC + metrics.json written to {OUT}/ and {DOCS}/")


if __name__ == "__main__":
    main()
