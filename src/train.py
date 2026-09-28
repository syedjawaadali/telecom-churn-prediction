"""Train and evaluate a churn classifier."""
from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA = Path(__file__).resolve().parents[1] / "data" / "churn.csv"
NUM = ["tenure_months", "monthly_charges", "support_calls"]
CAT = ["contract", "internet_service"]


def build_pipeline() -> Pipeline:
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
    ])
    clf = RandomForestClassifier(n_estimators=300, max_depth=12,
                                 class_weight="balanced", random_state=42)
    return Pipeline([("pre", pre), ("clf", clf)])


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

    importances = pipe.named_steps["clf"].feature_importances_
    names = pipe.named_steps["pre"].get_feature_names_out()
    top = sorted(zip(names, importances), key=lambda t: -t[1])[:6]
    print("\nTop drivers of churn:")
    for name, imp in top:
        print(f"  {name:<35} {imp:.3f}")


if __name__ == "__main__":
    main()
