"""Generate a synthetic telecom customer churn dataset."""
import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(7)
DATA = Path(__file__).resolve().parents[1] / "data"


def generate(n: int = 8000) -> pd.DataFrame:
    tenure = RNG.integers(0, 72, n)
    monthly = RNG.uniform(20, 120, n).round(2)
    contract = RNG.choice(["Month-to-month", "One year", "Two year"],
                          n, p=[0.55, 0.25, 0.20])
    support_calls = RNG.poisson(1.2, n)
    internet = RNG.choice(["DSL", "Fiber", "None"], n, p=[0.35, 0.5, 0.15])
    # churn probability driven by real-world style signals
    logit = (-2.0
             + 0.9 * (contract == "Month-to-month")
             + 0.35 * support_calls
             - 0.03 * tenure
             + 0.012 * monthly
             + 0.4 * (internet == "Fiber"))
    prob = 1 / (1 + np.exp(-logit))
    churn = (RNG.uniform(size=n) < prob).astype(int)
    return pd.DataFrame({
        "tenure_months": tenure, "monthly_charges": monthly,
        "contract": contract, "support_calls": support_calls,
        "internet_service": internet, "churned": churn})


if __name__ == "__main__":
    DATA.mkdir(exist_ok=True)
    df = generate()
    out = DATA / "churn.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {len(df):,} rows ({df.churned.mean():.1%} churn) -> {out}")
