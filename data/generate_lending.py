"""Generate a realistic synthetic lending dataset for Ledger's flagship demo.

Why synthetic: it lets us bake in a KNOWN structure (real signal + a deliberate
"loss driver" + a time trend) so the eval harness can verify the agent finds the
truth, and so the leadership "what's driving our losses?" story actually has an
answer. Output: data/sample_lending.csv

Run: python -m data.generate_lending   (from the ledger/ root, venv active)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N = 8000

GRADES = ["A", "B", "C", "D", "E"]
GRADE_BASE_RISK = {"A": 0.03, "B": 0.06, "C": 0.11, "D": 0.19, "E": 0.30}
PURPOSES = ["debt_consolidation", "credit_card", "home_improvement", "major_purchase",
            "small_business", "medical"]
PURPOSE_RISK = {"debt_consolidation": 0.00, "credit_card": 0.01, "home_improvement": -0.02,
                "major_purchase": 0.00, "small_business": 0.06, "medical": 0.03}
HOME = ["RENT", "MORTGAGE", "OWN"]
REGIONS = ["West", "Midwest", "Northeast", "South"]


def generate() -> pd.DataFrame:
    # --- issue dates spread over 24 months so there is a time dimension ---
    months = pd.date_range("2023-01-01", periods=24, freq="MS")
    issue_date = RNG.choice(months, size=N)

    grade = RNG.choice(GRADES, size=N, p=[0.25, 0.30, 0.22, 0.15, 0.08])
    purpose = RNG.choice(PURPOSES, size=N, p=[0.35, 0.22, 0.15, 0.12, 0.08, 0.08])
    home = RNG.choice(HOME, size=N, p=[0.45, 0.40, 0.15])
    region = RNG.choice(REGIONS, size=N, p=[0.27, 0.23, 0.22, 0.28])

    fico = np.clip(RNG.normal(700, 45, N), 580, 850).round().astype(int)
    annual_income = np.clip(RNG.lognormal(11.0, 0.5, N), 18_000, 400_000).round(-2)
    loan_amount = np.clip(RNG.normal(15_000, 8_000, N), 1_000, 40_000).round(-2)
    term = RNG.choice([36, 60], size=N, p=[0.7, 0.3])
    emp_length = RNG.integers(0, 11, size=N)
    int_rate = np.clip(
        6 + (pd.Series(grade).map({g: i for i, g in enumerate(GRADES)}).to_numpy() * 3.2)
        + RNG.normal(0, 1.2, N), 5, 30).round(2)
    dti = np.clip(RNG.normal(18, 8, N), 0, 45).round(1)

    # --- assemble the true default probability from real drivers ---
    logit = -2.6
    logit += pd.Series(grade).map(GRADE_BASE_RISK).to_numpy() * 6.0
    logit += pd.Series(purpose).map(PURPOSE_RISK).to_numpy() * 6.0
    logit += (700 - fico) / 100.0 * 0.9          # lower FICO -> higher risk
    logit += (dti - 18) / 10.0 * 0.5             # higher DTI -> higher risk
    logit += (loan_amount / annual_income) * 1.2  # loan-to-income strain
    logit += np.where(home == "RENT", 0.25, 0.0)

    # DELIBERATE loss driver + time trend (the "story" leadership will ask about):
    # small_business loans issued in the most recent 6 months default much more.
    issue_month_idx = pd.Series(issue_date).dt.to_period("M").astype(str)
    recent = pd.Series(issue_date) >= (months[-6])
    logit += np.where((purpose == "small_business") & recent.to_numpy(), 1.1, 0.0)
    # gentle overall upward trend in risk over time
    month_rank = pd.Series(issue_date).rank(method="dense").to_numpy()
    logit += (month_rank / month_rank.max()) * 0.4

    p_default = 1 / (1 + np.exp(-logit))
    default = (RNG.uniform(0, 1, N) < p_default).astype(int)

    df = pd.DataFrame({
        "loan_id": [f"L{100000 + i}" for i in range(N)],
        "issue_date": pd.to_datetime(issue_date),
        "loan_amount": loan_amount,
        "term_months": term,
        "int_rate": int_rate,
        "grade": grade,
        "emp_length_years": emp_length,
        "home_ownership": home,
        "annual_income": annual_income,
        "dti": dti,
        "fico_score": fico,
        "purpose": purpose,
        "region": region,
        "default": default,
    })
    # inject a little realistic messiness: some missing incomes & emp_length
    miss_inc = RNG.choice(N, size=int(0.03 * N), replace=False)
    df.loc[miss_inc, "annual_income"] = np.nan
    miss_emp = RNG.choice(N, size=int(0.05 * N), replace=False)
    df.loc[miss_emp, "emp_length_years"] = np.nan
    return df.sort_values("issue_date").reset_index(drop=True)


if __name__ == "__main__":
    import pathlib
    out = pathlib.Path(__file__).parent / "sample_lending.csv"
    df = generate()
    df.to_csv(out, index=False)
    rate = df["default"].mean()
    print(f"Wrote {out}  ({len(df):,} rows, {df.shape[1]} cols)")
    print(f"Overall default rate: {rate:.1%}")
    print("Default rate by grade:")
    print(df.groupby("grade")["default"].mean().round(3).to_string())
