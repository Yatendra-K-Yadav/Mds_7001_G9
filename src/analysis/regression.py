"""
Cross-sectional OLS: what predicts gambling losses per adult?

Each model is fitted to a SINGLE year. The panel has the same councils observed
repeatedly, so pooling every row would treat Bundaberg-2015 and Bundaberg-2016
as independent and understate the standard errors. Fitting year by year avoids
that, and repeating across years shows whether a result is stable.

    python src/regression.py
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.config import PROCESSED, drop_outliers

OUTCOME = "loss_per_adult"
HEADLINE_YEAR = 2024
STABILITY_YEARS = range(2010, 2026)
MIN_COUNCILS = 20


def fit(df, year, predictors, label):
    """Fit one year's OLS and print an annotated coefficient table."""
    data = df[(df.year == year) & df.complete].dropna(subset=[OUTCOME] + predictors)
    model = sm.OLS(data[OUTCOME], sm.add_constant(data[predictors])).fit()

    print(f"\n{'-' * 70}\n{label}   (year {year}, n = {len(data)})\n{'-' * 70}")
    print(f"R2 = {model.rsquared:.3f}   adjusted R2 = {model.rsquared_adj:.3f}"
          f"   F p-value = {model.f_pvalue:.2e}")

    table = pd.DataFrame({"coef": model.params, "std_err": model.bse,
                          "t": model.tvalues, "p": model.pvalues})
    table["sig"] = np.select([table.p < 0.001, table.p < 0.01, table.p < 0.05],
                             ["***", "**", "*"], "")
    print(table.round(4).to_string())
    return model, data


def main():
    df = drop_outliers(pd.read_csv(PROCESSED / "analysis_table.csv"))

    print("=" * 70)
    print(f"WHAT PREDICTS GAMBLING LOSSES PER ADULT?   ({HEADLINE_YEAR})")
    print("=" * 70)

    disadvantage, _ = fit(df, HEADLINE_YEAR, ["irsd_score"],
                          "MODEL 1 - disadvantage only")
    density, _ = fit(df, HEADLINE_YEAR, ["egms_per_1000_adults"],
                     "MODEL 2 - machine density only")
    both, both_data = fit(df, HEADLINE_YEAR, ["irsd_score", "egms_per_1000_adults"],
                          "MODEL 3 - both together")

    print(f"\n{'=' * 70}\nMODEL COMPARISON\n{'=' * 70}")
    print(pd.DataFrame({
        "model": ["1. disadvantage only", "2. machine density only", "3. both"],
        "R2": [m.rsquared for m in (disadvantage, density, both)],
        "adj_R2": [m.rsquared_adj for m in (disadvantage, density, both)],
        "AIC": [m.aic for m in (disadvantage, density, both)],
    }).round(3).to_string(index=False))

    print(f"\n{'=' * 70}\nSTANDARDISED COEFFICIENTS (model 3)\n{'=' * 70}")
    print("Each variable rescaled to mean 0, sd 1, so the sizes are comparable.")
    predictors = ["irsd_score", "egms_per_1000_adults"]
    z = both_data[[OUTCOME] + predictors].apply(lambda s: (s - s.mean()) / s.std())
    z_model = sm.OLS(z[OUTCOME], sm.add_constant(z[predictors])).fit()
    print(pd.DataFrame({"std_coef": z_model.params,
                        "p": z_model.pvalues}).round(4).to_string())

    print(f"\n{'=' * 70}\nSTABILITY - the same model, every year\n{'=' * 70}")
    rows = []
    for year in STABILITY_YEARS:
        data = df[(df.year == year) & df.complete].dropna(subset=[OUTCOME] + predictors)
        if len(data) < MIN_COUNCILS:
            continue
        model = sm.OLS(data[OUTCOME], sm.add_constant(data[predictors])).fit()
        rows.append({"year": year, "n": len(data),
                     "R2": round(model.rsquared, 3),
                     "b_irsd": round(model.params["irsd_score"], 3),
                     "p_irsd": round(model.pvalues["irsd_score"], 4),
                     "b_egm": round(model.params["egms_per_1000_adults"], 1),
                     "p_egm": round(model.pvalues["egms_per_1000_adults"], 6)})

    stability = pd.DataFrame(rows)
    print(stability.to_string(index=False))
    print(f"\ndisadvantage significant (p < 0.05) in "
          f"{(stability.p_irsd < 0.05).sum()} of {len(stability)} years")
    print(f"machine density significant (p < 0.05) in "
          f"{(stability.p_egm < 0.05).sum()} of {len(stability)} years")


if __name__ == "__main__":
    main()
