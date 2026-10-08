"""
Pool several years per council to reduce year-to-year noise.

Averaging a council's complete years gives one row per council, which removes
the repeated-measures problem in the raw panel. Note that it does NOT raise the
sample size: n is bounded by the number of councils, not observations.

    python src/pooled.py
"""

import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

from src.config import PROCESSED, RANDOM_SEED, drop_outliers

PREDICTORS = ["irsd_score", "egms_per_1000_adults", "median_income", "adults"]
OUTCOME = "loss_per_adult"
WINDOWS = [(2021, 2025), (2015, 2025)]


def pool_window(df, first_year, last_year, min_years=3):
    """Average each council's complete years within a window."""
    window = df[df.year.between(first_year, last_year) & df.complete]
    pooled = (window
              .groupby(["lga_key", "lga_raw"])
              .agg(years=("year", "nunique"),
                   loss_per_adult=("loss_per_adult", "mean"),
                   egms_per_1000_adults=("egms_per_1000_adults", "mean"),
                   loss_per_egm=("loss_per_egm", "mean"),
                   irsd_score=("irsd_score", "mean"),
                   median_income=("median_income", "mean"),
                   adults=("adults", "mean"))
              .reset_index())
    return pooled[pooled.years >= min_years]


def specification_ladder(data):
    """Refit with progressively more controls to show whether IRSD is stable."""
    print("  adding controls one at a time:")
    for count in range(1, len(PREDICTORS) + 1):
        predictors = PREDICTORS[:count]
        model = sm.OLS(data[OUTCOME], sm.add_constant(data[predictors])).fit()
        print(f"    {count} predictor(s)  b_irsd = {model.params['irsd_score']:+8.3f}"
              f"  p = {model.pvalues['irsd_score']:.4f}"
              f"  R2 = {model.rsquared:.3f}  AIC = {model.aic:.1f}")


def report_full_model(data):
    model = sm.OLS(data[OUTCOME], sm.add_constant(data[PREDICTORS])).fit()
    print(f"\n  full model (n = {len(data)}, "
          f"{len(data) / len(PREDICTORS):.1f} observations per parameter)")
    table = pd.DataFrame({"coef": model.params, "std_err": model.bse,
                          "t": model.tvalues, "p": model.pvalues})
    print(table.round(4).to_string().replace("\n", "\n    ").rjust(4))

    design = sm.add_constant(data[PREDICTORS])
    vif = {c: round(variance_inflation_factor(design.values, i), 2)
           for i, c in enumerate(design.columns) if c != "const"}
    print(f"    VIF: {vif}")

    standardised = data[[OUTCOME] + PREDICTORS].apply(lambda s: (s - s.mean()) / s.std())
    betas = sm.OLS(standardised[OUTCOME],
                   sm.add_constant(standardised[PREDICTORS])).fit().params.drop("const")
    print(f"    standardised betas: {betas.round(3).to_dict()}")

    scores = cross_val_score(LinearRegression(), data[PREDICTORS].values,
                             data[OUTCOME].values, scoring="r2",
                             cv=KFold(5, shuffle=True, random_state=RANDOM_SEED))
    print(f"    5-fold CV R2: mean {scores.mean():.3f}, sd {scores.std():.3f}")

    squared = data.assign(egm_squared=data.egms_per_1000_adults ** 2)
    quadratic = sm.OLS(squared[OUTCOME],
                       sm.add_constant(squared[PREDICTORS + ["egm_squared"]])).fit()
    print(f"    quadratic EGM term: p = {quadratic.pvalues['egm_squared']:.4f} "
          f"(R2 {quadratic.rsquared:.3f} vs {model.rsquared:.3f})")


def main():
    df = pd.read_csv(PROCESSED / "analysis_table.csv")

    for first_year, last_year in WINDOWS:
        # Pool every council (the saved file is descriptive and keeps outliers),
        # but fit only on the subset config.py documents as model-eligible.
        pooled = pool_window(df, first_year, last_year)
        complete = drop_outliers(pooled).dropna(subset=[OUTCOME] + PREDICTORS)

        print("=" * 72)
        print(f"POOLED {first_year}-{last_year}   n = {len(complete)} councils")
        print("=" * 72)
        specification_ladder(complete)
        report_full_model(complete)
        print()

        pooled.to_csv(PROCESSED / f"pooled_{first_year}_{last_year}.csv", index=False)

    print(f"Wrote pooled datasets to {PROCESSED}")


if __name__ == "__main__":
    main()
