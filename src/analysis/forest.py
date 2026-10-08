"""
Random Forest cross-check: is the relationship really linear?

A tree ensemble can capture curves and interactions that OLS cannot. If it
fails to beat OLS out of sample, that is evidence the underlying relationship
is linear -- a reportable result rather than a disappointment.

    python src/random_forest.py
"""

import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score

from src.config import PROCESSED, RANDOM_SEED, drop_outliers

OUTCOME = "loss_per_adult"
FEATURES = ["irsd_score", "egms_per_1000_adults", "median_income", "adults"]
HEADLINE_YEAR = 2023          # income data only runs 2019-2023
CORE_FEATURES = ["irsd_score", "egms_per_1000_adults"]


def build_forest():
    return RandomForestRegressor(n_estimators=500, min_samples_leaf=2,
                                 random_state=RANDOM_SEED)


def main():
    df = drop_outliers(pd.read_csv(PROCESSED / "analysis_table.csv"))
    data = df[(df.year == HEADLINE_YEAR) & df.complete].dropna(subset=[OUTCOME] + FEATURES)
    X, y = data[FEATURES].values, data[OUTCOME].values
    folds = KFold(5, shuffle=True, random_state=RANDOM_SEED)

    print("=" * 72)
    print(f"RANDOM FOREST vs OLS   (year {HEADLINE_YEAR}, n = {len(data)})")
    print("=" * 72)

    results = {}
    for name, model in [("Linear (OLS)", LinearRegression()),
                        ("Random Forest", build_forest())]:
        results[name] = {
            "r2": cross_val_score(model, X, y, cv=folds, scoring="r2"),
            "mae": -cross_val_score(model, X, y, cv=folds,
                                    scoring="neg_mean_absolute_error"),
        }

    print("\n5-fold cross-validated performance (out of sample)")
    print(f"  {'model':16s} {'mean R2':>9s} {'sd':>7s} {'mean MAE':>11s}")
    for name, scores in results.items():
        print(f"  {name:16s} {scores['r2'].mean():9.3f} {scores['r2'].std():7.3f} "
              f"{scores['mae'].mean():11,.0f}")
    better = max(results, key=lambda k: results[k]["r2"].mean())
    print(f"  -> better out of sample: {better}")

    forest = build_forest().fit(X, y)
    permutation = permutation_importance(forest, X, y, n_repeats=50,
                                         random_state=RANDOM_SEED)
    importance = (pd.DataFrame({"feature": FEATURES,
                                "gini": forest.feature_importances_,
                                "permutation": permutation.importances_mean})
                  .sort_values("gini", ascending=False))
    print("\nFeature importance (fitted on all rows)")
    print(importance.round(4).to_string(index=False))

    z = data[[OUTCOME] + FEATURES].apply(lambda s: (s - s.mean()) / s.std())
    z_model = sm.OLS(z[OUTCOME], sm.add_constant(z[FEATURES])).fit()
    print("\nOLS standardised coefficients, same features")
    print(pd.DataFrame({"std_coef": z_model.params.drop("const"),
                        "p": z_model.pvalues.drop("const")}).round(4).to_string())

    print(f"\n{'-' * 72}")
    print("Does a non-linear term help? (quadratic in machine density)")
    squared = data.assign(egm_squared=data.egms_per_1000_adults ** 2)
    linear = sm.OLS(squared[OUTCOME],
                    sm.add_constant(squared[["egms_per_1000_adults"]])).fit()
    quadratic = sm.OLS(squared[OUTCOME],
                       sm.add_constant(squared[["egms_per_1000_adults",
                                                "egm_squared"]])).fit()
    print(f"  linear       R2 = {linear.rsquared:.3f}   AIC = {linear.aic:.1f}")
    print(f"  + quadratic  R2 = {quadratic.rsquared:.3f}   AIC = {quadratic.aic:.1f}"
          f"   (term p = {quadratic.pvalues['egm_squared']:.3f})")
    verdict = ("non-linear term helps" if quadratic.pvalues["egm_squared"] < 0.05
               else "no evidence of non-linearity")
    print(f"  -> {verdict}")

    print(f"\n{'-' * 72}")
    print("Stability of the comparison across years (two core features)")
    rows = []
    for year in range(2012, 2026):
        subset = df[(df.year == year) & df.complete].dropna(subset=[OUTCOME] + CORE_FEATURES)
        if len(subset) < 25:
            continue
        Xs, ys = subset[CORE_FEATURES].values, subset[OUTCOME].values
        forest_r2 = cross_val_score(build_forest(), Xs, ys, cv=folds, scoring="r2").mean()
        linear_r2 = cross_val_score(LinearRegression(), Xs, ys, cv=folds, scoring="r2").mean()
        rows.append({"year": year, "n": len(subset),
                     "OLS_cv_R2": round(linear_r2, 3),
                     "RF_cv_R2": round(forest_r2, 3),
                     "winner": "RF" if forest_r2 > linear_r2 else "OLS"})

    comparison = pd.DataFrame(rows)
    print(comparison.to_string(index=False))
    print(f"\nOLS wins {(comparison.winner == 'OLS').sum()} of {len(comparison)} years")


if __name__ == "__main__":
    main()
