"""
Selection bias check: which councils can the gambling data actually see?

Two distinct gaps exist. Some councils never appear because they hold no
licensed machines. Others appear but have months withheld for privacy. This
tests whether either gap is related to socio-economic disadvantage -- if it is,
any finding about disadvantage is an artefact of who is visible.

    python src/bias_check.py
"""

import pandas as pd
from scipy import stats

from src.config import INTERIM, PROCESSED

ANALYSIS_YEARS = (2005, 2025)
RECENT_YEAR = 2024


def absent_councils(analysis, seifa):
    """Compare councils that appear in the gambling data against those that never do."""
    seifa = seifa.copy()
    seifa["in_data"] = seifa.lga_key.isin(set(analysis.lga_key))

    print("=" * 74)
    print("A. COUNCILS PRESENT IN vs ABSENT FROM THE GAMBLING DATA")
    print("=" * 74)
    print(seifa.groupby("in_data")["irsd_score"]
          .agg(["count", "mean", "median", "min", "max"]).round(1).to_string())

    present = seifa.loc[seifa.in_data, "irsd_score"]
    absent = seifa.loc[~seifa.in_data, "irsd_score"]
    t_stat, t_p = stats.ttest_ind(present, absent, equal_var=False)
    u_stat, u_p = stats.mannwhitneyu(present, absent)
    print(f"\nWelch t-test   t = {t_stat:.2f}, p = {t_p:.2e}")
    print(f"Mann-Whitney   U = {u_stat:.0f}, p = {u_p:.2e}")
    print(f"mean IRSD difference: {present.mean() - absent.mean():+.1f} points")

    print("\nThe absent councils, most disadvantaged first:")
    print(seifa[~seifa.in_data].sort_values("irsd_score")
          [["seifa_lga_name", "irsd_score", "irsd_decile"]].to_string(index=False))
    return seifa


def suppression_bias(analysis, seifa):
    """Among councils that do appear, is suppression related to disadvantage?"""
    print("\n" + "=" * 74)
    print("B. AMONG COUNCILS WITH DATA - WHO GETS SUPPRESSED?")
    print("=" * 74)

    years = analysis[analysis.year.between(*ANALYSIS_YEARS)]
    suppression = (years.groupby("lga_key")
                   .agg(months=("months", "sum"), suppressed=("months_suppressed", "sum"))
                   .reset_index())
    suppression["pct_suppressed"] = 100 * suppression.suppressed / suppression.months
    suppression = suppression.merge(
        seifa[["lga_key", "irsd_score", "irsd_decile"]], on="lga_key")

    print(f"  councils with any suppression : "
          f"{(suppression.pct_suppressed > 0).sum()} of {len(suppression)}")
    print(f"  councils never usable (100%)  : "
          f"{(suppression.pct_suppressed == 100).sum()}")

    pearson_r, pearson_p = stats.pearsonr(suppression.irsd_score,
                                          suppression.pct_suppressed)
    spearman_r, spearman_p = stats.spearmanr(suppression.irsd_score,
                                             suppression.pct_suppressed)
    print(f"\n  IRSD score vs percent suppressed:")
    print(f"    Pearson  r = {pearson_r:+.3f}  p = {pearson_p:.4f}")
    print(f"    Spearman r = {spearman_r:+.3f}  p = {spearman_p:.4f}")
    print("    (a negative r would mean more disadvantaged councils are more suppressed)")

    suppression["band"] = pd.cut(
        suppression.pct_suppressed, [-0.1, 0, 25, 75, 99.9, 100],
        labels=["0%", "1-25%", "25-75%", "75-99%", "100%"])
    print("\n  mean IRSD by suppression band:")
    print(suppression.groupby("band", observed=True)
          .agg(councils=("lga_key", "size"), mean_irsd=("irsd_score", "mean"))
          .round(1).to_string())


def population_coverage(analysis, seifa, population):
    """How much of Queensland's adult population does the analysable sample cover?"""
    print("\n" + "=" * 74)
    print("C. POPULATION COVERAGE OF THE ANALYSABLE SAMPLE")
    print("=" * 74)

    for year in (2014, RECENT_YEAR):
        complete = set(analysis[(analysis.year == year) & analysis.complete].lga_key)
        any_data = set(analysis[analysis.year == year].lga_key)
        adults = population[population.year == year][["lga_key", "adults"]]
        total = adults.adults.sum()

        print(f"\n  {year}:  Queensland adults = {total:,.0f}")
        for label, councils in [("complete-year sample", complete),
                                ("any gambling data", any_data)]:
            covered = adults[adults.lga_key.isin(councils)].adults.sum()
            print(f"    {label:22s} {len(councils):2d} councils, "
                  f"{covered:>10,.0f} adults ({100 * covered / total:5.1f}%)")
        absent_adults = total - adults[adults.lga_key.isin(any_data)].adults.sum()
        print(f"    {'absent entirely':22s} {len(seifa) - len(any_data):2d} councils, "
              f"{absent_adults:>10,.0f} adults ({100 * absent_adults / total:5.1f}%)")


def main():
    analysis = pd.read_csv(PROCESSED / "analysis_table.csv")
    seifa = pd.read_csv(INTERIM / "seifa_qld.csv")
    population = pd.read_csv(INTERIM / "population_qld.csv")

    seifa = absent_councils(analysis, seifa)
    suppression_bias(analysis, seifa)
    population_coverage(analysis, seifa, population)


if __name__ == "__main__":
    main()
