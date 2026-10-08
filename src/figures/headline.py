"""
Figure 1 - machine density vs disadvantage as predictors of gambling losses.

Two panels sharing a y-axis, so the contrast between the two candidate
explanations is visible directly. Mount Isa is drawn as a hollow ring and
excluded from the fits; the faint dashed line shows what including it would do.

    python src/figure_headline.py
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src import theme
from src.config import EXCLUDE_FROM_FITS, FIGURES, PROCESSED
from src.theme import AQUA, BLUE, GRID, INK, INK_SOFT, MUTED, ORANGE, SURFACE

import matplotlib.pyplot as plt

OUTCOME = "loss_per_adult"
OUTLIER = "MOUNT ISA"
OUTPUT = "headline_density_vs_disadvantage.png"


def scatter(ax, x, y, **kwargs):
    ax.scatter(x, y, s=74, c=BLUE, edgecolors=SURFACE, linewidths=2,
               zorder=3, alpha=0.92, **kwargs)


def main():
    theme.apply_defaults()
    data = pd.read_csv(PROCESSED / "pooled_2021_2025.csv").dropna(
        subset=[OUTCOME, "egms_per_1000_adults", "irsd_score"])
    fitted = data[~data.lga_raw.isin(EXCLUDE_FROM_FITS)]
    outlier = data[data.lga_raw == OUTLIER]

    fig, (left, right) = plt.subplots(
        1, 2, figsize=(11.2, 5.1), sharey=True,
        gridspec_kw={"wspace": 0.07, "left": 0.088, "right": 0.985,
                     "top": 0.785, "bottom": 0.155})

    # ---------------------------------------------------- machine density
    theme.style_axes(left)
    scatter(left, fitted.egms_per_1000_adults, fitted[OUTCOME])
    left.scatter(outlier.egms_per_1000_adults, outlier[OUTCOME], s=96,
                 facecolors="none", edgecolors=ORANGE, linewidths=2.2, zorder=4)

    without = sm.OLS(fitted[OUTCOME],
                     sm.add_constant(fitted.egms_per_1000_adults)).fit()
    with_all = sm.OLS(data[OUTCOME],
                      sm.add_constant(data.egms_per_1000_adults)).fit()
    grid = np.linspace(data.egms_per_1000_adults.min() * 0.95,
                       data.egms_per_1000_adults.max() * 1.03, 200)
    left.plot(grid, without.params.iloc[0] + without.params.iloc[1] * grid,
              color=ORANGE, lw=2.4, zorder=5)
    left.plot(grid, with_all.params.iloc[0] + with_all.params.iloc[1] * grid,
              color=ORANGE, lw=1.6, ls=(0, (4, 3)), alpha=0.55, zorder=5)

    left.set_xlabel("Poker machines per 1,000 adults", fontsize=10.5,
                    color=INK_SOFT, labelpad=8)
    left.set_ylabel("Average loss per adult per year (2025 dollars)",
                    fontsize=10.5, color=INK_SOFT, labelpad=8)
    left.set_title("Machine availability explains losses", fontsize=12.5,
                   color=INK, loc="left", pad=11, fontweight="bold")

    for offset, text, colour, size in [
            (0.95, f"R2 = {without.rsquared:.2f}", ORANGE, 11),
            (0.878, f"excluding Mount Isa  (R2 = {with_all.rsquared:.2f} with it)", MUTED, 8.6),
            (0.822, f"+${without.params.iloc[1]:.0f} lost per adult per extra machine", MUTED, 8.6)]:
        left.text(0.035, offset, text, transform=left.transAxes, fontsize=size,
                  color=colour, va="top",
                  fontweight="bold" if size > 10 else "normal")

    left.annotate("Mount Isa\nhigh-leverage outlier",
                  xy=(outlier.egms_per_1000_adults.iloc[0], outlier[OUTCOME].iloc[0]),
                  xytext=(-104, -6), textcoords="offset points", fontsize=8.4,
                  color=ORANGE, ha="left", va="center", linespacing=1.4,
                  arrowprops=dict(arrowstyle="-", color=ORANGE, lw=1,
                                  shrinkA=0, shrinkB=7))
    for name, dy in [("BRISBANE", 17), ("BUNDABERG", -21)]:
        row = data[data.lga_raw == name]
        if len(row):
            left.annotate(name.title(),
                          (row.egms_per_1000_adults.iloc[0], row[OUTCOME].iloc[0]),
                          textcoords="offset points", xytext=(0, dy), ha="center",
                          fontsize=8.4, color=INK_SOFT)

    # ------------------------------------------------------- disadvantage
    theme.style_axes(right)
    scatter(right, fitted.irsd_score, fitted[OUTCOME])
    right.scatter(outlier.irsd_score, outlier[OUTCOME], s=96, facecolors="none",
                  edgecolors=ORANGE, linewidths=2.2, zorder=4)

    disadvantage = sm.OLS(fitted[OUTCOME], sm.add_constant(fitted.irsd_score)).fit()
    grid = np.linspace(data.irsd_score.min() * 0.995,
                       data.irsd_score.max() * 1.005, 100)
    right.plot(grid, disadvantage.params.iloc[0] + disadvantage.params.iloc[1] * grid,
               color=ORANGE, lw=2.4, ls=(0, (5, 3)), zorder=5)

    right.set_xlabel("SEIFA disadvantage score   (lower = more disadvantaged)",
                     fontsize=10.5, color=INK_SOFT, labelpad=8)
    right.set_title("Disadvantage does not", fontsize=12.5, color=INK,
                    loc="left", pad=11, fontweight="bold")
    right.text(0.035, 0.95, f"R2 = {disadvantage.rsquared:.2f}",
               transform=right.transAxes, fontsize=11, color=MUTED,
               fontweight="bold", va="top")
    right.text(0.035, 0.878,
               f"p = {disadvantage.pvalues.iloc[1]:.2f} - no significant relationship",
               transform=right.transAxes, fontsize=8.6, color=MUTED, va="top")

    theme.dollars(left)
    theme.title_block(
        fig, "Where the machines are is where the money goes",
        f"Queensland councils, averaged 2021-2025  ·  n = {len(data)} councils "
        "with complete unsuppressed data", x=0.088)
    theme.source_note(fig, [
        "Source: QLD Office of Liquor and Gaming Regulation; ABS Regional Population, "
        "SEIFA 2021, CPI (Brisbane).  Losses deflated to 2025 dollars.  "
        "Fits exclude Mount Isa (Cook's D = 26.7)."], x=0.088)

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / OUTPUT, dpi=220, facecolor=SURFACE)
    print(f"Wrote {FIGURES / OUTPUT}")
    print(f"  machine density  R2 = {without.rsquared:.3f} excluding outlier "
          f"({with_all.rsquared:.3f} including), slope {without.params.iloc[1]:.1f}")
    print(f"  disadvantage     R2 = {disadvantage.rsquared:.3f}, "
          f"p = {disadvantage.pvalues.iloc[1]:.3f}")


if __name__ == "__main__":
    main()
