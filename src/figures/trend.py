"""
Figure 2 - have losses diverged by disadvantage over 22 years?

Councils are sorted once into disadvantage terciles using SEIFA 2021, then
tracked over time. SEIFA is a single snapshot, so it labels councils rather
than varying year to year.

    python src/figure_trend.py
"""

import pandas as pd

from src import theme
from src.config import FIGURES, PROCESSED, drop_outliers
from src.theme import GRID, INK_SOFT, MUTED, ORDINAL_3

import matplotlib.pyplot as plt

OUTCOME = "loss_per_adult"
YEARS = (2005, 2025)
GROUPS = ["Most disadvantaged", "Middle", "Least disadvantaged"]
COLOURS = dict(zip(GROUPS, reversed(ORDINAL_3)))   # most disadvantaged = darkest
OUTPUT = "trend_by_disadvantage.png"


def main():
    theme.apply_defaults()
    df = drop_outliers(pd.read_csv(PROCESSED / "analysis_table.csv"))
    data = df[df.complete & df.year.between(*YEARS) & df.irsd_score.notna()].copy()

    council_scores = data.groupby("lga_key").irsd_score.first()
    data["group"] = data.lga_key.map(pd.qcut(council_scores, 3, labels=GROUPS))

    # Reindexing over every year leaves 2020 as NaN, which breaks the line
    # rather than drawing a false straight segment across the closure.
    series = (data.groupby(["year", "group"], observed=True)[OUTCOME]
              .median().unstack().reindex(range(YEARS[0], YEARS[1] + 1)))

    fig, ax = plt.subplots(figsize=(10.4, 5.4))
    fig.subplots_adjust(left=0.085, right=0.80, top=0.78, bottom=0.145)
    theme.style_axes(ax)

    for group in GROUPS:
        line = series[group]
        ax.plot(line.index, line.values, color=COLOURS[group], lw=2.4,
                zorder=3, solid_capstyle="round")
        ax.scatter(line.index, line.values, s=26, color=COLOURS[group],
                   edgecolors=theme.SURFACE, linewidths=1.6, zorder=4)
        last_year = line.dropna().index.max()
        ax.annotate(group, (last_year, line[last_year]),
                    textcoords="offset points", xytext=(11, 0), va="center",
                    fontsize=10, color=COLOURS[group], fontweight="bold")

    ax.axvspan(2019.55, 2021.45, color=GRID, alpha=0.45, zorder=1)
    ax.text(2020.5, ax.get_ylim()[1] * 0.975, "2020\nvenues closed",
            ha="center", va="top", fontsize=8.2, color=MUTED, linespacing=1.3)
    ax.annotate("lines cross, 2022", xy=(2022, 842), xytext=(2016.4, 955),
                fontsize=8.6, color=MUTED, ha="center",
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.9,
                                shrinkA=2, shrinkB=4))

    theme.dollars(ax)
    ax.set_xlabel("Year", fontsize=10.5, color=INK_SOFT, labelpad=8)
    ax.set_ylabel("Median loss per adult (2025 dollars)", fontsize=10.5,
                  color=INK_SOFT, labelpad=8)
    ax.set_xticks(range(YEARS[0], YEARS[1] + 1, 5))
    ax.set_xlim(YEARS[0] - 0.8, YEARS[1] + 0.9)

    gap = (series["Most disadvantaged"] - series["Least disadvantaged"]).dropna()
    theme.title_block(
        fig, "The gap has closed - and reversed",
        f"In {YEARS[0]} the most disadvantaged councils lost \\${abs(gap.iloc[0]):,.0f} "
        f"LESS per adult than the least disadvantaged.  "
        f"By {YEARS[1]} they lost \\${abs(gap.iloc[-1]):,.0f} MORE.", x=0.085)
    theme.source_note(fig, [
        "Source: QLD Office of Liquor and Gaming Regulation; ABS Regional Population, "
        "SEIFA 2021, CPI (Brisbane).  Councils with complete unsuppressed years only "
        "(33-42 per year); Mount Isa excluded."], x=0.085)

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / OUTPUT, dpi=220, facecolor=theme.SURFACE)
    print(f"Wrote {FIGURES / OUTPUT}")
    print(f"  gap {YEARS[0]}: ${gap.iloc[0]:+,.0f}   "
          f"2015: ${gap.loc[2015]:+,.0f}   {YEARS[1]}: ${gap.iloc[-1]:+,.0f}")


if __name__ == "__main__":
    main()
