"""
Data quality report for the raw gambling data.

Checks temporal coverage, quantifies the regulator's privacy suppression, and
identifies councils that can never be analysed. Run after src/ingest.py.

    python src/data_quality.py
"""

import pandas as pd

from src.config import GAMBLING_CSV


def main():
    df = pd.read_csv(GAMBLING_CSV, encoding="utf-8-sig")
    df["date"] = pd.to_datetime(df["Month Year"], format="%B %Y")
    df["metered_win"] = pd.to_numeric(df["Metered Win"], errors="coerce")
    df["suppressed"] = df["metered_win"].isna()

    print("COVERAGE")
    expected = pd.date_range(df.date.min(), df.date.max(), freq="MS")
    missing = sorted(set(expected) - set(df.date.unique()))
    print(f"  rows            {len(df):,}")
    print(f"  period          {df.date.min():%B %Y} to {df.date.max():%B %Y}")
    print(f"  months present  {df.date.nunique()} of {len(expected)} expected")
    print(f"  missing months  {[m.strftime('%b %Y') for m in missing] or 'none'}")

    councils = df["LGA Region"].nunique()
    grid_gap = councils * len(expected) - len(df)
    print(f"  councils        {councils}")
    if grid_gap:
        absent = (df.pivot_table(index="LGA Region", columns="date",
                                 values="Approved Sites", aggfunc="size")
                    .isna().sum(axis=1))
        print(f"  absent rows     {grid_gap} "
              f"({absent[absent > 0].to_dict()})")

    print("\nSUPPRESSION")
    print("  The regulator withholds any council-month with fewer than six")
    print("  operating venues, so those rows have no loss figure.")
    print(f"  suppressed rows   {df.suppressed.sum():,} "
          f"({100 * df.suppressed.mean():.1f}%)")
    print(f"  max venues when suppressed      "
          f"{df.loc[df.suppressed, 'Operational Sites'].max():.0f}")
    print(f"  min venues when not suppressed  "
          f"{df.loc[~df.suppressed, 'Operational Sites'].min():.0f}")

    always = (df.groupby("LGA Region")["suppressed"].mean() == 1)
    print(f"\n  councils suppressed in every month: {always.sum()}")
    for name in sorted(always[always].index):
        print(f"    - {name.title()}")

    print("\nMISSINGNESS BY COLUMN")
    print(df[["Operational Sites", "Approved EGMs",
              "Operational EGMs", "metered_win"]].isna().sum().to_string())


if __name__ == "__main__":
    main()
