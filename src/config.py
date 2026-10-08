"""Shared configuration: file paths, analysis constants and documented exclusions."""

from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "outputs" / "figures"

GAMBLING_CSV = RAW / "Gaming Machine Data by Local Government Areas" / "gaming_machine_lga.csv"
POPULATION_XLSX = RAW / "Regional population by age and sex" / "32350DS0006_2001-25.xlsx"
SEIFA_XLSX = RAW / "SEIFA" / "Local Government Area, Indexes, SEIFA 2021.xlsx"
INCOME_XLSX = (RAW / "Personal Income in Australia" /
               "Table 1 - Total income, earners and summary statistics by geography, "
               "2018-19 to 2022-23.xlsx")

# ABS Statistical Data API — Brisbane, All groups CPI, index numbers, quarterly.
# Key is MEASURE.INDEX.TSEST.REGION.FREQ = 1 (index) . 10001 (all groups)
#                                          . 10 (original) . 3 (Brisbane) . Q
CPI_API_URL = ("https://data.api.abs.gov.au/rest/data/ABS,CPI,1.1.0/"
               "1.10001.10.3.Q?startPeriod=2004-Q1&format=csv")

# ------------------------------------------------------- analysis constants
BASE_YEAR = 2025          # all dollar figures expressed in this year's money
ADULT_AGE_FROM = 20       # ABS publishes 5-year bands, so 18+ is not available
QLD_LGA_CODES = (30000, 40000)   # Queensland LGA codes fall in this range
RANDOM_SEED = 42

# ------------------------------------------------------- documented exclusions
# Mount Isa is excluded from all FITTED MODELS (not from descriptive statistics).
#
# Justification: 29.2 EGMs per 1,000 adults against a state median of 11.6 (next
# highest 19.9), and $2,694 lost per adult against a median of $909. In the
# pooled 2021-2025 model it has leverage 0.924 (threshold 3k/n = 0.250) and
# Cook's distance 26.7 (threshold 4/n = 0.111) -- roughly 240x the influence
# threshold. Retaining it manufactures a spurious quadratic term (p = 0.0003
# with it, p = 0.224 without). As a remote mining centre with high transient-
# worker incomes and few alternative venues it is structurally unlike the other
# councils. It is plotted as a flagged outlier in Figure 1.
EXCLUDE_FROM_FITS = ["MOUNT ISA"]


def drop_outliers(df, column="lga_raw"):
    """Remove the documented high-influence councils from a dataframe."""
    return df[~df[column].isin(EXCLUDE_FROM_FITS)].copy()
