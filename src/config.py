"""Shared analysis configuration — documented exclusions."""

# Mount Isa is excluded from all fitted models.
# Justification: 29.2 EGMs per 1,000 adults vs a state median of 11.6 (next
# highest 19.9), and $2,694 lost per adult vs a median of $909. In the pooled
# 2021-2025 model it has leverage 0.924 (threshold 3k/n = 0.250) and Cook's
# distance 26.7 (threshold 4/n = 0.111) — roughly 240x the influence threshold.
# Retaining it manufactures a spurious quadratic term (p = 0.0003 with, p = 0.324
# without). As a remote mining centre with high transient-worker incomes and few
# alternative venues, it is structurally unlike the other councils.
# It is retained in all descriptive statistics and plotted as a flagged outlier.
EXCLUDE_FROM_FITS = ['MOUNT ISA']

def drop_outliers(df, col='lga_raw'):
    return df[~df[col].isin(EXCLUDE_FROM_FITS)].copy()
