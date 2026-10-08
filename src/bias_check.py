"""Phase 3a — does suppression/absence systematically exclude disadvantaged councils?"""
import pandas as pd, numpy as np
from scipy import stats as st
pd.set_option('display.width',170)

df   = pd.read_csv('data/processed/analysis_table.csv')
seifa= pd.read_csv('data/interim/seifa_qld.csv')
pop  = pd.read_csv('data/interim/population_qld.csv')

# ---------- A. councils ABSENT from the gambling data entirely ----------
in_gamb = set(df.lga_key.unique())
seifa['in_gambling_data'] = seifa.lga_key.isin(in_gamb)

print('═'*74); print('A. COUNCILS PRESENT vs ABSENT FROM THE GAMBLING DATASET'); print('═'*74)
a = seifa.groupby('in_gambling_data')['irsd_score'].agg(['count','mean','median','min','max']).round(1)
print(a.to_string())
pres = seifa.loc[seifa.in_gambling_data,'irsd_score']
absn = seifa.loc[~seifa.in_gambling_data,'irsd_score']
t,pv = st.ttest_ind(pres, absn, equal_var=False)
u,pu = st.mannwhitneyu(pres, absn)
print(f'\nWelch t-test : t={t:.2f}, p={pv:.2e}')
print(f'Mann-Whitney : U={u:.0f}, p={pu:.2e}')
print(f'difference in mean IRSD: {pres.mean()-absn.mean():+.1f} points')

print('\nThe 23 ABSENT councils, least to most disadvantaged:')
print(seifa[~seifa.in_gambling_data].sort_values('irsd_score')[
      ['seifa_lga_name','irsd_score','irsd_decile']].to_string(index=False))

# ---------- B. among councils WITH data, who gets suppressed? ----------
print('\n'+'═'*74); print('B. AMONG COUNCILS WITH DATA — WHO GETS SUPPRESSED?'); print('═'*74)
sup = (df[df.year.between(2005,2025)]
         .groupby('lga_key')
         .agg(months=('months','sum'), sup=('months_suppressed','sum'))
         .reset_index())
sup['pct_suppressed'] = 100*sup['sup']/sup['months']
sup = sup.merge(seifa[['lga_key','seifa_lga_name','irsd_score','irsd_decile']],on='lga_key')
sup['never_usable'] = sup.pct_suppressed==100

print(f"councils fully suppressed (100%): {sup.never_usable.sum()}")
print(f"councils with ANY suppression   : {(sup.pct_suppressed>0).sum()} of {len(sup)}")
print(f"councils fully usable (0%)      : {(sup.pct_suppressed==0).sum()}")

r,pr = st.pearsonr(sup.irsd_score, sup.pct_suppressed)
rs,ps= st.spearmanr(sup.irsd_score, sup.pct_suppressed)
print(f'\ncorrelation IRSD score vs % suppressed:')
print(f'   Pearson  r={r:+.3f}  p={pr:.4f}')
print(f'   Spearman r={rs:+.3f}  p={ps:.4f}')
print('   (negative r => MORE disadvantaged councils are MORE suppressed)')

print('\nmean IRSD by suppression band:')
sup['band']=pd.cut(sup.pct_suppressed,[-0.1,0,25,75,99.9,100],
                   labels=['0% (fully usable)','1-25%','25-75%','75-99%','100% (never usable)'])
print(sup.groupby('band',observed=True).agg(
      councils=('lga_key','size'), mean_irsd=('irsd_score','mean'),
      mean_decile=('irsd_decile','mean')).round(1).to_string())

# ---------- C. the analysable sample vs all of Queensland ----------
print('\n'+'═'*74); print('C. WHO IS IN THE ANALYSABLE SAMPLE?'); print('═'*74)
usable = set(sup.loc[sup.pct_suppressed==0,'lga_key'])
seifa['analysable'] = seifa.lga_key.isin(usable)
print(seifa.groupby('analysable')['irsd_score'].agg(['count','mean','median']).round(1).to_string())

# population coverage
p25 = pop[pop.year==2025][['lga_key','adults_20plus']]
cov = seifa.merge(p25,on='lga_key',how='left')
tot = cov.adults_20plus.sum()
print(f'\nQLD adults 20+ (2025): {tot:,.0f}')
for lbl,mask in [('in analysable sample',cov.analysable),
                 ('has some gambling data',cov.lga_key.isin(in_gamb)),
                 ('absent entirely',~cov.lga_key.isin(in_gamb))]:
    n=cov.loc[mask,'adults_20plus'].sum()
    print(f'   {lbl:24s}: {n:>10,.0f} adults ({100*n/tot:5.1f}%) across {mask.sum():2d} councils')
