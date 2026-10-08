"""Phase 3b — cross-sectional OLS: does disadvantage predict losses once machine density is controlled?"""
import pandas as pd, numpy as np, statsmodels.api as sm
pd.set_option('display.width',170)

df = pd.read_csv('data/processed/analysis_table.csv')
from config import drop_outliers, EXCLUDE_FROM_FITS
df=drop_outliers(df)
print(f'[config] excluded from fits: {EXCLUDE_FROM_FITS}')

def fit(year, predictors, label):
    d = df[(df.year==year) & df.complete].dropna(subset=['loss_per_adult']+predictors)
    X = sm.add_constant(d[predictors]); y = d['loss_per_adult']
    m = sm.OLS(y, X).fit()
    print(f'\n{"─"*70}\n{label}  (year={year}, n={len(d)})\n{"─"*70}')
    print(f'R² = {m.rsquared:.3f}   Adj R² = {m.rsquared_adj:.3f}   F p-value = {m.f_pvalue:.2e}')
    res = pd.DataFrame({'coef':m.params,'std_err':m.bse,'t':m.tvalues,'p':m.pvalues})
    res['sig'] = np.where(res.p<0.001,'***',np.where(res.p<0.01,'**',np.where(res.p<0.05,'*','')))
    print(res.round(4).to_string())
    return m, d

YR = 2024
print('='*70); print(f'RQ1 / RQ2 — WHAT PREDICTS GAMBLING LOSSES PER ADULT? ({YR})'); print('='*70)

# Model 1: disadvantage alone
m1,_ = fit(YR, ['irsd_score'], 'MODEL 1 — disadvantage only')
# Model 2: machine density alone
m2,_ = fit(YR, ['egms_per_1000_adults'], 'MODEL 2 — machine density only')
# Model 3: both together  ← the key test
m3,d3 = fit(YR, ['irsd_score','egms_per_1000_adults'], 'MODEL 3 — BOTH (the RQ2 test)')

print(f'\n{"═"*70}\nMODEL COMPARISON\n{"═"*70}')
cmp = pd.DataFrame({
 'model':['1. disadvantage only','2. machine density only','3. both'],
 'R2':[m1.rsquared,m2.rsquared,m3.rsquared],
 'adj_R2':[m1.rsquared_adj,m2.rsquared_adj,m3.rsquared_adj],
 'AIC':[m1.aic,m2.aic,m3.aic]}).round(3)
print(cmp.to_string(index=False))

# standardised coefficients for fair comparison
print(f'\n{"═"*70}\nSTANDARDISED COEFFICIENTS (model 3) — comparable effect sizes\n{"═"*70}')
z = d3[['loss_per_adult','irsd_score','egms_per_1000_adults']].apply(lambda s:(s-s.mean())/s.std())
mz = sm.OLS(z['loss_per_adult'], sm.add_constant(z[['irsd_score','egms_per_1000_adults']])).fit()
print(pd.DataFrame({'std_coef':mz.params,'p':mz.pvalues}).round(4).to_string())

# stability across years
print(f'\n{"═"*70}\nSTABILITY — same model, every year 2010-2025\n{"═"*70}')
rows=[]
for y in range(2010,2026):
    d = df[(df.year==y)&df.complete].dropna(subset=['loss_per_adult','irsd_score','egms_per_1000_adults'])
    if len(d)<20: continue
    mm = sm.OLS(d['loss_per_adult'], sm.add_constant(d[['irsd_score','egms_per_1000_adults']])).fit()
    rows.append({'year':y,'n':len(d),'R2':round(mm.rsquared,3),
                 'b_irsd':round(mm.params['irsd_score'],3),'p_irsd':round(mm.pvalues['irsd_score'],4),
                 'b_egm':round(mm.params['egms_per_1000_adults'],1),'p_egm':round(mm.pvalues['egms_per_1000_adults'],6)})
s=pd.DataFrame(rows)
print(s.to_string(index=False))
print(f"\nyears where disadvantage is significant (p<0.05): {(s.p_irsd<0.05).sum()} of {len(s)}")
print(f"years where machine density is significant (p<0.05): {(s.p_egm<0.05).sum()} of {len(s)}")
