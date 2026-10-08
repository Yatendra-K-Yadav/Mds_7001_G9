"""Phase 3d — pool years per council to reduce noise and raise effective n."""
import pandas as pd, numpy as np, statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
pd.set_option('display.width',175)
RNG=42

df=pd.read_csv('data/processed/analysis_table.csv')
from config import drop_outliers, EXCLUDE_FROM_FITS
df=drop_outliers(df)
print(f'[config] excluded from fits: {EXCLUDE_FROM_FITS}')

def build(lo,hi,min_years=3):
    """average each council over a window — one row per council"""
    d=df[(df.year.between(lo,hi)) & df.complete].copy()
    g=(d.groupby(['lga_key','lga_raw'])
         .agg(years=('year','nunique'),
              loss_per_adult=('loss_per_adult','mean'),
              egms_per_1000_adults=('egms_per_1000_adults','mean'),
              loss_per_egm=('loss_per_egm','mean'),
              irsd_score=('irsd_score','mean'),
              median_income=('median_income','mean'),
              adults_20plus=('adults_20plus','mean'))
         .reset_index())
    return g[g.years>=min_years]

for lo,hi in [(2021,2025),(2015,2025)]:
    g=build(lo,hi)
    print('='*76); print(f'POOLED {lo}-{hi}   councils n={len(g)}  (mean of {g.years.min()}-{g.years.max()} complete years each)'); print('='*76)

    F=['irsd_score','egms_per_1000_adults','median_income','adults_20plus']
    gg=g.dropna(subset=['loss_per_adult']+F)
    print(f'rows with all predictors: {len(gg)}')

    print('\n-- specification ladder: does irsd stay significant? --')
    for feats in [['irsd_score'],
                  ['irsd_score','egms_per_1000_adults'],
                  ['irsd_score','egms_per_1000_adults','median_income'],
                  ['irsd_score','egms_per_1000_adults','median_income','adults_20plus']]:
        m=sm.OLS(gg.loss_per_adult, sm.add_constant(gg[feats])).fit()
        print(f'   {len(feats)} pred | b_irsd={m.params["irsd_score"]:+8.3f} p={m.pvalues["irsd_score"]:.4f} '
              f'| R2={m.rsquared:.3f} adjR2={m.rsquared_adj:.3f} AIC={m.aic:.1f}')

    # full model detail
    m=sm.OLS(gg.loss_per_adult, sm.add_constant(gg[F])).fit()
    print(f'\n-- full model (n={len(gg)}, {len(gg)/len(F):.1f} obs per parameter) --')
    r=pd.DataFrame({'coef':m.params,'se':m.bse,'t':m.tvalues,'p':m.pvalues})
    r['sig']=np.where(r.p<0.001,'***',np.where(r.p<0.01,'**',np.where(r.p<0.05,'*','')))
    print(r.round(4).to_string())
    X=sm.add_constant(gg[F])
    print('   VIF:', {c:round(vif(X.values,i),2) for i,c in enumerate(X.columns) if c!='const'})

    # standardised
    z=gg[['loss_per_adult']+F].apply(lambda s:(s-s.mean())/s.std())
    mz=sm.OLS(z.loss_per_adult, sm.add_constant(z[F])).fit()
    print('\n   standardised betas:', {k:round(v,3) for k,v in mz.params.drop('const').items()})

    # out-of-sample
    cv=cross_val_score(LinearRegression(),gg[F].values,gg.loss_per_adult.values,
                       cv=KFold(5,shuffle=True,random_state=RNG),scoring='r2')
    print(f'   5-fold CV R2: mean={cv.mean():.3f} std={cv.std():.3f}')

    # quadratic check
    q=gg.copy(); q['egm_sq']=q.egms_per_1000_adults**2
    mq=sm.OLS(q.loss_per_adult, sm.add_constant(q[F+['egm_sq']])).fit()
    print(f'   + quadratic EGM term: p={mq.pvalues["egm_sq"]:.4f}  R2={mq.rsquared:.3f} (vs {m.rsquared:.3f})')
    print()

# save pooled dataset
build(2021,2025).to_csv('data/processed/pooled_2021_2025.csv',index=False)
build(2015,2025).to_csv('data/processed/pooled_2015_2025.csv',index=False)
print('saved: data/processed/pooled_2021_2025.csv, pooled_2015_2025.csv')
