"""Phase 3c — Random Forest cross-check: is the relationship really linear?"""
import pandas as pd, numpy as np, statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.inspection import permutation_importance
pd.set_option('display.width',170)
RNG=42

df=pd.read_csv('data/processed/analysis_table.csv')
FEATS=['irsd_score','egms_per_1000_adults','median_income','adults_20plus']

YR=2023   # 2023 has income data (income covers 2019-2023)
d=df[(df.year==YR)&df.complete].dropna(subset=['loss_per_adult']+FEATS)
X=d[FEATS].values; y=d['loss_per_adult'].values
print('='*72); print(f'RANDOM FOREST vs OLS  (year={YR}, n={len(d)}, features={FEATS})'); print('='*72)

kf=KFold(n_splits=5,shuffle=True,random_state=RNG)
rf=RandomForestRegressor(n_estimators=500,random_state=RNG,min_samples_leaf=2)
lr=LinearRegression()

rf_cv=cross_val_score(rf,X,y,cv=kf,scoring='r2')
lr_cv=cross_val_score(lr,X,y,cv=kf,scoring='r2')
rf_mae=-cross_val_score(rf,X,y,cv=kf,scoring='neg_mean_absolute_error')
lr_mae=-cross_val_score(lr,X,y,cv=kf,scoring='neg_mean_absolute_error')

print('\n5-FOLD CROSS-VALIDATED PERFORMANCE (out-of-sample)')
print(f'  {"model":16s} {"mean R2":>9s} {"std":>7s} {"mean MAE $":>12s}')
print(f'  {"Linear (OLS)":16s} {lr_cv.mean():9.3f} {lr_cv.std():7.3f} {lr_mae.mean():12,.0f}')
print(f'  {"Random Forest":16s} {rf_cv.mean():9.3f} {rf_cv.std():7.3f} {rf_mae.mean():12,.0f}')
winner = 'Random Forest' if rf_cv.mean()>lr_cv.mean() else 'Linear (OLS)'
print(f'  --> better out-of-sample: {winner}  (gap {abs(rf_cv.mean()-lr_cv.mean()):.3f} R2)')

# feature importance
rf.fit(X,y)
imp=pd.DataFrame({'feature':FEATS,'gini_importance':rf.feature_importances_}).sort_values('gini_importance',ascending=False)
perm=permutation_importance(rf,X,y,n_repeats=50,random_state=RNG)
imp['perm_importance']=[perm.importances_mean[FEATS.index(f)] for f in imp.feature]
print('\nFEATURE IMPORTANCE (Random Forest, fitted on all rows)')
print(imp.round(4).to_string(index=False))

# OLS standardised coefficients for comparison
z=d[['loss_per_adult']+FEATS].apply(lambda s:(s-s.mean())/s.std())
mz=sm.OLS(z['loss_per_adult'],sm.add_constant(z[FEATS])).fit()
print('\nOLS STANDARDISED COEFFICIENTS (same features, for comparison)')
print(pd.DataFrame({'std_coef':mz.params.drop("const"),'p':mz.pvalues.drop("const")}).round(4).to_string())

# does RF find non-linearity in the key predictor?
print('\n'+'─'*72)
print('NON-LINEARITY CHECK — add a quadratic term to OLS')
d2=d.copy(); d2['egm_sq']=d2.egms_per_1000_adults**2
m_lin=sm.OLS(d2.loss_per_adult,sm.add_constant(d2[['egms_per_1000_adults']])).fit()
m_quad=sm.OLS(d2.loss_per_adult,sm.add_constant(d2[['egms_per_1000_adults','egm_sq']])).fit()
print(f'  linear      : R2={m_lin.rsquared:.3f}  AIC={m_lin.aic:.1f}')
print(f'  + quadratic : R2={m_quad.rsquared:.3f}  AIC={m_quad.aic:.1f}  (quad term p={m_quad.pvalues["egm_sq"]:.3f})')
print(f'  --> {"non-linear term helps" if m_quad.pvalues["egm_sq"]<0.05 else "NO evidence of non-linearity"}')

# repeat the model comparison across several years
print('\n'+'─'*72)
print('STABILITY OF THE COMPARISON (2-feature model, no income, more years)')
F2=['irsd_score','egms_per_1000_adults']; rows=[]
for yy in range(2012,2026):
    dd=df[(df.year==yy)&df.complete].dropna(subset=['loss_per_adult']+F2)
    if len(dd)<25: continue
    Xa,ya=dd[F2].values,dd['loss_per_adult'].values
    r=cross_val_score(RandomForestRegressor(n_estimators=500,random_state=RNG,min_samples_leaf=2),Xa,ya,cv=kf,scoring='r2').mean()
    l=cross_val_score(LinearRegression(),Xa,ya,cv=kf,scoring='r2').mean()
    rows.append({'year':yy,'n':len(dd),'OLS_cv_R2':round(l,3),'RF_cv_R2':round(r,3),'winner':'RF' if r>l else 'OLS'})
s=pd.DataFrame(rows); print(s.to_string(index=False))
print(f"\nOLS wins {sum(s.winner=='OLS')} of {len(s)} years | RF wins {sum(s.winner=='RF')}")
