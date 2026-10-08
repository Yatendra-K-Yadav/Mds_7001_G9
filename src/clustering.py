"""Phase 3e — k-means clustering of councils by gambling profile."""
import pandas as pd, numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from config import EXCLUDE_FROM_FITS
pd.set_option('display.width',175)
RNG=42

d=pd.read_csv('data/processed/pooled_2021_2025.csv')
d=d[~d.lga_raw.isin(EXCLUDE_FROM_FITS)]
FEATS=['loss_per_adult','egms_per_1000_adults','loss_per_egm']
d=d.dropna(subset=FEATS).reset_index(drop=True)
X=StandardScaler().fit_transform(d[FEATS])
print(f'councils: {len(d)} | features: {FEATS}\n')

print('── choosing k: elbow (WCSS) + silhouette ──')
print(f'{"k":>3} {"WCSS":>9} {"drop":>8} {"silhouette":>12}')
prev=None
for k in range(2,9):
    km=KMeans(n_clusters=k,n_init=50,random_state=RNG).fit(X)
    sil=silhouette_score(X,km.labels_)
    drop='' if prev is None else f'{prev-km.inertia_:8.1f}'
    print(f'{k:>3} {km.inertia_:9.1f} {drop:>8} {sil:12.3f}')
    prev=km.inertia_

# stability check across random seeds (Report-2 style)
print('\n── stability: best-silhouette k over 100 random restarts ──')
best=[]
for seed in range(100):
    scores={k:silhouette_score(X,KMeans(n_clusters=k,n_init=10,random_state=seed).fit(X).labels_)
            for k in range(2,9)}
    best.append(max(scores,key=scores.get))
vc=pd.Series(best).value_counts().sort_index()
print({int(k):int(v) for k,v in vc.items()})
K=int(vc.idxmax()); print(f'--> most frequently optimal k = {K}')

km=KMeans(n_clusters=K,n_init=100,random_state=RNG).fit(X)
d['cluster']=km.labels_
print(f'\nfinal silhouette (k={K}): {silhouette_score(X,km.labels_):.3f}')
print('cluster sizes:', d.cluster.value_counts().sort_index().to_dict())

print('\n── cluster profiles (means) ──')
prof=d.groupby('cluster')[FEATS+['irsd_score','median_income','adults_20plus']].mean()
prof['n']=d.groupby('cluster').size()
print(prof.round(1).to_string())

# name the clusters by profile
order=prof.sort_values('loss_per_adult').index.tolist()
names={}
for rank,c in enumerate(order):
    hi_egm = prof.loc[c,'egms_per_1000_adults']>prof.egms_per_1000_adults.mean()
    hi_int = prof.loc[c,'loss_per_egm']>prof.loss_per_egm.mean()
    names[c]=f"{'High' if rank==len(order)-1 else ('Low' if rank==0 else 'Mid')} loss / " \
             f"{'many' if hi_egm else 'few'} machines / {'high' if hi_int else 'low'} per-machine"
d['cluster_name']=d.cluster.map(names)
for c,n in names.items(): print(f'  cluster {c}: {n}')

print('\n── do clusters align with disadvantage? ──')
d['seifa_tercile']=pd.qcut(d.irsd_score,3,labels=['Most disadv','Middle','Least disadv'])
ct=pd.crosstab(d.cluster_name,d.seifa_tercile)
print(ct.to_string())
from scipy.stats import chi2_contingency
chi2,p,dof,_=chi2_contingency(ct)
print(f'\nchi-square test: chi2={chi2:.2f}, dof={dof}, p={p:.3f}')
print('  -> clusters are', 'associated with' if p<0.05 else 'INDEPENDENT of', 'disadvantage')

print('\n── councils in each cluster ──')
for c in sorted(d.cluster.unique()):
    mem=sorted(d[d.cluster==c].lga_raw.str.title())
    print(f'  [{names[c]}]  n={len(mem)}')
    print('     '+', '.join(mem))

d.to_csv('data/processed/clustered_councils.csv',index=False)
print('\nsaved -> data/processed/clustered_councils.csv')
