"""Cluster figure — three types of gambling council in Queensland."""
import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

SURFACE='#fcfcfb'; INK='#0b0b0b'; INK2='#52514e'; MUTED='#898781'
GRID='#e1e0d9'; AXIS='#c3c2b7'
C={0:'#2a78d6',1:'#eb6834',2:'#1baf7a'}          # validated all-pairs
plt.rcParams.update({'axes.formatter.use_mathtext':False,'font.family':'sans-serif',
    'font.sans-serif':['Helvetica Neue','Helvetica','Arial','DejaVu Sans'],
    'figure.facecolor':SURFACE,'axes.facecolor':SURFACE})

d=pd.read_csv('data/processed/clustered_councils.csv')
LABEL={0:'Metropolitan',1:'Regional centres',2:'Low exposure'}
DESC={0:'few machines, worked hard',1:'many machines, lower intensity',2:'few machines, low intensity'}

fig,ax=plt.subplots(figsize=(10.6,6.0))
fig.subplots_adjust(left=0.095,right=0.985,top=0.77,bottom=0.175)
ax.set_facecolor(SURFACE); ax.grid(True,color=GRID,lw=0.8,zorder=0); ax.set_axisbelow(True)
for s in ['top','right']: ax.spines[s].set_visible(False)
for s in ['left','bottom']: ax.spines[s].set_color(AXIS); ax.spines[s].set_linewidth(1)
ax.tick_params(colors=MUTED,labelsize=9.5,length=0)

sz=40+ (d.adults_20plus/d.adults_20plus.max())*720
for c in sorted(d.cluster.unique()):
    m=d.cluster==c
    ax.scatter(d.loc[m,'egms_per_1000_adults'],d.loc[m,'loss_per_egm'],
               s=sz[m],c=C[c],edgecolors=SURFACE,linewidths=2,alpha=0.88,zorder=3)

# direct cluster labels (also satisfies the contrast-relief rule)
POS={0:(6.4,121000,'left'),1:(15.3,118000,'left'),2:(6.4,66500,'left')}
for c,(x,y,ha) in POS.items():
    ax.text(x,y,LABEL[c],fontsize=12,color=C[c],fontweight='bold',ha=ha,va='center')
    ax.text(x,y-4600,DESC[c],fontsize=8.8,color=MUTED,ha=ha,va='center')
    ax.text(x,y-8600,f'n = {(d.cluster==c).sum()} councils',fontsize=8.4,color=MUTED,ha=ha,va='center')

ax.set_xlim(4.6,21.2); ax.set_ylim(50000,133000)
for nm,dx,dy in [('BRISBANE',-6,-26),('GOLD COAST',0,-24),('BUNDABERG',0,18),('NOOSA',14,-14)]:
    r=d[d.lga_raw==nm]
    if len(r): ax.annotate(nm.title(),(r.egms_per_1000_adults.iloc[0],r.loss_per_egm.iloc[0]),
        textcoords='offset points',xytext=(dx,dy),ha='center',fontsize=8.4,color=INK2,zorder=6)

ax.yaxis.set_major_formatter(FuncFormatter(lambda v,_: f'${v/1000:,.0f}k'))
ax.set_xlabel('Poker machines per 1,000 adults',fontsize=10.5,color=INK2,labelpad=8)
ax.set_ylabel('Lost per machine per year (2025 dollars)',fontsize=10.5,color=INK2,labelpad=8)

fig.text(0.095,0.945,'Three kinds of gambling council — and none of them is about wealth',
         fontsize=15,color=INK,fontweight='bold',ha='left')
fig.text(0.095,0.888,'k-means clusters of 35 Queensland councils, averaged 2021–2025.  Bubble size = adult population.',
         fontsize=9.6,color=MUTED,ha='left')
fig.text(0.095,0.832,'Clusters are statistically independent of socio-economic disadvantage  (chi-square p = 0.94)',
         fontsize=9.6,color=INK2,ha='left',style='italic')
fig.text(0.095,0.052,'Source: QLD Office of Liquor and Gaming Regulation; ABS Regional Population, SEIFA 2021, CPI (Brisbane).',
         fontsize=7.5,color=MUTED,ha='left')
fig.text(0.095,0.022,'k = 3 selected by silhouette score (optimal in 99 of 100 restarts); features standardised.  Mount Isa excluded.',
         fontsize=7.5,color=MUTED,ha='left')

fig.savefig('outputs/figures/clusters.png',dpi=220,facecolor=SURFACE)
print('saved -> outputs/figures/clusters.png')
