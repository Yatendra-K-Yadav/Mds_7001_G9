"""Headline figure — machine density vs disadvantage as predictors of gambling losses."""
import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import statsmodels.api as sm

SURFACE='#fcfcfb'; INK='#0b0b0b'; INK2='#52514e'; MUTED='#898781'
GRID='#e1e0d9'; AXIS='#c3c2b7'
BLUE='#2a78d6'; ORANGE='#eb6834'
plt.rcParams.update({'font.family':'sans-serif',
    'font.sans-serif':['Helvetica Neue','Helvetica','Arial','DejaVu Sans'],
    'figure.facecolor':SURFACE,'axes.facecolor':SURFACE})

d=pd.read_csv('data/processed/pooled_2021_2025.csv').dropna(
   subset=['loss_per_adult','egms_per_1000_adults','irsd_score'])
OUT='MOUNT ISA'
main=d[d.lga_raw!=OUT]; out=d[d.lga_raw==OUT]

fig,axes=plt.subplots(1,2,figsize=(11.2,5.1),sharey=True,
    gridspec_kw={'wspace':0.07,'left':0.088,'right':0.985,'top':0.785,'bottom':0.155})

def style(ax):
    ax.set_facecolor(SURFACE); ax.grid(True,color=GRID,lw=0.8,zorder=0); ax.set_axisbelow(True)
    for s in ['top','right']: ax.spines[s].set_visible(False)
    for s in ['left','bottom']: ax.spines[s].set_color(AXIS); ax.spines[s].set_linewidth(1)
    ax.tick_params(colors=MUTED,labelsize=9,length=0)

# ── PANEL A ──────────────────────────────────────────────────
ax=axes[0]; style(ax)
ax.scatter(main.egms_per_1000_adults,main.loss_per_adult,s=74,c=BLUE,
           edgecolors=SURFACE,linewidths=2,zorder=3,alpha=0.92)
ax.scatter(out.egms_per_1000_adults,out.loss_per_adult,s=96,facecolors='none',
           edgecolors=ORANGE,linewidths=2.2,zorder=4)

xm=main.egms_per_1000_adults.values; ym=main.loss_per_adult.values
m_ex=sm.OLS(ym,sm.add_constant(xm)).fit()
xa=d.egms_per_1000_adults.values; ya=d.loss_per_adult.values
m_all=sm.OLS(ya,sm.add_constant(xa)).fit()
xs=np.linspace(xa.min()*0.95,xa.max()*1.03,200)
ax.plot(xs,m_ex.params[0]+m_ex.params[1]*xs,color=ORANGE,lw=2.4,zorder=5)
ax.plot(xs,m_all.params[0]+m_all.params[1]*xs,color=ORANGE,lw=1.6,ls=(0,(4,3)),alpha=0.55,zorder=5)

ax.set_xlabel('Poker machines per 1,000 adults',fontsize=10.5,color=INK2,labelpad=8)
ax.set_ylabel('Average loss per adult per year (2025 dollars)',fontsize=10.5,color=INK2,labelpad=8)
ax.set_title('Machine availability explains losses',fontsize=12.5,color=INK,loc='left',pad=11,fontweight='bold')
ax.text(0.035,0.95,f'R² = {m_ex.rsquared:.2f}',transform=ax.transAxes,fontsize=11,
        color=ORANGE,fontweight='bold',va='top')
ax.text(0.035,0.878,f'excluding Mount Isa  (R² = {m_all.rsquared:.2f} with it)',
        transform=ax.transAxes,fontsize=8.6,color=MUTED,va='top')
ax.text(0.035,0.822,f'+${m_ex.params[1]:.0f} lost per adult per extra machine',
        transform=ax.transAxes,fontsize=8.6,color=MUTED,va='top')
ax.annotate('Mount Isa\nhigh-leverage outlier',
            xy=(out.egms_per_1000_adults.iloc[0],out.loss_per_adult.iloc[0]),
            xytext=(-104,-6),textcoords='offset points',fontsize=8.4,color=ORANGE,
            ha='left',va='center',linespacing=1.4,
            arrowprops=dict(arrowstyle='-',color=ORANGE,lw=1,shrinkA=0,shrinkB=7))
for nm,dy in [('BRISBANE',17),('BUNDABERG',-21)]:
    r=d[d.lga_raw==nm]
    if len(r): ax.annotate(nm.title(),(r.egms_per_1000_adults.iloc[0],r.loss_per_adult.iloc[0]),
        textcoords='offset points',xytext=(0,dy),ha='center',fontsize=8.4,color=INK2)

# ── PANEL B ──────────────────────────────────────────────────
ax=axes[1]; style(ax)
ax.scatter(main.irsd_score,main.loss_per_adult,s=74,c=BLUE,edgecolors=SURFACE,
           linewidths=2,zorder=3,alpha=0.92)
ax.scatter(out.irsd_score,out.loss_per_adult,s=96,facecolors='none',edgecolors=ORANGE,
           linewidths=2.2,zorder=4)
x2=main.irsd_score.values
m2=sm.OLS(ym,sm.add_constant(x2)).fit()
xs2=np.linspace(d.irsd_score.min()*0.995,d.irsd_score.max()*1.005,100)
ax.plot(xs2,m2.params[0]+m2.params[1]*xs2,color=ORANGE,lw=2.4,ls=(0,(5,3)),zorder=5)
ax.set_xlabel('SEIFA disadvantage score   (lower = more disadvantaged)',fontsize=10.5,color=INK2,labelpad=8)
ax.set_title('Disadvantage does not',fontsize=12.5,color=INK,loc='left',pad=11,fontweight='bold')
ax.text(0.035,0.95,f'R² = {m2.rsquared:.2f}',transform=ax.transAxes,fontsize=11,
        color=MUTED,fontweight='bold',va='top')
ax.text(0.035,0.878,f'p = {m2.pvalues[1]:.2f} — no significant relationship',
        transform=ax.transAxes,fontsize=8.6,color=MUTED,va='top')

axes[0].yaxis.set_major_formatter(FuncFormatter(lambda v,_: f'${v:,.0f}'))

fig.text(0.088,0.945,'Where the machines are is where the money goes',
         fontsize=15.5,color=INK,fontweight='bold',ha='left')
fig.text(0.088,0.893,f'Queensland councils, averaged 2021–2025  ·  n = {len(d)} councils with complete unsuppressed data',
         fontsize=9.5,color=MUTED,ha='left')
fig.text(0.088,0.030,'Source: QLD Office of Liquor and Gaming Regulation; ABS Regional Population, SEIFA 2021, CPI (Brisbane).  '
                     'Losses deflated to 2025 dollars.  Fits exclude Mount Isa (Cook\'s D = 26.7).',
         fontsize=7.6,color=MUTED,ha='left')

fig.savefig('outputs/figures/headline_density_vs_disadvantage.png',dpi=220,facecolor=SURFACE)
print('saved')
print(f'A: R2 excl={m_ex.rsquared:.3f} (slope {m_ex.params[1]:.1f}, p={m_ex.pvalues[1]:.2e}) | incl={m_all.rsquared:.3f}')
print(f'B: R2={m2.rsquared:.3f} p={m2.pvalues[1]:.3f}')
