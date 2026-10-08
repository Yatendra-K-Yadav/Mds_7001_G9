"""Time-trend figure — have losses diverged by disadvantage over 22 years? (RQ3)"""
import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from config import drop_outliers

SURFACE='#fcfcfb'; INK='#0b0b0b'; INK2='#52514e'; MUTED='#898781'
GRID='#e1e0d9'; AXIS='#c3c2b7'
RAMP={'Most disadvantaged':'#104281','Middle':'#2a78d6','Least disadvantaged':'#86b6ef'}
plt.rcParams.update({'axes.formatter.use_mathtext':False,'font.family':'sans-serif',
    'font.sans-serif':['Helvetica Neue','Helvetica','Arial','DejaVu Sans'],
    'figure.facecolor':SURFACE,'axes.facecolor':SURFACE})

df=drop_outliers(pd.read_csv('data/processed/analysis_table.csv'))
d=df[df.complete & df.year.between(2005,2025) & df.irsd_score.notna()].copy()
cs=d.groupby('lga_key').irsd_score.first()
d['grp']=d.lga_key.map(pd.qcut(cs,3,labels=['Most disadvantaged','Middle','Least disadvantaged']))

ts=(d.groupby(['year','grp'],observed=True)['loss_per_adult'].median().unstack())
ts=ts.reindex(range(2005,2026))           # leaves 2020 as NaN -> line breaks

fig,ax=plt.subplots(figsize=(10.4,5.4))
fig.subplots_adjust(left=0.085,right=0.80,top=0.78,bottom=0.145)
ax.set_facecolor(SURFACE); ax.grid(True,color=GRID,lw=0.8,zorder=0); ax.set_axisbelow(True)
for s in ['top','right']: ax.spines[s].set_visible(False)
for s in ['left','bottom']: ax.spines[s].set_color(AXIS); ax.spines[s].set_linewidth(1)
ax.tick_params(colors=MUTED,labelsize=9.5,length=0)

for g in ['Most disadvantaged','Middle','Least disadvantaged']:
    s=ts[g]
    ax.plot(s.index,s.values,color=RAMP[g],lw=2.4,zorder=3,solid_capstyle='round')
    ax.scatter(s.index,s.values,s=26,color=RAMP[g],edgecolors=SURFACE,linewidths=1.6,zorder=4)
    last=s.dropna().index.max()
    ax.annotate(g,(last,s[last]),textcoords='offset points',xytext=(11,0),
                va='center',fontsize=10,color=RAMP[g],fontweight='bold')

# COVID gap
ax.axvspan(2019.55,2021.45,color=GRID,alpha=0.45,zorder=1)
ax.text(2020.5,ax.get_ylim()[1]*0.975,'2020\nvenues closed',ha='center',va='top',
        fontsize=8.2,color=MUTED,linespacing=1.3)

ax.annotate('lines cross, 2022',xy=(2022,842),xytext=(2016.4,955),fontsize=8.6,color=MUTED,
            ha='center',arrowprops=dict(arrowstyle='-',color=MUTED,lw=0.9,shrinkA=2,shrinkB=4))
ax.yaxis.set_major_formatter(FuncFormatter(lambda v,_: f'${v:,.0f}'))
ax.set_xlabel('Year',fontsize=10.5,color=INK2,labelpad=8)
ax.set_ylabel('Median loss per adult (2025 dollars)',fontsize=10.5,color=INK2,labelpad=8)
ax.set_xticks(range(2005,2026,5))
ax.set_xlim(2004.2,2025.9)

fig.text(0.085,0.935,'The gap has closed — and reversed',fontsize=15.5,
         color=INK,fontweight='bold',ha='left')
fig.text(0.085,0.878,'In 2005 the most disadvantaged councils lost \$222 LESS per adult than the least disadvantaged.  By 2025 they lost \$99 MORE.',
         fontsize=9.6,color=MUTED,ha='left')
fig.text(0.085,0.028,'Source: QLD Office of Liquor and Gaming Regulation; ABS Regional Population, SEIFA 2021, CPI (Brisbane).  '
         'Councils with complete unsuppressed years only (33–42 per year); Mount Isa excluded.',
         fontsize=7.6,color=MUTED,ha='left')

fig.savefig('outputs/figures/trend_by_disadvantage.png',dpi=220,facecolor=SURFACE)
print('saved -> outputs/figures/trend_by_disadvantage.png\n')
print(ts.round(0).to_string())
print('\ngap (most - least disadvantaged):')
gap=(ts['Most disadvantaged']-ts['Least disadvantaged']).dropna()
print(f'  2005: ${gap.iloc[0]:,.0f}   2015: ${gap.loc[2015]:,.0f}   2025: ${gap.iloc[-1]:,.0f}')
