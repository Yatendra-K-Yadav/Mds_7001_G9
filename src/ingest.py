"""Phase 2 — parse all five sources into tidy frames and build the analysis table."""
import pandas as pd, numpy as np, openpyxl, re, unicodedata, urllib.request, io
import os

RAW='data/raw'; INT='data/interim'; OUT='data/processed'

def norm(s):
    s=str(s).strip()
    s=re.sub(r'\s*\((?:Qld|S|C|R|T|DC|M|A)\)\s*$','',s,flags=re.I)
    s=unicodedata.normalize('NFKD',s).encode('ascii','ignore').decode()
    s=re.sub(r'[^A-Za-z0-9 ]',' ',s); return re.sub(r'\s+',' ',s).strip().upper()

# ── 1. GAMBLING ───────────────────────────────────────────────
g=pd.read_csv(f'{RAW}/Gaming Machine Data by Local Government Areas/gaming_machine_lga.csv',
              encoding='utf-8-sig')
g=g.rename(columns={'LGA Region':'lga_raw','Approved Sites':'approved_sites',
                    'Operational Sites':'operational_sites','Approved EGMs':'approved_egms',
                    'Operational EGMs':'operational_egms','Metered Win':'metered_win'})
g['date']=pd.to_datetime(g['Month Year'],format='%B %Y')
g['year']=g['date'].dt.year
g['lga_key']=g['lga_raw'].map(norm)
g['suppressed']=g['metered_win'].isna()
g.to_parquet(f'{INT}/gambling_monthly.parquet') if False else g.to_csv(f'{INT}/gambling_monthly.csv',index=False)

# annual aggregation — only complete calendar years, flag incomplete LGA-years
ann=(g.groupby(['lga_key','lga_raw','year'])
       .agg(months=('date','nunique'),
            months_suppressed=('suppressed','sum'),
            metered_win=('metered_win','sum'),
            operational_egms=('operational_egms','mean'),
            operational_sites=('operational_sites','mean'))
       .reset_index())
ann['complete']=(ann['months']==12)&(ann['months_suppressed']==0)
print(f'[gambling] {len(g):,} monthly rows | {g.year.min()}-{g.year.max()}')
print(f'[gambling] LGA-years: {len(ann):,} | fully complete: {ann.complete.sum():,} ({100*ann.complete.mean():.1f}%)')

# ── 2. POPULATION (adults 20+) ────────────────────────────────
wb=openpyxl.load_workbook(f'{RAW}/Regional population by age and sex/32350DS0006_2001-25.xlsx',
                          read_only=True,data_only=True)
ws=wb['Table 3']
rows=list(ws.iter_rows(min_row=5,values_only=True))
bands=[str(c).strip() for c in rows[0][5:] if c]
hdr=rows[1]
data=[r for r in rows[2:] if r[0] and str(r[2]).strip()=='Queensland']
wb.close()
pop=pd.DataFrame(data,columns=[str(c).strip() if c else f'c{i}' for i,c in enumerate(hdr)])
popcols=list(pop.columns)
agecols=popcols[5:5+len(bands)]
pop.columns=popcols[:5]+bands+popcols[5+len(bands):]
pop=pop.rename(columns={popcols[0]:'year',popcols[3]:'abs_lga_code',popcols[4]:'abs_lga_name'})
age_bands=[b for b in bands if b!='Total persons']      # exclude the pre-computed total
adult_bands=[b for b in age_bands if re.match(r'^(\d+)',b) and int(re.match(r'^(\d+)',b).group(1))>=20]
pop['adults_20plus']=pop[adult_bands].apply(pd.to_numeric,errors='coerce').sum(axis=1)
pop['pop_total']=pd.to_numeric(pop['Total persons'],errors='coerce')
# validation: summed bands must equal the published total
_chk=pop[age_bands].apply(pd.to_numeric,errors='coerce').sum(axis=1)
assert (abs(_chk-pop['pop_total'])<2).all(), 'age bands do not sum to published total'
pop['year']=pop['year'].astype(int)
pop['lga_key']=pop['abs_lga_name'].map(norm)
pop=pop[['year','abs_lga_code','abs_lga_name','lga_key','adults_20plus','pop_total']]
print(f'[population] age bands: {bands[:3]} ... {bands[-2:]}')
print(f'[population] adult bands used: {len(adult_bands)} (from {adult_bands[0]})')
print(f'[population] QLD LGA-years: {len(pop):,} | {pop.year.min()}-{pop.year.max()}')

# ── 3. CPI (Brisbane, All groups, quarterly → annual) ─────────
url=('https://data.api.abs.gov.au/rest/data/ABS,CPI,1.1.0/1.10001.10.3.Q'
     '?startPeriod=2004-Q1&format=csv')
req=urllib.request.Request(url,headers={'Accept':'application/vnd.sdmx.data+csv'})
cpi_raw=pd.read_csv(io.BytesIO(urllib.request.urlopen(req,timeout=90).read()))
cpi_raw['year']=cpi_raw['TIME_PERIOD'].str[:4].astype(int)
cpi=cpi_raw.groupby('year')['OBS_VALUE'].mean().rename('cpi_index').reset_index()
BASE=2025
base_val=float(cpi.loc[cpi.year==BASE,'cpi_index'].iloc[0])
cpi['cpi_factor']=base_val/cpi['cpi_index']        # multiply nominal $ to get BASE dollars
cpi.to_csv(f'{INT}/cpi_brisbane_annual.csv',index=False)
print(f'[cpi] Brisbane annual {cpi.year.min()}-{cpi.year.max()} | base={BASE} (index {base_val:.1f})')

# ── 4. SEIFA (IRSD, LGA, 2021) ───────────────────────────────
wb=openpyxl.load_workbook(f'{RAW}/SEIFA/Local Government Area, Indexes, SEIFA 2021.xlsx',
                          read_only=True,data_only=True)
ws=wb['Table 1']
sr=[]
for r in ws.iter_rows(min_row=7,max_col=4,values_only=True):
    try: code=int(r[0])
    except (TypeError,ValueError): continue
    if 30000<=code<40000 and r[2] is not None:
        sr.append({'abs_lga_code':code,'seifa_lga_name':str(r[1]).strip(),
                   'irsd_score':float(r[2]),'irsd_decile':int(r[3])})
wb.close()
seifa=pd.DataFrame(sr); seifa['lga_key']=seifa['seifa_lga_name'].map(norm)
print(f'[seifa] QLD LGAs: {len(seifa)} | IRSD score {seifa.irsd_score.min():.0f}-{seifa.irsd_score.max():.0f}')

# ── 5. INCOME (LGA, Table 1.5) ───────────────────────────────
wb=openpyxl.load_workbook(f'{RAW}/Personal Income in Australia/Table 1 - Total income, earners and summary statistics by geography, 2018-19 to 2022-23.xlsx',
                          read_only=True,data_only=True)
ws=wb['Table 1.5']
raw=list(ws.iter_rows(min_row=6,values_only=True)); wb.close()
grp=[str(c).strip() if c else '' for c in raw[0]]
yrs=[str(c).strip() if c else '' for c in raw[1]]
cols=[]; cur=''
for gcell,ycell in zip(grp,yrs):
    if gcell: cur=gcell
    cols.append(f'{cur}|{ycell}' if ycell not in ('','None') else (gcell or ycell))
inc=pd.DataFrame(raw[2:],columns=cols)
inc=inc.rename(columns={cols[0]:'abs_lga_code',cols[1]:'abs_lga_name'})
med=[c for c in cols if c.startswith('Median ($)|')]
inc_q=inc[pd.to_numeric(inc['abs_lga_code'],errors='coerce').between(30000,39999)].copy()
inc_q['lga_key']=inc_q['abs_lga_name'].map(norm)
inc_long=inc_q.melt(id_vars=['lga_key','abs_lga_name'],value_vars=med,
                    var_name='col',value_name='median_income')
inc_long['fy']=inc_long['col'].str.split('|').str[1]
inc_long['year']=inc_long['fy'].str[:4].astype(int)+1     # 2018-19 -> 2019
inc_long['median_income']=pd.to_numeric(inc_long['median_income'],errors='coerce')
inc_long=inc_long[['lga_key','year','median_income']]
inc_long.to_csv(f'{INT}/income_qld.csv',index=False)
print(f'[income] QLD LGAs: {len(inc_q)} | median cols: {len(med)} | long rows: {len(inc_long)} | years {sorted(inc_long.year.unique())}')

# ── 6. JOIN ───────────────────────────────────────────────────
df=(ann.merge(pop,on=['lga_key','year'],how='left')
       .merge(cpi,on='year',how='left')
       .merge(seifa[['lga_key','abs_lga_code','irsd_score','irsd_decile']],
              on='lga_key',how='left',suffixes=('','_seifa'))
       .merge(inc_long,on=['lga_key','year'],how='left'))
df['real_win_2025']=df['metered_win']*df['cpi_factor']
df['loss_per_adult']=df['real_win_2025']/df['adults_20plus']
df['loss_per_egm']=df['real_win_2025']/df['operational_egms']
df['egms_per_1000_adults']=1000*df['operational_egms']/df['adults_20plus']
df['irsd_quintile']=pd.qcut(df['irsd_score'],5,labels=['Q1 most disadv','Q2','Q3','Q4','Q5 least disadv'])

os.makedirs(OUT,exist_ok=True)
df.to_csv(f'{OUT}/analysis_table.csv',index=False)
seifa.to_csv(f'{INT}/seifa_qld.csv',index=False)
pop.to_csv(f'{INT}/population_qld.csv',index=False)

print('\n[join] analysis table:',df.shape)
print('[join] unmatched population:',df.adults_20plus.isna().sum(),'| unmatched seifa:',df.irsd_score.isna().sum())
print('\nSAMPLE (complete years, 2023):')
print(df[(df.year==2023)&df.complete][['lga_raw','metered_win','real_win_2025','adults_20plus',
      'loss_per_adult','irsd_decile']].head(8).to_string(index=False))
