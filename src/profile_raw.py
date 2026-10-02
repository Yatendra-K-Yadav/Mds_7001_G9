import pandas as pd, numpy as np
pd.set_option('display.width', 160)

df = pd.read_csv('data/raw/gaming_machine_lga.csv', encoding='utf-8-sig')
print('SHAPE:', df.shape)
print('\nCOLUMNS / DTYPES:'); print(df.dtypes)

df['date'] = pd.to_datetime(df['Month Year'], format='%B %Y', errors='coerce')
print('\nDATE RANGE:', df['date'].min().date(), '->', df['date'].max().date())
print('unparsed dates:', df['date'].isna().sum())
print('distinct months:', df['date'].nunique())

print('\nLGAs:', df['LGA Region'].nunique())
print('LGAs per year (first/last):')
yr = df.groupby(df['date'].dt.year)['LGA Region'].nunique()
print(yr.head(3).to_string(), '\n ...\n', yr.tail(3).to_string())

print('\nMISSINGNESS:')
print(df.isna().sum().to_string())

# suppression check
df['Metered Win'] = pd.to_numeric(df['Metered Win'], errors='coerce')
sup = df['Metered Win'].isna()
print('\nSUPPRESSED rows (Metered Win blank): %d  (%.2f%%)' % (sup.sum(), 100*sup.mean()))
print('\nOperational Sites distribution WHERE suppressed:')
print(df.loc[sup,'Operational Sites'].value_counts().sort_index().to_string())
print('\nMax Operational Sites where suppressed:', df.loc[sup,'Operational Sites'].max())
print('Min Operational Sites where NOT suppressed:', df.loc[~sup,'Operational Sites'].min())

print('\nTOP 5 LGAs by suppressed months:')
print(df[sup]['LGA Region'].value_counts().head(5).to_string())

print('\nMETERED WIN summary (non-suppressed):')
print(df['Metered Win'].describe().apply(lambda v: f'{v:,.0f}').to_string())
