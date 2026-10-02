import pandas as pd
df = pd.read_csv('data/raw/gaming_machine_lga.csv', encoding='utf-8-sig')
df['date'] = pd.to_datetime(df['Month Year'], format='%B %Y')

print('data rows      :', len(df))
print('first month    :', df['date'].min().strftime('%B %Y'))
print('last month     :', df['date'].max().strftime('%B %Y'))
print('distinct months:', df['date'].nunique())

# expected month sequence
full = pd.date_range(df['date'].min(), df['date'].max(), freq='MS')
print('expected months:', len(full))
missing = set(full) - set(df['date'].unique())
print('MISSING MONTHS :', sorted(pd.Timestamp(m).strftime("%b %Y") for m in missing) or 'none')

print('\nrows per month  -> min %d, max %d' % (df.groupby('date').size().min(), df.groupby('date').size().max()))
print('grid check: 55 LGAs x 266 months =', 55*266, 'vs actual', len(df), '=> gap of', 55*266-len(df))

# which LGA-months are absent from the grid
piv = df.pivot_table(index='LGA Region', columns='date', values='Approved Sites', aggfunc='size')
absent = piv.isna().sum(axis=1)
print('\nLGAs with absent month rows:')
print(absent[absent>0].to_string() if (absent>0).any() else ' none')

print('\nyears covered:', sorted(df["date"].dt.year.unique()))
