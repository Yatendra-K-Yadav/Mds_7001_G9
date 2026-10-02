"""Phase 2a — build LGA name crosswalk between OLGR gambling data and ABS geography."""
import pandas as pd, openpyxl, re, unicodedata

def norm(s):
    """normalise an LGA name for matching"""
    s = str(s).strip()
    s = re.sub(r'\s*\((?:Qld|S|C|R|T|DC|M|A)\)\s*$', '', s, flags=re.I)  # drop state/type suffix
    s = unicodedata.normalize('NFKD', s).encode('ascii','ignore').decode()
    s = re.sub(r'[^A-Za-z0-9 ]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip().upper()
    return s

g = pd.read_csv('data/raw/Gaming Machine Data by Local Government Areas/gaming_machine_lga.csv', encoding='utf-8-sig')
gamb = sorted(g['LGA Region'].unique())

wb = openpyxl.load_workbook('data/raw/Regional population by age and sex/32350DS0006_2001-25.xlsx',
                            read_only=True, data_only=True)
ws = wb['Table 3']
abs_rows = {}
for r in ws.iter_rows(min_row=7, max_col=5, values_only=True):
    if r[2] == 'Queensland' and r[4]:
        abs_rows[str(r[4]).strip()] = int(r[3])   # name -> LGA code
wb.close()

abs_lookup = {}
for name, code in abs_rows.items():
    abs_lookup.setdefault(norm(name), (name, code))

rows, unmatched = [], []
for gname in gamb:
    key = norm(gname)
    if key in abs_lookup:
        abs_name, abs_code = abs_lookup[key]
        rows.append({'gambling_lga': gname, 'abs_lga_name': abs_name,
                     'abs_lga_code': abs_code, 'match': 'auto'})
    else:
        unmatched.append(gname)
        rows.append({'gambling_lga': gname, 'abs_lga_name': None,
                     'abs_lga_code': None, 'match': 'UNMATCHED'})

cw = pd.DataFrame(rows)
cw.to_csv('data/interim/lga_crosswalk.csv', index=False)

print('gambling LGAs      :', len(gamb))
print('ABS QLD LGAs       :', len(abs_rows))
print('matched            :', (cw['match']=='auto').sum())
print('UNMATCHED          :', len(unmatched), unmatched if unmatched else '')
print()
abs_only = sorted(set(abs_rows) - set(cw['abs_lga_name'].dropna()))
print('ABS LGAs with NO gambling data (%d):' % len(abs_only))
for n in abs_only: print('   -', n)
