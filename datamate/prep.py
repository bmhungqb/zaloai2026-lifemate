"""Per-business schema preview (raw first rows, sheet names, row counts) + pickle cache of every table (all cells as str)."""
import json, os, hashlib, sqlite3, glob
from pathlib import Path
import pandas as pd

CACHE = Path(os.environ.get('DM_CACHE', '/tmp/dm_cache'))

def cache_path(path, sheet, header):
    h = hashlib.md5(f'{path}|{sheet}|{header}'.encode()).hexdigest()[:16]; return CACHE / f'{h}.pkl'

def read(path, sheet=None, header=0, table=None):
    """Read one table; every cell as str. Cached."""
    CACHE.mkdir(parents=True, exist_ok=True); cp = cache_path(path, sheet or table, header)
    if cp.exists(): return pd.read_pickle(cp)
    p = str(path); low = p.lower()
    if low.endswith('.csv'): df = pd.read_csv(p, dtype=str, header=header, encoding='utf-8-sig', on_bad_lines='skip')
    elif low.endswith(('.xlsx', '.xlsm')): df = pd.read_excel(p, sheet_name=sheet if sheet is not None else 0, dtype=str, header=header, engine='openpyxl')
    elif low.endswith(('.sqlite', '.db', '.sqlite3')):
        con = sqlite3.connect(p); df = pd.read_sql(f'select * from "{table or sheet}"', con).astype(str); con.close()
    else: raise ValueError(p)
    df.to_pickle(cp); return df

def _xlsx_preview(p, nrows=6, max_sheets=3):
    import openpyxl
    wb = openpyxl.load_workbook(p, read_only=True, data_only=True); out = []
    names = wb.sheetnames
    out.append(f'  sheets ({len(names)}): ' + ', '.join(f'{n}[{wb[n].max_row or "?"} rows]' for n in names[:40]) + (' …' if len(names) > 40 else ''))
    for n in names[:max_sheets]:
        ws = wb[n]; out.append(f'  -- sheet "{n}" raw first rows:')
        for i, row in enumerate(ws.iter_rows(values_only=True)):
            if i >= nrows: break
            out.append('   | ' + ' | '.join('' if v is None else str(v)[:40] for v in row[:25]))
    wb.close(); return '\n'.join(out)

def _csv_preview(p, nrows=6):
    df = pd.read_csv(p, dtype=str, header=None, nrows=nrows, encoding='utf-8-sig', on_bad_lines='skip')
    with open(p, 'rb') as f: n = sum(1 for _ in f)
    rows = ['   | ' + ' | '.join('' if pd.isna(v) else str(v)[:40] for v in r[:25]) for r in df.values.tolist()]
    return f'  {n} lines, raw first rows:\n' + '\n'.join(rows)

def _sqlite_preview(p):
    con = sqlite3.connect(p); out = []
    for (name, sql) in con.execute("select name, sql from sqlite_master where type='table'").fetchall():
        n = con.execute(f'select count(*) from "{name}"').fetchone()[0]; out.append(f'  table {name} ({n} rows): {sql}')
        for r in con.execute(f'select * from "{name}" limit 3').fetchall(): out.append('   | ' + ' | '.join(str(v)[:40] for v in r))
    con.close(); return '\n'.join(out)

def schema_text(bdir):
    bdir = Path(bdir); parts = []
    for p in sorted(bdir.iterdir()):
        if p.name.lower() == 'readme.md': continue
        parts.append(f'FILE `{p.name}` ({p.stat().st_size // 1024} KB)')
        try:
            low = p.name.lower()
            parts.append(_csv_preview(p) if low.endswith('.csv') else _xlsx_preview(p) if low.endswith(('.xlsx', '.xlsm')) else _sqlite_preview(p) if low.endswith(('.sqlite', '.db', '.sqlite3')) else '  (unknown format)')
        except Exception as e: parts.append(f'  (preview failed: {e})')
    return '\n'.join(parts)

def warm(bdir):
    """Pre-parse every table (header=0, all sheets) so question-time code starts fast."""
    for p in sorted(Path(bdir).iterdir()):
        low = p.name.lower()
        try:
            if low.endswith('.csv'): read(p)
            elif low.endswith(('.xlsx', '.xlsm')):
                for s in pd.ExcelFile(p, engine='openpyxl').sheet_names: read(p, s)
            elif low.endswith(('.sqlite', '.db')):
                con = sqlite3.connect(p); ts = [r[0] for r in con.execute("select name from sqlite_master where type='table'")]; con.close()
                for t in ts: read(p, table=t)
        except Exception as e: print('warm fail', p, e)

def get_schema(bdir, cache_dir):
    f = Path(cache_dir) / f'schema_{Path(bdir).name}.json'
    if f.exists(): return json.load(open(f, encoding='utf-8'))['text']
    t = schema_text(bdir); f.parent.mkdir(parents=True, exist_ok=True); json.dump({'text': t}, open(f, 'w', encoding='utf-8'), ensure_ascii=False); return t
