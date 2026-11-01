"""Run model-written pandas code in a subprocess; the code must set ANSWER = (type, value)."""
import json, subprocess, sys, tempfile, os, re
from pathlib import Path

PRELUDE = '''
import sys, json, re, math, datetime as dt
sys.path.insert(0, {pkgdir!r})
import numpy as np, pandas as pd
from datamate.prep import read as _read
BASE = {base!r}
AS_OF = pd.Timestamp({asof!r})
CALENDAR = pd.read_csv({cal!r}, dtype=str)
def read(name, sheet=None, header=0):
    """Return DataFrame (all cells str) of file `name` in the business folder; sheet = sheet name (xlsx) or table name (sqlite)."""
    p = BASE + '/' + name
    return _read(p, table=sheet) if name.lower().endswith(('.sqlite', '.db')) else _read(p, sheet, header)
def sheets(name):
    return pd.ExcelFile(BASE + '/' + name, engine='openpyxl').sheet_names
'''
EPILOGUE = '''
def _clean(v):
    if isinstance(v, (np.integer,)): return int(v)
    if isinstance(v, (np.floating,)): v = float(v)
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v): return None
        v = round(v, 2); return int(v) if v == int(v) and abs(v) < 1e15 else v
    if isinstance(v, (pd.Timestamp, dt.datetime, dt.date)): return pd.Timestamp(v).strftime('%Y-%m-%d')
    if isinstance(v, np.bool_): return bool(v)
    if isinstance(v, dict): return {str(k): _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, np.ndarray, pd.Series)): return [_clean(x) for x in list(v)]
    if isinstance(v, pd.DataFrame): return {'columns': [str(c) for c in v.columns], 'rows': [[_clean(x) for x in r] for r in v.values.tolist()]}
    return v
try:
    _t, _v = ANSWER
    print('@@ANSWER@@' + json.dumps({'type': _t, 'answer': _clean(_v)}, ensure_ascii=False))
except NameError:
    print('@@ERR@@ ANSWER was not set')
'''

def extract_code(text):
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.S)
    m = re.findall(r'```(?:python|py)?\s*\n(.*?)```', text, flags=re.S)
    return (m[-1] if m else text).strip()

def run_code(code, bdir, asof, cal, timeout=150, cache='/tmp/dm_cache'):
    pkgdir = str(Path(__file__).resolve().parent.parent)
    src = PRELUDE.format(pkgdir=pkgdir, base=os.path.abspath(str(bdir)), asof=asof, cal=os.path.abspath(str(cal))) + '\n' + code + '\n' + EPILOGUE
    with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False, encoding='utf-8') as f: f.write(src); fn = f.name
    try:
        r = subprocess.run([sys.executable, '-I', fn], capture_output=True, text=True, timeout=timeout, env={**os.environ, 'DM_CACHE': cache, 'PYTHONPATH': pkgdir, 'PYTHONIOENCODING': 'utf-8'}, cwd='/tmp')
    except subprocess.TimeoutExpired: return None, 'TimeoutError: code ran longer than %ds' % timeout
    finally: os.unlink(fn)
    for line in r.stdout.splitlines()[::-1]:
        if line.startswith('@@ANSWER@@'): return json.loads(line[10:]), None
    err = (r.stderr or r.stdout or 'no output').strip().splitlines()
    return None, '\n'.join(err[-8:])[:1500]
