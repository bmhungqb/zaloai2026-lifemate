"""LLM writes pandas code per question; K samples -> execute -> (1 repair round) -> majority vote with abstain."""
import json, collections, re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from .sandbox import run_code, extract_code
from .prep import get_schema

SYSTEM = """You answer Vietnamese business questions by writing Python (pandas) code over a small shop's spreadsheet data.

OUTPUT: exactly one ```python code block, nothing else. The code runs with these predefined names: pd, np, json, re, dt, BASE, AS_OF (today as pd.Timestamp), CALENDAR (lunar calendar DataFrame, all str), read(filename, sheet=None, header=0) -> DataFrame with ALL cells as str (parse numbers/dates yourself: "10.000đ", "105k", "1tr440", "531 500", dd/mm/yyyy, yyyy-mm-dd, Excel serials; strip, unify case/NFC; status words like Xong/ok/Đã hoàn thành/hủy vary), sheets(filename) -> sheet names. Files in the business folder are listed below (raw first rows are shown so you can pick the right header row: use header=N if there are title rows; skip repeated header rows, "Tổng"/"Ghi chú" rows). Data may be split over many sheets/files by month: concatenate them.
Finish by setting ANSWER = (type, value) with type one of:
 number (JSON number: int, or float rounded to 2 decimals) | percent (number in percent units, e.g. 12.5 means 12.5%; keep sign) | string (name of the item/customer as written in the data) | date ('yyyy-mm-dd', or 'yyyy-mm' when the question asks for a month) | boolean (True/False) | list (ordered python list of numbers or strings) | table ({'columns': [...], 'rows': [[...]]}) | unanswerable (value None).
Use ('unanswerable', None) only when the data cannot answer it per the rulebook (missing column, period completely outside the data range, entity absent, opinion/forecast).
Apply the RULEBOOK exactly (valid orders only, week/month/Tet definitions, units, ranking tie-break = code ascending). 'Today' = AS_OF. Do not print long output. Be careful and precise; compute directly from the data, never guess."""

def build_prompt(q, rulebook, readme, schema):
    sysm = SYSTEM + '\n\n=== RULEBOOK ===\n' + rulebook
    user = f"=== BUSINESS {q['business_id']} README ===\n{readme}\n\n=== FILES ===\n{schema}\n\n=== QUESTION ===\nas_of_date (today) = {q['as_of_date']}\n{q['question']}\n\nWrite the code."
    return [{'role': 'system', 'content': sysm}, {'role': 'user', 'content': user}]

def canon(ans):
    if ans is None: return None
    return json.dumps(ans, ensure_ascii=False, sort_keys=True)

def solve(questions, split_dir, backend, K=3, repair=True, workers=4, cache='/tmp/dm_cache', log=print, vote_min=2):
    split_dir = Path(split_dir); rule = (split_dir / 'rulebook.md').read_text(encoding='utf-8'); cal = split_dir / 'calendar.csv'
    readmes, schemas = {}, {}
    for q in questions:
        b = q['business_id']
        if b not in schemas:
            bd = split_dir / 'data' / b; schemas[b] = get_schema(bd, Path(cache) / 'schemas')
            rp = bd / 'README.md'; readmes[b] = rp.read_text(encoding='utf-8') if rp.exists() else '(no README)'
    msgs = [build_prompt(q, rule, readmes[q['business_id']], schemas[q['business_id']]) for q in questions]
    outs = backend.generate(msgs, n=K)                       # list[list[str]]
    jobs = [(qi, k, extract_code(o)) for qi, os_ in enumerate(outs) for k, o in enumerate(os_)]
    def ex(job):
        qi, k, code = job; q = questions[qi]
        return run_code(code, split_dir / 'data' / q['business_id'], q['as_of_date'], cal, cache=cache)
    with ThreadPoolExecutor(workers) as tp: res = list(tp.map(ex, jobs))
    log(f'round0: {sum(r[0] is not None for r in res)}/{len(res)} executed')
    if repair:
        bad = [i for i, r in enumerate(res) if r[0] is None]
        rm = []
        for i in bad:
            qi, k, code = jobs[i]
            rm.append(msgs[qi] + [{'role': 'assistant', 'content': '```python\n' + code + '\n```'}, {'role': 'user', 'content': f'It failed:\n{res[i][1]}\nFix the code. Reply with one ```python block.'}])
        if rm:
            fix = backend.generate(rm, n=1); jobs2 = [(jobs[i][0], jobs[i][1], extract_code(f[0])) for i, f in zip(bad, fix)]
            with ThreadPoolExecutor(workers) as tp: res2 = list(tp.map(ex, jobs2))
            for i, j2, r2 in zip(bad, jobs2, res2): jobs[i], res[i] = j2, r2
            log(f'after repair: {sum(r[0] is not None for r in res)}/{len(res)} executed')
    by = collections.defaultdict(list)
    for (qi, k, code), (ans, err) in zip(jobs, res): by[qi].append((ans, err, code))
    preds, logs = {}, {}
    for qi, q in enumerate(questions):
        cands = by[qi]; votes = collections.Counter(canon(a) for a, e, c in cands if a is not None)
        top = votes.most_common(1)
        pred = None
        if top and top[0][1] >= vote_min: pred = next(a for a, e, c in cands if a is not None and canon(a) == top[0][0])
        preds[q['id']] = pred
        logs[q['id']] = {'question': q['question'], 'votes': dict(votes), 'pred': pred, 'errors': [e for a, e, c in cands if a is None][:2], 'code': next((c for a, e, c in cands if a is not None and canon(a) == (top[0][0] if top else None)), cands[0][2] if cands else '')}
    return preds, logs
