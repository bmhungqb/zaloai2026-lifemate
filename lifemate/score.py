"""Local re-implementation of the official LifeMate metric (from the portal's Evaluation tab).
Sample score: gold empty -> 1 iff submission empty else 0; state_hash != gold or M==0 -> 0; else F1 with P=M/(M+U), R=M/|G|.
A call matches a gold call if same tool name, every non-fuzzy arg identical, every fuzzy arg token-F1 > 0.6 (lower+NFC).
Each call matches at most one; order ignored. Rejected (schema-invalid) calls count as unmatched. Final = 100/N * sum."""
import json, unicodedata, re, collections
from .common import add_zaic

def _nfc(s): return unicodedata.normalize('NFC', str(s)).lower()
def _toks(s): return re.findall(r'\w+', _nfc(s))
def token_f1(a, b):
    ta, tb = _toks(a), _toks(b)
    if not ta and not tb: return 1.0
    common = sum((collections.Counter(ta) & collections.Counter(tb)).values())
    if common == 0: return 0.0
    p, r = common / len(ta), common / len(tb); return 2 * p * r / (p + r)

def fuzzy_keys(catalog):
    out = {}
    for a in catalog['apps']:
        for t in a['tools']:
            props = (t.get('parameters') or {}).get('properties', {})
            out[t['name']] = {k for k, v in props.items() if v.get('x-match') == 'fuzzy'}
    return out

def _same(a, b):
    if isinstance(a, str) and isinstance(b, str): return ' '.join(a.split()) == ' '.join(b.split())
    return json.dumps(a, sort_keys=True, ensure_ascii=False) == json.dumps(b, sort_keys=True, ensure_ascii=False)

def calls_match(sub, gold, fz):
    if sub['name'] != gold['name']: return False
    sa, ga = sub.get('arguments') or {}, gold.get('arguments') or {}
    f = fz.get(gold['name'], set())
    for k in set(sa) | set(ga):
        if k in f:
            if k not in sa or k not in ga: return False
            if token_f1(sa[k], ga[k]) <= 0.6: return False
        elif k not in sa or k not in ga or not _same(sa[k], ga[k]): return False
    return True

def sample_score(sub_calls, sub_hash, gold_calls, gold_hash, fz):
    if not gold_calls: return 1.0 if not sub_calls else 0.0
    if sub_hash != gold_hash: return 0.0
    used, M = set(), 0
    for s in sub_calls:
        for j, g in enumerate(gold_calls):
            if j not in used and calls_match(s, g, fz): used.add(j); M += 1; break
    if M == 0: return 0.0
    U = len(sub_calls) - M; P, R = M / (M + U), M / len(gold_calls)
    return 2 * P * R / (P + R)

def gold_hashes(root, split, rows):
    """state_hash of the gold final state, obtained by replaying the gold calls in a fresh ToolEnv."""
    z = add_zaic(root, split); out = {}
    for it in rows:
        env = z.ToolEnv(split, it['id'], root=root)
        for c in it['label']['tool_calls']: env.call(c['name'], c['arguments'])
        out[it['id']] = env.get_state_hash()
    return out

def evaluate(root, split, rows, subs, catalog):
    """rows: train rows (with label); subs: {id: {'tool_calls', 'state_hash'}}. Returns (final_score, per-id scores, breakdown)."""
    fz = fuzzy_keys(catalog); gh = gold_hashes(root, split, rows); per = {}
    for it in rows:
        s = subs.get(it['id'], {'tool_calls': [], 'state_hash': None})
        per[it['id']] = sample_score(s['tool_calls'], s['state_hash'], it['label']['tool_calls'], gh[it['id']], fz)
    groups = collections.defaultdict(list)
    for it in rows:
        g = it['label']['tool_calls']
        kind = 'out_of_scope' if not g else ('ask' if g[-1]['name'] == 'sys.ask_missing_info' else 'execute')
        groups[kind].append(per[it['id']])
    n = len(rows)
    return 100 * sum(per.values()) / n, per, {k: (len(v), 100 * sum(v) / len(v)) for k, v in groups.items()}
