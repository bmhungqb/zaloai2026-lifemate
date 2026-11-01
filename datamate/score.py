"""Penalized accuracy for DataMate: +1 correct, -1 wrong, 0 abstain (missing/None). Matching rules from the portal (secondary sources)."""
import unicodedata, re, json

def norm(s): return ' '.join(re.sub(r'[^\w]+', ' ', unicodedata.normalize('NFC', str(s)).lower()).replace('_', ' ').split())
def num(x):
    try: return round(float(x), 2)
    except Exception: return None

def _elem(p, g, alias=()):
    if isinstance(g, (int, float)) and not isinstance(g, bool):
        return isinstance(p, (int, float)) and not isinstance(p, bool) and num(p) == num(g)
    cands = [g] + list(alias or [])
    return isinstance(p, str) and norm(p) in {norm(c) for c in cands}

def match(pred, gold):
    """pred: {'type','answer'} ; gold: row of answers.jsonl. Returns True/False (caller handles abstain)."""
    t = gold['type']
    if pred.get('type') != t: return False
    a, g, al = pred.get('answer'), gold.get('answer'), gold.get('alias') or []
    if t == 'unanswerable': return True
    if t in ('number', 'percent'): return isinstance(a, (int, float)) and not isinstance(a, bool) and num(a) == num(g)
    if t == 'string': return isinstance(a, str) and norm(a) in {norm(x) for x in [g] + list(al)}
    if t == 'date': return str(a) == str(g)
    if t == 'boolean': return a is g or a == g and isinstance(a, bool)
    if t == 'list':
        if not isinstance(a, list) or len(a) != len(g): return False
        return all(_elem(p, x, al[i] if i < len(al) else ()) for i, (p, x) in enumerate(zip(a, g)))
    if t == 'table':
        if not isinstance(a, dict) or not isinstance(g, dict): return False
        ac, gc = [norm(c) for c in a.get('columns', [])], [norm(c) for c in g['columns']]
        if len(ac) != len(gc): return False
        perm = [ac.index(c) if c in ac and ac.count(c) == 1 else i for i, c in enumerate(gc)]
        def key(row, order): return json.dumps([(num(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else norm(v)) for v in (row[i] for i in order)])
        try:
            prow = sorted(key(r, perm) for r in a['rows']); grow = sorted(key(r, range(len(gc))) for r in g['rows'])
        except Exception: return False
        if prow == grow: return True
        return False
    return False

def evaluate(preds, golds):
    """preds {id: {'type','answer'} or None}, golds {id: row}. Returns PA (0-1) and counts."""
    c = w = ab = 0
    for i, g in golds.items():
        p = preds.get(i)
        if not p or p.get('type') in (None, 'abstain'): ab += 1
        elif match(p, g): c += 1
        else: w += 1
    n = len(golds); return {'PA': (c - w) / n, 'correct': c, 'wrong': w, 'abstain': ab, 'n': n, 'coverage': (c + w) / n, 'acc_answered': c / max(1, c + w)}
