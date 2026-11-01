"""Shared helpers for the LifeMate baseline. Pure stdlib (+ optional numpy)."""
import json, sys, re, math, unicodedata, collections
from pathlib import Path

def find_root(hint=None):
    """Folder that holds train/train.jsonl and public_test/public_test.jsonl (works locally and on Kaggle)."""
    cands = [Path(hint)] if hint else []
    cands += [Path('.'), Path('data/lifemate/train+public_test'), Path('/kaggle/input')]
    for base in cands:
        if not base.exists(): continue
        for p in [base] + list(base.glob('*')) + list(base.glob('*/*')) + list(base.glob('*/*/*')):
            if (p / 'train' / 'train.jsonl').is_file(): return p
    raise FileNotFoundError('cannot find train/train.jsonl; pass --root')

def jl(path):
    with open(path, encoding='utf-8') as f: return [json.loads(x) for x in f if x.strip()]

def add_zaic(root, split):
    """Make the split's zaic_tools importable (each split ships its own version)."""
    for k in [k for k in sys.modules if k == 'zaic_tools' or k.startswith('zaic_tools.')]: del sys.modules[k]
    sys.path = [p for p in sys.path if not p.endswith(('/train', '/public_test', '/private_test'))]
    sys.path.insert(0, str(Path(root) / split))
    import zaic_tools
    return zaic_tools

def fold(s):
    t = unicodedata.normalize('NFD', str(s).lower()).replace('đ', 'd')
    return ' '.join(''.join(c for c in t if not unicodedata.combining(c)).split())

# ---------- tiny TF-IDF retriever (char 3-grams over folded text) ----------
class Retriever:
    def __init__(self, docs):
        self.docs = docs
        self.vecs = [self._tf(d) for d in docs]
        df = collections.Counter(k for v in self.vecs for k in v)
        n = len(docs)
        self.idf = {k: math.log((1 + n) / (1 + c)) + 1 for k, c in df.items()}
        self.vecs = [self._norm(v) for v in self.vecs]
    @staticmethod
    def _grams(t):
        t = ' ' + fold(t) + ' '
        return [t[i:i + 3] for i in range(len(t) - 2)]
    def _tf(self, t): return collections.Counter(self._grams(t))
    def _norm(self, v):
        w = {k: c * getattr(self, 'idf', {}).get(k, 1.0) for k, c in v.items()}
        nrm = math.sqrt(sum(x * x for x in w.values())) or 1.0
        return {k: x / nrm for k, x in w.items()}
    def top(self, q, k=4, exclude=None):
        qv = self._norm(self._tf(q)); sc = []
        for i, v in enumerate(self.vecs):
            if exclude is not None and i in exclude: continue
            s = sum(x * v.get(g, 0.0) for g, x in qv.items()); sc.append((s, i))
        sc.sort(reverse=True)
        return [i for _, i in sc[:k]]
