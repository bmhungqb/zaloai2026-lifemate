#!/usr/bin/env python3
"""python -m datamate.run prep|solve --split train|public_test --n 120 --backend vllm --model Qwen/Qwen3-8B --K 3"""
import argparse, json, random, os, sys, time, collections
from pathlib import Path

def jl(p): return [json.loads(x) for x in open(p, encoding='utf-8') if x.strip()]

class VLLM:
    def __init__(self, model, tp=1, max_len=16384, gpu_util=0.9, max_tokens=1400, temperature=0.6):
        from vllm import LLM, SamplingParams
        self.SP = SamplingParams; self.t, self.mt = temperature, max_tokens
        self.llm = LLM(model=model, dtype='half', max_model_len=max_len, tensor_parallel_size=tp, gpu_memory_utilization=gpu_util, enable_prefix_caching=True, seed=0)
    def generate(self, batch, n=1):
        sp = self.SP(n=n, temperature=(self.t if n > 1 else 0.0), top_p=0.95, max_tokens=self.mt, seed=0)
        outs = self.llm.chat(batch, sp, use_tqdm=False, chat_template_kwargs={'enable_thinking': False})
        return [[c.text for c in o.outputs] for o in outs]

class Stub:
    def generate(self, batch, n=1): return [['```python\nANSWER = ("unanswerable", None)\n```'] * n for _ in batch]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('cmd', choices=['prep', 'solve'])
    ap.add_argument('--root', default='data/datamate'); ap.add_argument('--split', default='train'); ap.add_argument('--n', type=int, default=120)
    ap.add_argument('--K', type=int, default=3); ap.add_argument('--backend', default='stub'); ap.add_argument('--model', default='Qwen/Qwen3-8B')
    ap.add_argument('--tp', type=int, default=1); ap.add_argument('--out', default='outputs'); ap.add_argument('--cache', default='/tmp/dm_cache'); ap.add_argument('--tag', default='')
    ap.add_argument('--biz', default=''); ap.add_argument('--workers', type=int, default=4); ap.add_argument('--vote_min', type=int, default=2)
    a = ap.parse_args(); os.environ['DM_CACHE'] = a.cache
    root = Path(a.root); sd = root / a.split; out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    qs = jl(sd / 'questions.jsonl'); gold = {r['id']: r for r in jl(sd / 'answers.jsonl')} if (sd / 'answers.jsonl').exists() else {}
    if a.biz: qs = [q for q in qs if q['business_id'] in a.biz.split(',')]
    if a.split == 'train' and a.n < len(qs):
        random.Random(0).shuffle(qs); qs = qs[:a.n]
    from .prep import warm, get_schema
    from .agent import solve; from .score import evaluate
    bizs = sorted({q['business_id'] for q in qs}); t0 = time.time()
    for b in bizs: get_schema(sd / 'data' / b, Path(a.cache) / 'schemas'); warm(sd / 'data' / b); print('prepped', b, f'{time.time() - t0:.0f}s', flush=True)
    if a.cmd == 'prep': return
    be = VLLM(a.model, a.tp) if a.backend == 'vllm' else Stub()
    t0 = time.time(); preds, logs = solve(qs, sd, be, K=a.K, workers=a.workers, cache=a.cache, vote_min=a.vote_min, log=lambda s: print(s, flush=True))
    dt = time.time() - t0; tag = a.tag or f'{a.split}_K{a.K}'
    json.dump(logs, open(out / f'logs_{tag}.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f'{len(qs)} questions in {dt:.0f}s')
    if gold:
        g = {q['id']: gold[q['id']] for q in qs}; print('RESULT', json.dumps(evaluate(preds, g)))
        for lo, name in [(0, 'all')]: pass
        by = collections.defaultdict(lambda: [0, 0, 0])
        for i, gr in g.items():
            p = preds.get(i); k = by[gr['type']]
            if not p: k[2] += 1
            elif __import__('datamate.score', fromlist=['x']).match(p, gr): k[0] += 1
            else: k[1] += 1
        print('by type [correct, wrong, abstain]:', dict(by))
    else:
        with open(out / 'submission.jsonl', 'w', encoding='utf-8') as f:
            for q in qs:
                p = preds.get(q['id']); f.write(json.dumps({'id': q['id'], **(p if p else {'type': 'abstain', 'answer': None})}, ensure_ascii=False) + '\n')
        print('wrote', out / 'submission.jsonl')
if __name__ == '__main__': main()
