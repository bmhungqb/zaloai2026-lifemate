#!/usr/bin/env python3
"""LifeMate baseline driver.
  python -m lifemate.run asr    --split train --n 300            # transcribe holdout (+history) -> outputs/asr_train.json
  python -m lifemate.run agent  --split train --n 300 --mode oracle|asr --shots 4 --backend vllm
  python -m lifemate.run agent  --split public_test --mode asr --shots 4  # writes outputs/submission_public_test.jsonl
Dev holdout = fixed random sample of train (seed 0); retrieval pool = train minus holdout."""
import argparse, json, random, sys, time, os
from pathlib import Path
from .common import find_root, jl, Retriever
from . import agent as A, score as S

def holdout(rows, n, seed=0):
    ids = [r['id'] for r in rows]; random.Random(seed).shuffle(ids); return set(ids[:n])

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('cmd', choices=['asr', 'agent', 'score_gold'])
    ap.add_argument('--root'); ap.add_argument('--split', default='train'); ap.add_argument('--n', type=int, default=300)
    ap.add_argument('--mode', default='oracle', choices=['oracle', 'asr']); ap.add_argument('--shots', type=int, default=0)
    ap.add_argument('--backend', default='stub'); ap.add_argument('--model', default='Qwen/Qwen3-4B-Instruct-2507')
    ap.add_argument('--asr_model', default='openai/whisper-large-v3-turbo'); ap.add_argument('--out', default='outputs')
    ap.add_argument('--max_len', type=int, default=12288); ap.add_argument('--tp', type=int, default=1); ap.add_argument('--tag', default='')
    a = ap.parse_args(); root = find_root(a.root); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    train = jl(root / 'train' / 'train.jsonl'); split = a.split
    rows = jl(root / split / f'{split}.jsonl'); cat = json.load(open(root / split / f'{split}_tools_catalog.json', encoding='utf-8'))
    ho = holdout(train, a.n) if split == 'train' else None
    items = [r for r in rows if ho is None or r['id'] in ho]
    if a.cmd == 'asr':
        from .asr import collect, transcribe
        paths = collect(root, split, items); print(len(paths), 'wav files')
        transcribe(root, split, paths, a.asr_model, out=str(out / f'asr_{split}.json')); return
    if a.cmd == 'score_gold':   # sanity: gold calls -> 100, empty -> share of out-of-scope
        zs = S.add_zaic(root, split); subs = {}
        for it in items:
            env = zs.ToolEnv(split, it['id'], root=root)
            for c in it['label']['tool_calls']: env.call(c['name'], c['arguments'])
            subs[it['id']] = {'tool_calls': env.get_tool_calls(), 'state_hash': env.get_state_hash()}
        print('gold ->', S.evaluate(root, split, items, subs, cat)[0])
        zs = S.add_zaic(root, split); empty = {}
        for it in items:
            env = zs.ToolEnv(split, it['id'], root=root); empty[it['id']] = {'tool_calls': [], 'state_hash': env.get_state_hash()}
        print('empty ->', S.evaluate(root, split, items, empty, cat)[0]); return
    # ---- agent ----
    asr_path = out / f'asr_{split}.json'; asr = json.load(open(asr_path, encoding='utf-8')) if asr_path.exists() else {}
    if a.mode == 'asr' and not asr: sys.exit('run the asr stage first')
    def texts_of(it):
        if a.mode == 'oracle' and 'audio_text' in it:
            return it['audio_text'], {h['audio']: h.get('audio_text', '') for h in it.get('history') or [] if h['role'] == 'user'}
        return asr.get(it['audio'], ''), {h['audio']: asr.get(h['audio'], '') for h in it.get('history') or [] if h['role'] == 'user'}
    pool = [r for r in train if (ho is None or r['id'] not in ho)]
    retr = Retriever([r['audio_text'] for r in pool]) if a.shots else None
    kw = {'max_len': a.max_len, 'tp': a.tp} if a.backend == 'vllm' else {}
    from .backends import make; be = make(a.backend, a.model, **kw)
    t0 = time.time(); res = A.run_batch(items, root, split, cat, be, texts_of, pool, retr, a.shots, exclude=False)
    dt = time.time() - t0; tag = a.tag or f'{split}_{a.mode}_s{a.shots}'
    json.dump(res, open(out / f'pred_{tag}.json', 'w', encoding='utf-8'), ensure_ascii=False)
    print(f'{len(items)} samples in {dt:.0f}s ({dt / len(items):.2f}s/sample amortised); max per-sample steps {max(r["steps"] for r in res.values())}')
    if split == 'train':
        sc, per, br = S.evaluate(root, split, items, res, cat); print(f'SCORE {sc:.2f}  breakdown(n, score): {br}')
        json.dump({'score': sc, 'breakdown': br, 'per': per, 'n': len(items), 'seconds': dt}, open(out / f'score_{tag}.json', 'w'), ensure_ascii=False)
    else:
        with open(out / f'submission_{split}.jsonl', 'w', encoding='utf-8') as f:
            for it in items: f.write(json.dumps({'id': it['id'], 'tool_calls': res[it['id']]['tool_calls'], 'state_hash': res[it['id']]['state_hash']}, ensure_ascii=False) + '\n')
        print('wrote', out / f'submission_{split}.jsonl')
if __name__ == '__main__': main()
