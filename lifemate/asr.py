"""Stage 1: transcribe every wav (final turn + history user turns) with Whisper; cache to JSON {relative_audio_path: text}."""
import json, sys, time
from pathlib import Path

def collect(root, split, rows, final_only=False):
    paths = []
    for it in rows:
        paths.append(it['audio'])
        if not final_only:
            for h in it.get('history') or []:
                if h['role'] == 'user': paths.append(h['audio'])
    return sorted(set(paths))

def transcribe(root, split, paths, model='openai/whisper-large-v3-turbo', bs=16, out=None):
    import torch, soundfile as sf
    from transformers import pipeline
    pipe = pipeline('automatic-speech-recognition', model=model, torch_dtype=torch.float16, device=0, chunk_length_s=30)
    res = json.load(open(out, encoding='utf-8')) if out and Path(out).exists() else {}
    todo = [p for p in paths if p not in res]; t0 = time.time()
    for i in range(0, len(todo), bs):
        chunk = todo[i:i + bs]
        arrs = [sf.read(str(Path(root) / split / p))[0].astype('float32') for p in chunk]
        outs = pipe([{'raw': a, 'sampling_rate': 16000} for a in arrs], batch_size=bs, generate_kwargs={'language': 'vi', 'task': 'transcribe', 'num_beams': 1})
        for p, o in zip(chunk, outs): res[p] = o['text'].strip()
        if out and (i // bs) % 10 == 0:
            json.dump(res, open(out, 'w', encoding='utf-8'), ensure_ascii=False); print(f'asr {len(res)}/{len(paths)} {time.time() - t0:.0f}s', flush=True)
    if out: json.dump(res, open(out, 'w', encoding='utf-8'), ensure_ascii=False)
    return res
