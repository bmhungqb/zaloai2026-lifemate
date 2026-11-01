#!/usr/bin/env python3
"""Profile the OFFICIAL LifeMate data kit (downloaded by you from the portal).
Usage: python3 -I analyze_lifemate.py <kit_dir> [out.md]
Schema-adaptive: finds *.json / *.jsonl, reports keys, counts tool usage, answer-behaviour mix, audio durations.
Aggregate stats only."""
import sys, os, json, wave, collections, statistics
root = sys.argv[1]; out = sys.argv[2] if len(sys.argv) > 2 else "lifemate_profile.md"
L = []; p = L.append
files = [os.path.join(d, f) for d, _, fs in os.walk(root) for f in fs]
p(f"# LifeMate profile: {root}\n"); p(f"- files: {len(files)}; by ext: {dict(collections.Counter(os.path.splitext(f)[1] for f in files))}")
durs = []
for f in files:
    if f.lower().endswith(".wav"):
        try:
            with wave.open(f) as w: durs.append(w.getnframes() / w.getframerate())
        except Exception: pass
if durs: p(f"- wav: {len(durs)} files, total {sum(durs)/3600:.2f} h, mean {statistics.mean(durs):.2f}s, p95 {sorted(durs)[int(.95*len(durs))]:.1f}s, max {max(durs):.1f}s")
def load(f):
    try:
        if f.endswith(".jsonl"): return [json.loads(x) for x in open(f, encoding="utf-8") if x.strip()]
        if f.endswith(".json"):
            d = json.load(open(f, encoding="utf-8")); return d if isinstance(d, list) else [d]
    except Exception: return []
    return []
def calls(rec):
    lab = rec.get("label") or rec
    return lab.get("tool_calls") or []
for f in sorted(x for x in files if x.endswith((".json", ".jsonl")) and os.path.getsize(x) < 2e9):
    R = load(f)
    if not R or not isinstance(R[0], dict): continue
    p(f"\n## {os.path.relpath(f, root)} — {len(R)} records"); p(f"- top-level keys: {sorted(R[0].keys())}")
    if any("label" in r or "tool_calls" in r for r in R[:50]):
        n = [len(calls(r)) for r in R]; p(f"- calls/record dist: {dict(sorted(collections.Counter(n).items()))}")
        names = collections.Counter(c.get("name") for r in R for c in calls(r)); p("- top tools: " + ", ".join(f"{k}:{v}" for k, v in names.most_common(25)))
        ask = sum(1 for r in R if any((c.get('name') == 'sys.ask_missing_info') for c in calls(r))); oos = sum(1 for x in n if x == 0)
        p(f"- ask_missing_info records: {ask} ({ask/len(R):.1%}); zero-call (out-of-scope) records: {oos} ({oos/len(R):.1%})")
    ctx = [r.get("context") for r in R if isinstance(r.get("context"), dict)]
    if ctx:
        apps = collections.Counter(a.get("name", a.get("app")) if isinstance(a, dict) else a for c in ctx for a in (c.get("installed_apps") or []))
        p(f"- distinct apps in installed_apps: {len(apps)}; most common: {apps.most_common(8)}")
        p(f"- avg installed apps per record: {statistics.mean(len(c.get('installed_apps') or []) for c in ctx):.1f}")
    hist = [len(r.get("history") or []) for r in R if "history" in r]
    if hist: p(f"- history length: mean {statistics.mean(hist):.1f}, max {max(hist)}, zero-history {sum(1 for h in hist if h==0)/len(hist):.1%}")
open(out, "w", encoding="utf-8").write("\n".join(L)); print("\n".join(L)); print(f"\n-> {out}")
