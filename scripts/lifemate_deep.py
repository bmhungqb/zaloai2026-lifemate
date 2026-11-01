#!/usr/bin/env python3
"""Targeted LifeMate analysis: reads only <split>.jsonl, catalogs and wav headers (NOT *_environment/state.json)."""
import sys, os, json, wave, collections as C, statistics as S, re
root = sys.argv[1]
def jl(p): return [json.loads(x) for x in open(p, encoding="utf-8") if x.strip()]
tr = jl(f"{root}/train/train.jsonl"); pu = jl(f"{root}/public_test/public_test.jsonl")
ctr = json.load(open(f"{root}/train/train_tools_catalog.json")); cpu = json.load(open(f"{root}/public_test/public_test_tools_catalog.json"))
def apps(c): return {a["id"]: {t["name"]: t for t in a["tools"]} for a in c["apps"]}
atr, apu = apps(ctr), apps(cpu)
P = print
P(f"train {len(tr)}  public {len(pu)}  catalog schema {ctr['schema_version']} / {cpu['schema_version']}")
P(f"apps: train {len(atr)} public {len(apu)} common {len(set(atr)&set(apu))} only-public {sorted(set(apu)-set(atr))} only-train {sorted(set(atr)-set(apu))}")
nt = lambda a: sum(len(v) for v in a.values())
P(f"tools: train {nt(atr)} public {nt(apu)}; common tool names {len({t for a in atr.values() for t in a} & {t for a in apu.values() for t in a})}")
kinds = C.Counter(t["kind"] for a in atr.values() for t in a.values()); P("tool kinds (train):", dict(kinds))
def inst(x): return list(x["context"]["installed_apps"].keys())
for name, D in (("train", tr), ("public", pu)):
    n = [len(inst(x)) for x in D]; h = [len(x.get("history") or []) for x in D]
    P(f"[{name}] installed apps/sample mean {S.mean(n):.1f} min {min(n)} max {max(n)}; history turns mean {S.mean(h):.1f} max {max(h)}; no-history {sum(1 for v in h if v==0)/len(h):.1%}")
    up = [x["context"].get("user_profile") or {} for x in D]
    P("   user_profile keys:", sorted({k for u in up for k in u}))
    ac = C.Counter(a for x in D for a in inst(x)); P("   most installed:", ac.most_common(10))
calls = [x["label"]["tool_calls"] for x in tr]
ncl = C.Counter(len(c) for c in calls); P("calls/sample:", dict(sorted(ncl.items())))
ask = [any(c["name"] == "sys.ask_missing_info" for c in cs) for cs in calls]; oos = [len(cs) == 0 for cs in calls]
P(f"ask_missing_info {sum(ask)} ({sum(ask)/len(tr):.1%}); out-of-scope(empty) {sum(oos)} ({sum(oos)/len(tr):.1%}); ask-as-last-call {sum(1 for cs in calls if cs and cs[-1]['name']=='sys.ask_missing_info')}")
tn = C.Counter(c["name"] for cs in calls for c in cs); P("top gold tools:", tn.most_common(30))
kind_of = {t: v["kind"] for a in atr.values() for t, v in a.items()}; kind_of["sys.ask_missing_info"] = "meta"
nw = C.Counter(sum(1 for c in cs if kind_of.get(c["name"]) == "write") for cs in calls); P("write-calls/sample:", dict(sorted(nw.items())))
ro = sum(1 for cs in calls if cs and all(kind_of.get(c["name"]) == "read" for c in cs)); P(f"read-only non-empty samples: {ro}")
tl = [len(x["audio_text"].split()) for x in tr]; P(f"utterance words mean {S.mean(tl):.1f} median {S.median(tl)} max {max(tl)}")
words = C.Counter(w for x in tr for w in re.findall(r"\w+", x["audio_text"].lower())); P("top words:", words.most_common(40))
neg = sum(1 for x in tr if re.search(r"\b(đừng|không cần|thôi|khỏi|huỷ|hủy)\b", x["audio_text"].lower())); P(f"utterances with negation/cancel words: {neg} ({neg/len(tr):.1%})")
sc = sum(1 for x in tr if re.search(r"\b(à không|ơ không|nhầm|à nhầm)\b", x["audio_text"].lower())); P(f"self-correction markers: {sc}")
dia = C.Counter(w for x in tr for w in re.findall(r"\b(mô|rứa|hông|coi|giùm|nghen|nhen|dzậy|zậy|trỏng|bển|hén)\b", x["audio_text"].lower())); P("dialect markers:", dict(dia))
hit = tot = 0
for x, cs in zip(tr, calls):
    u = x["audio_text"].lower()
    for c in cs:
        if kind_of.get(c["name"]) != "write": continue
        for k, v in c["arguments"].items():
            if isinstance(v, str) and len(v) > 2: tot += 1; hit += v.lower() in u
P(f"write string args appearing verbatim in utterance: {hit}/{tot} ({hit/max(tot,1):.1%})")
def dur(p):
    try:
        with wave.open(p) as w: return w.getnframes() / w.getframerate()
    except Exception: return None
for name, D, sp in (("train", tr, "train"), ("public", pu, "public_test")):
    d = [dur(f"{root}/{sp}/{x['audio']}") for x in D]; d = [v for v in d if v]
    P(f"[{name}] final-turn audio: {len(d)} files, {sum(d)/3600:.2f} h, mean {S.mean(d):.2f}s p95 {sorted(d)[int(.95*len(d))]:.1f}s max {max(d):.1f}s")
P("public samples with all apps in train catalog:", sum(1 for x in pu if all(a in atr for a in inst(x))), "/", len(pu))
P("public samples with >=1 app absent from train:", sum(1 for x in pu if any(a not in atr for a in inst(x))))
sub = jl(f"{root}/public_test/sample_submission.jsonl"); P("sample_submission lines:", len(sub))
