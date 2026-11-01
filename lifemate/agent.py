"""Baseline LifeMate agent: transcript -> LLM (JSON action per step) -> ToolEnv. Batched across samples (step-synchronous)."""
import json, re, time
from .common import add_zaic, Retriever

SYSTEM = """You are a Vietnamese voice-assistant agent. You get the TRANSCRIPT of the user's latest spoken request (it may contain speech-recognition mistakes) plus context and the tools of the installed apps. You act ONLY by calling tools, one tool per step.

Reply with exactly ONE JSON object and nothing else:
{"action":"call","name":"<app.tool>","arguments":{...}}
{"action":"ask","tool_name":"<app.tool>","params":["<missing_param>", ...]}   (required info is missing and cannot be found with read tools; this ends the episode)
{"action":"finish"}   (the request is fully done; or NO installed app can handle it - then call nothing at all)

Rules:
- Use only tools listed below; never invent tools or parameters. Follow each tool's parameter names, types and enums exactly.
- Never guess an id. Look ids up with read tools first (contacts.search, *.list_*, *.get_status ...) and copy them exactly from tool results or earlier turns.
- Do read calls before write calls. Do not repeat a call. Do nothing the user did not ask for.
- If the user cancels or negates ("thôi", "khỏi", "đừng"), or no installed app fits the request, reply {"action":"finish"} immediately without any call.
- If a required parameter cannot be determined from the request, context or tool results, use the "ask" action (name the tool you wanted and the missing parameter names).
- Dates/times: ISO 8601 with +07:00, computed from current_time ("mai" = tomorrow, "ba giờ chiều" = 15:00, "5 rưỡi" = 05:30). Numbers: convert spoken numbers ("hai triệu tám" = 2800000, "năm trăm nghìn" = 500000).
- Free-text fields (title, content, label, name, note ...): keep the user's own words.
- If the user corrects themselves ("5 giờ, à không, 5 rưỡi"), use the final value."""

def tool_line(t):
    props = (t.get('parameters') or {}).get('properties', {})
    req = set((t.get('parameters') or {}).get('required', []))
    ps = []
    for k, v in props.items():
        s = f"{k}:{v.get('type', 'any')}"
        if 'enum' in v: s += '=' + '|'.join(map(str, v['enum']))
        if k in req: s += '*'
        if v.get('x-ref'): s += '(id)'
        ps.append(s)
    oneof = (t.get('parameters') or {}).get('x-one-of')
    extra = f" ONE-OF{json.dumps(oneof, ensure_ascii=False)}" if oneof else ''
    return f"- {t['name']} [{t['kind']}] {t.get('description', '')} | params: {', '.join(ps) or '-'}{extra}"

def tools_for(item, catalog_by_app):
    lines = []
    for app in item['context']['installed_apps']:
        for t in catalog_by_app.get(app, []): lines.append(tool_line(t))
    return '\n'.join(lines)

def short(x, n=1500):
    s = json.dumps(x, ensure_ascii=False); return s if len(s) <= n else s[:n] + '…'

def render_history(item, texts):
    out = []
    for i, h in enumerate(item.get('history') or []):
        if h['role'] == 'user':
            out.append('User: ' + texts.get(h['audio'], h.get('audio_text', '')))
        else:
            s = 'Assistant: ' + h.get('text', '')
            for c, r in zip(h.get('tool_calls') or [], h.get('tool_results') or []):
                s += f"\n  [tool] {c['name']} {json.dumps(c['arguments'], ensure_ascii=False)} -> {short(r, 600)}"
            out.append(s)
    return '\n'.join(out) or '(none)'

def parse_action(text):
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.S).strip()
    m = re.search(r'\{.*\}', text, flags=re.S)
    if not m: return None
    s = m.group(0)
    for cand in (s, s[:s.rfind('}') + 1]):
        try: return json.loads(cand)
        except Exception: continue
    return None

class Run:
    def __init__(self, item, zaic, root, split, catalog_by_app, utter, texts, shots_text, max_steps=8):
        self.item, self.split, self.max_steps = item, split, max_steps
        self.env = zaic.ToolEnv(split, item['id'], root=root)
        ctx = item['context']; prof = ctx.get('user_profile', {})
        user = (f"current_time: {ctx['current_time']} ({ctx['timezone']})\nlocation: {ctx.get('location')}\nuser_profile: {json.dumps(prof, ensure_ascii=False)}\n"
                f"installed apps: {', '.join(ctx['installed_apps'])}\n\nTOOLS:\n{tools_for(item, catalog_by_app)}\n"
                f"- sys.ask_missing_info [meta] use via the \"ask\" action\n\nCONVERSATION SO FAR:\n{render_history(item, texts)}\n\n"
                f"CURRENT REQUEST (transcript): {utter}")
        sysmsg = SYSTEM + ('\n\n' + shots_text if shots_text else '')
        self.messages = [{'role': 'system', 'content': sysmsg}, {'role': 'user', 'content': user}]
        self.done, self.steps, self.bad, self.seen, self.t0 = False, 0, 0, set(), time.time()
        self.log = []
    def feed(self, text):
        self.steps += 1; self.messages.append({'role': 'assistant', 'content': text})
        act = parse_action(text)
        if not isinstance(act, dict) or act.get('action') not in ('call', 'ask', 'finish'):
            self.bad += 1; self._tool('Invalid output. Reply with exactly one JSON action object.')
            if self.bad >= 2: self.done = True
            return
        a = act['action']
        if a == 'finish': self.done = True; return
        if a == 'ask':
            tn, ps = act.get('tool_name'), act.get('params') or []
            self.env.call('sys.ask_missing_info', {'tool_name': tn, 'params': ps}); self.done = True; return
        name, args = act.get('name'), act.get('arguments') or {}
        key = (name, json.dumps(args, sort_keys=True, ensure_ascii=False))
        if key in self.seen:
            self._tool('Duplicate call ignored. Continue with the next step or finish.'); self.bad += 1
            if self.bad >= 3: self.done = True
            return
        v = self.env.validate(name, args)
        if not v.get('ok', False):
            self._tool('REJECTED (not executed): ' + str(v.get('error'))); self.bad += 1
            if self.bad >= 3: self.done = True
            return
        self.seen.add(key); res = self.env.call(name, args); self._tool(short(res))
        self.log.append((name, args, res.get('ok')))
    def _tool(self, content): self.messages.append({'role': 'user', 'content': 'TOOL RESULT: ' + content})
    def result(self): return {'tool_calls': self.env.get_tool_calls(), 'state_hash': self.env.get_state_hash(), 'seconds': time.time() - self.t0, 'steps': self.steps}

def build_shots(item, train_rows, retr, k, exclude_ids=()):
    if not k: return ''
    q = item.get('_text', '')
    idx = [i for i in retr.top(q, k + len(exclude_ids)) if train_rows[i]['id'] not in exclude_ids][:k]
    parts = ['EXAMPLES from similar past requests (ids below are illustrative: ALWAYS look real ids up with tools). Format: request -> ordered calls.']
    for i in idx:
        r = train_rows[i]; cs = r['label']['tool_calls']
        seq = '; '.join(f"{c['name']} {json.dumps(c['arguments'], ensure_ascii=False)}" for c in cs) or '(no calls: out of scope / cancelled)'
        parts.append(f"- \"{r['audio_text']}\" -> {seq}")
    return '\n'.join(parts)

def run_batch(items, root, split, catalog, backend, texts_of, train_rows=None, retr=None, shots=0, exclude=None, max_steps=8, log=print):
    """items: sample dicts; texts_of(item)->(utterance, {audio_path: text}). Returns {id: result}."""
    zaic = add_zaic(root, split)
    cba = {a['id']: a['tools'] for a in catalog['apps']}
    runs = []
    for it in items:
        utt, texts = texts_of(it); it['_text'] = utt
        ex = {it['id']} if (exclude is None or exclude) and shots else set()
        runs.append(Run(it, zaic, root, split, cba, utt, texts, build_shots(it, train_rows, retr, shots, ex) if shots else '', max_steps))
    for step in range(max_steps):
        act = [r for r in runs if not r.done and r.steps < max_steps]
        if not act: break
        t0 = time.time(); outs = backend.generate([r.messages for r in act])
        for r, o in zip(act, outs): r.feed(o)
        log(f'step {step}: {len(act)} active, {time.time() - t0:.1f}s')
    return {r.item['id']: r.result() for r in runs}
