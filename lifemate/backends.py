"""LLM backends. 'vllm' (fast, batched) or 'hf' (transformers, slower), plus 'stub' for dry runs."""
class Stub:
    def __init__(self, reply='{"action":"finish"}'): self.reply = reply
    def generate(self, batch): return [self.reply] * len(batch)

class VLLM:
    def __init__(self, model, max_len=12288, tp=1, gpu_util=0.90, dtype='half', max_tokens=256):
        from vllm import LLM, SamplingParams
        self.llm = LLM(model=model, dtype=dtype, max_model_len=max_len, tensor_parallel_size=tp, gpu_memory_utilization=gpu_util, enable_prefix_caching=True, seed=0)
        self.sp = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    def generate(self, batch):
        outs = self.llm.chat(batch, self.sp, use_tqdm=False, chat_template_kwargs={'enable_thinking': False})
        return [o.outputs[0].text for o in outs]

class HF:
    def __init__(self, model, max_tokens=200, bs=4):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.tok = AutoTokenizer.from_pretrained(model); self.tok.padding_side = 'left'
        self.m = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.float16, device_map='auto').eval()
        self.mt, self.bs, self.torch = max_tokens, bs, torch
    def generate(self, batch):
        out = []
        for i in range(0, len(batch), self.bs):
            chunk = batch[i:i + self.bs]
            txt = [self.tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False) for m in chunk]
            enc = self.tok(txt, return_tensors='pt', padding=True).to(self.m.device)
            with self.torch.no_grad(): g = self.m.generate(**enc, max_new_tokens=self.mt, do_sample=False)
            out += [self.tok.decode(x[enc['input_ids'].shape[1]:], skip_special_tokens=True) for x in g]
        return out

def make(name, model, **kw):
    if name == 'stub': return Stub()
    if name == 'vllm': return VLLM(model, **kw)
    return HF(model)
