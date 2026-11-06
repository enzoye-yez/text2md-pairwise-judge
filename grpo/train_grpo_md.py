"""GRPO for the 4B Text2MD model with the pairwise md judge as reward (Krea-3 style pivot), adapted from
scripts/grpo_dpo/grpo/train_grpo_ml.py (branch codex/text2json-4b-b300-zane50m-752): TRL GRPOTrainer, vLLM colocate,
DAPO loss, group-scaled rewards, importance-sampling correction. No span advantage.

Reward of each rollout (one scalar per rollout, group = the G rollouts of one prompt):
  format gate   not the four sections in order, or truncated / no EOS  -> -1 (not judged)
  preference    one valid rollout of the group is the pivot; every other valid rollout is compared with it once by terra v1b
                (A/B order by hash): P(this better) - P(pivot better) in [-1, 1]; the pivot itself gets 0
  length        ratio = tokens / ref_ntok (ref = mean SFT sampled length for the prompt):
                - len_pen * max(0, (len_floor - ratio) / len_floor) - len_ceil_pen * min(1, max(0, (ratio - len_ceil) / len_ceil))
  repetition    8-gram repeat share of the completion ids above rep_thr: - rep_pen * (rep - rep_thr) / (1 - rep_thr)
  text          each quoted request string missing from the md: - text_pen, at most text_cap; '"X" or "Y"' / 'or similar': - hedge_pen each
  gate          groups whose reward std < std_gate get all rewards set to the group mean (zero advantage: no update from judge noise)
    accelerate launch --config_file zero2.yaml train_grpo_md.py --model M --data D --out OUT [options]
"""
import argparse, hashlib, json, os, random, re, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, wait

ap = argparse.ArgumentParser()
ap.add_argument('--model', required=True); ap.add_argument('--data', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--steps', type=int, default=150); ap.add_argument('--prompts-per-step', type=int, default=32)
ap.add_argument('--group', type=int, default=8); ap.add_argument('--micro', type=int, default=2)
ap.add_argument('--seed', type=int, default=1105); ap.add_argument('--lr', type=float, default=3e-6)
ap.add_argument('--temperature', type=float, default=1.0); ap.add_argument('--beta', type=float, default=0.02)
ap.add_argument('--max-prompt-tokens', type=int, default=2048); ap.add_argument('--max-completion', type=int, default=2048)
ap.add_argument('--vllm-mem', type=float, default=0.30); ap.add_argument('--save-steps', type=int, default=25)
ap.add_argument('--judge-deadline', type=float, default=240); ap.add_argument('--judge-threads', type=int, default=64)
ap.add_argument('--len-floor', type=float, default=0.85); ap.add_argument('--len-pen', type=float, default=0.5)
ap.add_argument('--len-ceil', type=float, default=1.2); ap.add_argument('--len-ceil-pen', type=float, default=1.0)
ap.add_argument('--rep-pen', type=float, default=1.0); ap.add_argument('--rep-thr', type=float, default=0.2)
ap.add_argument('--text-pen', type=float, default=0.3); ap.add_argument('--text-cap', type=float, default=0.6)
ap.add_argument('--hedge-pen', type=float, default=0.15); ap.add_argument('--std-gate', type=float, default=0.1)
a = ap.parse_args()

import httpx, torch
from datasets import Dataset
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
import trl.generation.vllm_generation as VG
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root: pairwise_judge.py
import pairwise_judge as PJ  # noqa: E402  terra v1b prompt / schema

# vLLM: the export's config is Qwen3_5ForConditionalGeneration without vision weights -> text-only engine (as Aran's infer.py)
_LLM = VG.LLM


class _TextLLM(_LLM):
    def __init__(self, *args, **kw):
        kw.setdefault('language_model_only', True)
        super().__init__(*args, **kw)


VG.LLM = _TextLLM
# the policy is Qwen3_5ForCausalLM (parameter names model.layers.*); vLLM's model expects the export's model.language_model.*
_iter = VG.VLLMGeneration._iter_named_params


def _iter_named_params(self):
    for name, p in _iter(self):
        if name.startswith('model.') and not name.startswith('model.language_model.'):
            name = 'model.language_model.' + name[len('model.'):]
        if name == 'lm_head.weight':
            continue  # tied to the embedding
        yield name, p


VG.VLLMGeneration._iter_named_params = _iter_named_params

RANK = int(os.environ.get('RANK', 0)); WORLD = int(os.environ.get('WORLD_SIZE', 1))
os.makedirs(a.out, exist_ok=True)
tok = AutoTokenizer.from_pretrained(a.model)
HEAD = ['## High-level Description', '## Style Description', '## Background', '## Semantic Structure']

rows = []
for l in open(a.data):
    r = json.loads(l); m = r['messages']
    n = len(tok(tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False), add_special_tokens=False)['input_ids'])
    if n > a.max_prompt_tokens:
        continue
    u = json.loads(m[1]['content'])
    rows.append({'prompt': m, 'key': r['id'], 'req': u['prompt'], 'ar': json.dumps(u.get('aspect_ratio')), 'ref_ntok': float(r['ref_ntok'])})
if RANK == 0:
    print(f'[grpo] prompts kept {len(rows)}', flush=True)
ds = Dataset.from_list(rows)
LOG = open(f'{a.out}/rollouts.rank{RANK}.jsonl', 'a'); LOCK = threading.Lock()
POOL = ThreadPoolExecutor(a.judge_threads)
KEY = os.environ['OPENAI_API_KEY']
_local = threading.local()


def client():
    if not hasattr(_local, 'c'):
        _local.c = httpx.Client(timeout=httpx.Timeout(120, connect=10), headers={'Authorization': f'Bearer {KEY}'})
    return _local.c


def judge_pair(req, ar, md_i, md_p, salt):
    """terra v1b: margin = P(md_i better) - P(md_p better); None on failure."""
    order = int(hashlib.md5(salt.encode()).hexdigest(), 16) % 2  # 0: md_i shown as A
    A, B = (md_i, md_p) if order == 0 else (md_p, md_i)
    body = {'model': PJ.MODEL, 'reasoning': {'effort': 'low'},
            'input': [{'role': 'developer', 'content': PJ.SYSTEM}, {'role': 'user', 'content': PJ.user_text(req, ar, A, B)}],
            'text': {'format': {'type': 'json_schema', 'name': 'pair', 'strict': True, 'schema': PJ.SCHEMA}}}
    for att in range(6):
        try:
            r = client().post('https://api.openai.com/v1/responses', json=body)
            if r.status_code == 200:
                b = r.json(); t = json.loads(''.join(c.get('text', '') for o in b['output'] if o.get('type') == 'message' for c in o['content']))
                p = PJ.norm({'A_better': t['p_A_better'], 'B_better': t['p_B_better'], 'similar': t['p_similar']})
                m = p['A_better'] - p['B_better']
                return {'margin': m if order == 0 else -m, 'reason': t['reason'], 'usage': b.get('usage')}
            if r.status_code not in (408, 409, 429, 500, 502, 503, 504):
                return None
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            pass
        time.sleep(min(30, 2 ** att + random.random()))
    return None


def valid(text, ids):
    trunc = len(ids) >= a.max_completion or not ids or ids[-1] != tok.eos_token_id
    md = re.sub(r'^\s*Aspect ratio:\s*\d+\s*:\s*\d+\s*\n', '', text.strip())
    return not trunc and re.findall(r'^## .+$', md, re.M) == HEAD


def rule_terms(text, ids, req, ref):
    pen = 0.0; info = {}
    ratio = len(ids) / ref if ref > 0 else 1.0
    pen += a.len_pen * max(0.0, (a.len_floor - ratio) / a.len_floor)
    pen += a.len_ceil_pen * min(1.0, max(0.0, (ratio - a.len_ceil) / a.len_ceil))
    if len(ids) > 64:
        grams = [tuple(ids[j:j + 8]) for j in range(len(ids) - 7)]; rep = 1.0 - len(set(grams)) / len(grams)
        if rep > a.rep_thr:
            pen += a.rep_pen * (rep - a.rep_thr) / (1.0 - a.rep_thr)
        info['rep'] = round(rep, 3)
    quoted = [q for q in re.findall(r'["“”]([^"“”]{2,80})["“”]', req)]
    miss = [q for q in quoted if q.lower() not in text.lower()]
    pen += min(a.text_cap, a.text_pen * len(miss))
    hedge = len(re.findall(r'"[^"\n]{1,60}"\s+or\s+"[^"\n]{1,60}"', text)) + len(re.findall(r'\bor similar\b', text, re.I))
    pen += a.hedge_pen * hedge
    info.update(ratio=round(ratio, 3), miss=len(miss), hedge=hedge)
    return pen, info


def md_reward(prompts, completions, completion_ids, key, req, ar, ref_ntok, trainer_state=None, **kw):
    texts = [c[0]['content'] if isinstance(c, list) else c for c in completions]
    step = trainer_state.global_step if trainer_state is not None else -1
    groups = {}
    for i, k in enumerate(key):
        groups.setdefault(k, []).append(i)
    ok = [valid(t, ids) for t, ids in zip(texts, completion_ids)]
    pivot, jobs = {}, {}
    rng = random.Random(f'{a.seed}|{step}|{RANK}')
    for k, idx in groups.items():
        cand = [i for i in idx if ok[i]]
        if len(cand) < 2:
            continue
        p = rng.choice(cand); pivot[k] = p
        for i in cand:
            if i != p:
                jobs[i] = POOL.submit(judge_pair, req[i], json.loads(ar[i]), texts[i], texts[p], f'{key[i]}|{step}|{i}')
    t0 = time.time(); wait(list(jobs.values()), timeout=a.judge_deadline)
    res = {i: (f.result() if f.done() else None) for i, f in jobs.items()}
    raw = [None] * len(texts); infos = [None] * len(texts)
    for i in range(len(texts)):
        if not ok[i]:
            raw[i] = -1.0; continue
        k = key[i]
        if k not in pivot:
            pref = 0.0
        elif i == pivot[k]:
            pref = 0.0
        else:
            pref = res[i]['margin'] if res.get(i) else 0.0
        pen, infos[i] = rule_terms(texts[i], completion_ids[i], req[i], ref_ntok[i])
        raw[i] = pref - pen
    out = list(raw); gated = 0
    for k, idx in groups.items():
        vals = [raw[i] for i in idx]; mu = sum(vals) / len(vals)
        sd = (sum((v - mu) ** 2 for v in vals) / len(vals)) ** .5
        if len(idx) > 1 and sd < a.std_gate:
            for i in idx:
                out[i] = mu
            gated += 1
    with LOCK:
        for i in range(len(texts)):
            j = res.get(i)
            LOG.write(json.dumps({'step': step, 'key': key[i], 'valid': ok[i], 'pivot': pivot.get(key[i]) == i, 'ntok': len(completion_ids[i]),
                                  'margin': j['margin'] if j else None, 'reason': j['reason'] if j else None, 'usage': j['usage'] if j else None,
                                  'rules': infos[i], 'raw': raw[i], 'reward': out[i], 'text': texts[i]}, ensure_ascii=False) + '\n')
        LOG.flush()
    if RANK == 0:
        n_err = sum(1 for i in jobs if res.get(i) is None)
        print(f'[grpo] step {step} rollouts {len(texts)} invalid {ok.count(False)} judged {len(jobs)} failed {n_err} '
              f'gated groups {gated}/{len(groups)} judge {time.time() - t0:.0f}s mean raw {sum(raw) / len(raw):.3f}', flush=True)
    return out


def load_policy(path):  # bf16 with the GDN state (A_log / dt_bias) restored to the export's fp32
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16)
    params = dict(model.named_parameters())
    for key, shard in json.load(open(os.path.join(path, 'model.safetensors.index.json')))['weight_map'].items():
        if key.endswith(('.linear_attn.A_log', '.linear_attn.dt_bias')) and '.mtp.' not in key:
            with safe_open(os.path.join(path, shard), framework='pt', device='cpu') as f:
                params[re.sub(r'^model\.language_model\.', 'model.', key)].data = f.get_tensor(key).clone()
    model.name_or_path = path; model.config._name_or_path = path
    return model


ga = a.prompts_per_step * a.group // (a.micro * WORLD)
cfg = GRPOConfig(
    output_dir=a.out, max_steps=a.steps, learning_rate=a.lr, lr_scheduler_type='constant_with_warmup', warmup_steps=5,
    per_device_train_batch_size=a.micro, gradient_accumulation_steps=ga, steps_per_generation=ga, num_iterations=1,
    num_generations=a.group, max_completion_length=a.max_completion, temperature=a.temperature, top_p=1.0,
    beta=a.beta, epsilon=0.2, epsilon_high=0.28, loss_type='dapo', scale_rewards='group', mask_truncated_completions=False,
    use_vllm=True, vllm_mode='colocate', vllm_gpu_memory_utilization=a.vllm_mem, vllm_enable_sleep_mode=True,
    vllm_max_model_length=a.max_prompt_tokens + a.max_completion + 64, vllm_importance_sampling_correction=True,
    chat_template_kwargs={'enable_thinking': False},
    bf16=True, gradient_checkpointing=True, max_grad_norm=1.0, adam_beta2=0.999, weight_decay=0.0,
    logging_steps=1, save_steps=a.save_steps, save_only_model=True, report_to='none', seed=a.seed,
    shuffle_dataset=True, log_completions=False, dataloader_num_workers=0,
)
trainer = GRPOTrainer(model=load_policy(a.model), reward_funcs=[md_reward], args=cfg, train_dataset=ds, processing_class=tok)
trainer.train()
trainer.save_model(f'{a.out}/final')
