"""Sampled Text2MD rollouts for DPO pairs; executed inside the pinned Docker runtime (same model load, GDN FP32 patch,
system prompt and chat template as infer.py; only the sampling differs: n samples per prompt, temperature / top_p, one
fixed seed). Writes one row per sample to <output>/rollouts.jsonl; resumable (prompts with all n samples are skipped).
"""
import argparse
import hashlib
import json
import re
import time
from pathlib import Path

from infer import HEADINGS, inputs, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--n', type=int, default=2)
    p.add_argument('--temperature', type=float, default=0.7)
    p.add_argument('--top-p', type=float, default=0.95)
    p.add_argument('--seed', type=int, default=1793850504)
    p.add_argument('--max-tokens', type=int, default=4096)
    p.add_argument('--max-num-seqs', type=int, default=256)
    p.add_argument('--chunk', type=int, default=512, help='prompts per generate() call (results are flushed after each)')
    p.add_argument('--gpu-memory-utilization', type=float, default=0.85)
    args = p.parse_args()
    rows = inputs(args.input)
    args.output.mkdir(parents=True, exist_ok=True)
    out_path = args.output / 'rollouts.jsonl'
    seen = {}
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            r = json.loads(line); seen[r['id']] = seen.get(r['id'], 0) + 1
    todo = [r for r in rows if seen.get(r['id'], 0) < args.n]
    import torch
    from safetensors import safe_open
    from vllm import LLM, SamplingParams
    model = args.model_dir / 'hf'
    metadata = json.loads((args.model_dir / 'model.json').read_text())
    expected = {}
    weights = json.loads((model / 'model.safetensors.index.json').read_text())['weight_map']
    for key, shard in weights.items():
        if key.endswith(('.linear_attn.A_log', '.linear_attn.dt_bias')) and '.mtp.' not in key:
            with safe_open(model / shard, framework='pt', device='cpu') as f:
                tensor = f.get_tensor(key).contiguous()
                if tensor.dtype != torch.float32:
                    raise RuntimeError('Exported GDN state is not FP32: ' + key)
                match = re.search(r'\.layers\.(\d+)\.linear_attn\.(A_log|dt_bias)$', key)
                expected[(int(match[1]), match[2])] = hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest()
    llm = LLM(model=str(model), tokenizer=str(model), dtype='bfloat16', tensor_parallel_size=1,
        language_model_only=True, max_model_len=16384, max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization, attention_backend='FLASH_ATTN',
        enable_prefix_caching=False,
        worker_extension_cls='scripts.t2i_eval.vllm_gdn_fp32_runtime.GDNFP32WorkerExtension')
    receipts = llm.collective_rpc('attest_gdn_state_fp32', args=(len(expected),))
    for receipt in receipts:
        actual = {(r['layer_index'], r['state_name']): r['payload_sha256'] for r in receipt['tensors']}
        if actual != expected:
            raise RuntimeError('Runtime GDN tensors do not match exported FP32 bytes')
    tokenizer = llm.get_tokenizer()
    t0, done, ntok = time.time(), 0, 0
    print(json.dumps(dict(todo=len(todo), total=len(rows), loaded_s=round(time.time() - t0, 1))), flush=True)
    with out_path.open('a') as out:
        for start in range(0, len(todo), args.chunk):
            batch = todo[start:start + args.chunk]
            prompts = [tokenizer.apply_chat_template(r['messages'], tokenize=False, add_generation_prompt=True, enable_thinking=False) for r in batch]
            params = []
            for row, prompt in zip(batch, prompts):
                if '<think>\n\n</think>' not in prompt:
                    raise RuntimeError('Non-thinking template prefix is missing')
                budget = min(args.max_tokens, 16384 - len(tokenizer.encode(prompt, add_special_tokens=False)) - 1)
                params.append(SamplingParams(n=args.n, temperature=args.temperature, top_p=args.top_p, seed=args.seed,
                                             max_tokens=budget, repetition_penalty=1.0))
            answers = llm.generate(prompts, params, use_tqdm=False)
            for row, answer in zip(batch, answers):
                for k, g in enumerate(answer.outputs):
                    md = g.text.strip()
                    valid = g.finish_reason == 'stop' and re.findall(r'^## .+$', md, re.M) == HEADINGS
                    out.write(json.dumps(dict(id=row['id'], k=k, prompt=row['prompt'], aspect_ratio=row['aspect_ratio'], markdown=md,
                                              finish_reason=g.finish_reason, format_valid=valid, output_tokens=len(g.token_ids),
                                              model_id=metadata['id']), ensure_ascii=False) + '\n')
                    ntok += len(g.token_ids)
            out.flush(); done += len(batch); el = time.time() - t0
            print(json.dumps(dict(done=done, todo=len(todo), elapsed_s=round(el), rollouts_per_min=round(done * args.n / el * 60), out_tok_per_s=round(ntok / el))), flush=True)
    write(args.output / 'summary.json', dict(model=metadata, n=args.n, temperature=args.temperature, top_p=args.top_p, seed=args.seed,
                                             max_tokens=args.max_tokens, prompts=len(rows), elapsed_s=round(time.time() - t0)))


if __name__ == '__main__':
    main()
