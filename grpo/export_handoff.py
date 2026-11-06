"""Turn a DPO output (Qwen3_5ForCausalLM save: text weights only, GDN A_log / dt_bias in fp32) back into the layout of Aran's
handoff checkpoint, so that his inference runtime (code/inference: Qwen3_5ForConditionalGeneration config, language_model_only,
FP32-GDN attestation) loads it unchanged: original config / tokenizer / chat template / processor files, trained text tensors,
and the original (untrained, unused at inference) MTP tensors copied over. Writes <dst>/hf/ and <dst>/model.json.
usage: export_handoff.py SRC_FINAL_DIR BASE_MODEL_DIR DST_DIR   (BASE_MODEL_DIR contains hf/ and model.json)
"""
import json, os, shutil, sys
import torch
from safetensors import safe_open
from safetensors.torch import save_file

src, base, dst = sys.argv[1:4]
bh, dh = os.path.join(base, "hf"), os.path.join(dst, "hf")
os.makedirs(dh, exist_ok=False)
bmap = json.load(open(f"{bh}/model.safetensors.index.json"))["weight_map"]
new = {}
for f in sorted(x for x in os.listdir(src) if x.endswith(".safetensors")):
    with safe_open(os.path.join(src, f), "pt") as s:
        for k in s.keys():
            new[k] = s.get_tensor(k)
if "lm_head.weight" in new and "lm_head.weight" not in bmap:  # tie_word_embeddings: the base export stores only the embedding
    assert torch.equal(new["lm_head.weight"], new["model.language_model.embed_tokens.weight"]), "untied lm_head differs"
    del new["lm_head.weight"]
missing = [k for k in bmap if k not in new]
assert all(".mtp." in k or k.startswith("mtp.") for k in missing), [k for k in missing if "mtp" not in k][:5]
assert not [k for k in new if k not in bmap], "unexpected tensors"
for k in missing:
    with safe_open(os.path.join(bh, bmap[k]), "pt") as s:
        new[k] = s.get_tensor(k)
for k, t in new.items():  # keep each tensor's exported dtype (bf16, and fp32 for the GDN state / anything stored in fp32)
    with safe_open(os.path.join(bh, bmap[k]), "pt") as s:
        want = s.get_slice(k).get_dtype()
    if {"BF16": torch.bfloat16, "F32": torch.float32}[want] != t.dtype:
        new[k] = t.to({"BF16": torch.bfloat16, "F32": torch.float32}[want])
shards, cur, size, wmap = [], {}, 0, {}
for k in bmap:
    t = new[k].contiguous(); cur[k] = t; size += t.numel() * t.element_size()
    if size > 4.5e9:
        shards.append(cur); cur, size = {}, 0
shards.append(cur)
total = 0
for i, sh in enumerate(shards):
    name = f"model-{i + 1:05d}-of-{len(shards):05d}.safetensors"
    save_file(sh, os.path.join(dh, name), metadata={"format": "pt"})
    for k, t in sh.items():
        wmap[k] = name; total += t.numel() * t.element_size()
json.dump({"metadata": {"total_size": total}, "weight_map": wmap}, open(f"{dh}/model.safetensors.index.json", "w"), indent=2)
for f in os.listdir(bh):
    if not f.endswith(".safetensors") and f != "model.safetensors.index.json" and os.path.isfile(os.path.join(bh, f)):
        shutil.copy(os.path.join(bh, f), dh)
meta = json.load(open(os.path.join(base, "model.json")))
meta = {**meta, "id": os.path.basename(os.path.normpath(dst)), "base_model_id": meta.get("id"),
        "note": "DPO fine-tune of the base model; text weights trained, MTP tensors copied from the base"}
json.dump(meta, open(os.path.join(dst, "model.json"), "w"), indent=2, ensure_ascii=False)
print(f"{len(wmap)} tensors ({len(missing)} MTP copied from base) in {len(shards)} shards -> {dh}")
