#!/usr/bin/env python3
"""Extract hidden states for newly appended samples and merge into enhanced_states.pt."""
import sys, json, torch
sys.path.insert(0, '/root/train')
from pathlib import Path

MODEL_PATH = "/root/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a"
OLD_DATA = "/root/train/v5/data/enhanced_states.pt"
NEW_JSONL = "/root/train/v3/data/new_samples.jsonl"
OUT_DIR = "/root/train/v5/data"
N_LAYERS = 4
BATCH = 8
CAT_MAP = {"normal":0,"political":1,"porn":2,"violence":3,"illegal":4,"privacy":5,"discrimination":6,"rumor":7}

from transformers import AutoTokenizer, AutoModelForCausalLM

print("Loading model...")
tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto",
    trust_remote_code=True).eval()

# Load existing data
# The original extract_and_merge.py processed the first 549 lines of new_samples.jsonl
# Now we have 573 lines, so process lines 549 onward
PREV_PROCESSED_LINES = 585
lines = Path(NEW_JSONL).read_text(encoding="utf-8").strip().split("\n")
new_lines = lines[PREV_PROCESSED_LINES:]  # everything beyond what was already processed

# Load existing data for merging
old = torch.load(OLD_DATA, weights_only=True)
V_old = old["vectors"]
L_old = old["labels"]
original_count = len(V_old)
print(f"Existing vectors: {list(V_old.shape)}")
print(f"New lines to process: {len(new_lines)}")

if not new_lines:
    print("No new samples to extract. Done.")
    sys.exit(0)

all_vec, all_lbl = [], []
for i in range(0, len(new_lines), BATCH):
    batch = new_lines[i:i+BATCH]
    texts = [json.loads(l)["input"] for l in batch]
    labels = [CAT_MAP.get(json.loads(l).get("category","normal"), 0) for l in batch]
    inp = tok(texts, padding=True, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model(**inp, use_cache=False, output_hidden_states=True)
    hs = out.hidden_states
    pooled = torch.cat([h.float().mean(dim=1) for h in hs[-N_LAYERS:]], dim=-1)
    all_vec.append(pooled.cpu())
    all_lbl.extend(labels)
    if (i // BATCH) % 5 == 0:
        print(f"  [{i}/{len(new_lines)}]")

V_new = torch.cat(all_vec, dim=0)
L_new = torch.tensor(all_lbl)
print(f"New vectors: {list(V_new.shape)}")

# Append and save
V_all = torch.cat([V_old, V_new], dim=0)
L_all = torch.cat([L_old, L_new], dim=0)
print(f"Combined: {list(V_all.shape)}")

Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
torch.save({
    "vectors": V_all, "labels": L_all,
    "category_map": CAT_MAP, "pool_mode": "mean", "num_last_layers": N_LAYERS,
    "version": "v5-enhanced",
    "original_count": original_count,
    "added_count": len(V_new),
}, f"{OUT_DIR}/enhanced_states.pt")
print(f"Saved -> {OUT_DIR}/enhanced_states.pt")

from collections import Counter
cnt = Counter()
for lbl in L_all.tolist():
    for cat, i in CAT_MAP.items():
        if i == lbl: cnt[cat] += 1; break
print("Combined distribution:")
for cat, n in cnt.most_common():
    print(f"  {cat}: {n}")
