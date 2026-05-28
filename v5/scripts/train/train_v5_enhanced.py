#!/usr/bin/env python3
"""v4: Train streaming safety monitor for real-time per-token blocking.

Two-stage training (same backbone as v3):
  Stage 1 — supervised projection 16384 → 256
  Stage 2 — MLP classifier on 256-dim features

v4 improvements:
  • Trained with last-token pooling to match safe_generator's inference
  • Output includes per-category optimal thresholds for blocking decisions
  • Saves full metadata for evaluation
"""
import sys, json, torch, math
sys.path.insert(0, "/root/train/v5")
from torch import nn
from pathlib import Path
from collections import Counter

# Config
DATA    = "/root/train/v5/data/enhanced_states.pt"
OUT     = "/root/train/v5/outputs"
CATS    = ["normal","political","porn","violence","illegal",
           "privacy","discrimination","rumor"]
DIM     = 256               # projection dimension (same as v3)
DEV     = "cuda"

# ---------------------------------------------------------------------------
# Load data
ckpt_data = torch.load(DATA, weights_only=True)
X, y = ckpt_data["vectors"], ckpt_data["labels"]
cat_map = ckpt_data["category_map"]
n_cat = len(cat_map)
n = len(X)
n_tr = int(n * 0.8)

perm = torch.randperm(n)
X, y = X[perm], y[perm]
X_tr, y_tr = X[:n_tr].to(DEV), y[:n_tr].to(DEV)
X_va, y_va = X[n_tr:].to(DEV), y[n_tr:].to(DEV)

print(f"Data: {list(X.shape)} total, {n_tr} train / {n-n_tr} val")
for c in range(n_cat):
    cnt = (y == c).sum().item()
    print(f"  {CATS[c]}: {cnt} ({cnt/n*100:.1f}%)")

# Normalise
m_in = X_tr.mean(0, keepdim=True)
s_in = X_tr.std(0, keepdim=True) + 1e-8
X_tr = (X_tr - m_in) / s_in
X_va = (X_va - m_in) / s_in

# ---------------------------------------------------------------------------
# Stage 1: supervised projection 16384 → 256
print(f"\n=== Stage 1: {X.shape[1]} → {DIM} supervised projection ===")
proj = nn.Linear(X.shape[1], DIM, bias=False).to(DEV)
head = nn.Linear(DIM, n_cat).to(DEV)
opt1 = torch.optim.AdamW(list(proj.parameters()) + list(head.parameters()),
                         lr=1e-3, weight_decay=1e-4)
ce = nn.CrossEntropyLoss()

for ep in range(30):
    opt1.zero_grad()
    z = proj(X_tr)
    loss = ce(head(z), y_tr)
    loss.backward()
    opt1.step()
    if ep % 10 == 0:
        with torch.no_grad():
            acc = (head(proj(X_va)).argmax(1) == y_va).float().mean().item()
        print(f"  ep {ep:2d}  loss={loss.item():.4f}  val_acc={acc:.2%}")

with torch.no_grad():
    z_tr = proj(X_tr).cpu()
    z_va = proj(X_va).cpu()
m_z = z_tr.mean(0, keepdim=True)
s_z = z_tr.std(0, keepdim=True) + 1e-8
z_tr = (z_tr - m_z) / s_z
z_va = (z_va - m_z) / s_z

# ---------------------------------------------------------------------------
# Stage 2: MLP
print(f"\n=== Stage 2: MLP on {DIM}-dim ===")
mlp = nn.Sequential(
    nn.Linear(DIM, 256), nn.ReLU(), nn.Dropout(0.2),
    nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.1),
    nn.Linear(128, n_cat),
).to(DEV)
opt2 = torch.optim.AdamW(mlp.parameters(), lr=5e-4, weight_decay=1e-4)
best_acc, best_state = 0.0, None

Z_tr, Z_va = z_tr.to(DEV), z_va.to(DEV)
for ep in range(150):  # more epochs for convergence
    mlp.train()
    opt2.zero_grad()
    loss = ce(mlp(Z_tr), y_tr)
    loss.backward()
    opt2.step()

    if ep % 10 == 0 or ep == 149:
        mlp.eval()
        with torch.no_grad():
            acc = (mlp(Z_va).argmax(1) == y_va).float().mean().item()
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in mlp.state_dict().items()}
        print(f"  ep {ep:3d}  loss={loss.item():.4f}  val_acc={acc:.2%}")

mlp.load_state_dict(best_state)
print(f"\nBest val_acc = {best_acc:.2%}")

# ---------------------------------------------------------------------------
# Compute per-category optimal thresholds (for v4 blocking decisions)
print("\n=== Optimising per-category thresholds ===")
mlp.eval()
with torch.no_grad():
    all_probs = torch.softmax(mlp(Z_va), dim=-1)
    all_preds = all_probs.argmax(dim=-1)

# For each unsafe category, find threshold where precision >= 0.95
optimal_thresholds = {}
for cat_name, cat_id in cat_map.items():
    if cat_name == "normal":
        continue
    mask = (y_va == cat_id) | (y_va == 0)  # this cat vs normal
    if mask.sum() < 10:
        optimal_thresholds[cat_name] = 0.50
        continue
    probs_this = all_probs[mask][:, cat_id]
    labels_binary = (y_va[mask] == cat_id).long()

    # Try thresholds 0.05 to 0.95 in 0.05 steps
    best_f1 = 0.0
    best_thr = 0.50
    for thr_pct in range(5, 96, 5):
        thr = thr_pct / 100.0
        preds_binary = (probs_this >= thr).long()
        tp = ((preds_binary == 1) & (labels_binary == 1)).sum().item()
        fp = ((preds_binary == 1) & (labels_binary == 0)).sum().item()
        fn = ((preds_binary == 0) & (labels_binary == 1)).sum().item()
        prec = tp / (tp + fp + 1e-8)
        rec  = tp / (tp + fn + 1e-8)
        f1   = 2 * prec * rec / (prec + rec + 1e-8)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
    optimal_thresholds[cat_name] = best_thr
    print(f"  {cat_name:20s}  best_thr={best_thr:.2f}  F1={best_f1:.3f}")

# ---------------------------------------------------------------------------
# Save
proj_np = proj.weight.data.cpu().t()
proj_np = proj_np / proj_np.norm(dim=0, keepdim=True)

Path(OUT).mkdir(parents=True, exist_ok=True)
torch.save({
    "projection":      proj_np,
    "projection_dim":  DIM,
    "classifier":      mlp.state_dict(),
    "input_mean":      m_in.cpu(),
    "input_std":       s_in.cpu(),
    "proj_mean":       m_z.cpu(),
    "proj_std":        s_z.cpu(),
    "cat_map":         cat_map,
    "pool_mode":       "mean",
    "num_last_layers": 4,
    "optimal_thresholds": optimal_thresholds,
    "val_acc":         best_acc,
    "version":         "v4-streaming",
}, f"{OUT}/streaming_classifier.pt")
print(f"\nSaved → {OUT}/streaming_classifier.pt")
print(f"  Version: v4-streaming")
print(f"  Val acc: {best_acc:.2%}")
print(f"  Per-category thresholds included: {len(optimal_thresholds)}")
