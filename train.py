"""
HGSN Trainer v3 — M2 MacBook Air Optimized
============================================
Expected training time: ~45 min for 20 epochs on M2 Air 8GB

Run: python train.py

M2 optimizations:
  ✅ MPS (Metal GPU) — torch.device("mps")
  ✅ float32 (MPS AMP not stable on M2 Air)
  ✅ Gradient accumulation (effective batch 32 with memory of 8)
  ✅ OneCycleLR scheduler for fast convergence
  ✅ Model size trimmed to hidden=128 (fits 8GB comfortably)
  ✅ torch.mps.empty_cache() between epochs
"""

import os, json, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR
from dataset_pipeline import (
    build_loaders, M2_CONFIG, DEVICE, code_to_ast
)


# ── Model components ──────────────────────────────────────────────────────

class ASTEmbedder(nn.Module):
    N = 52
    def __init__(self, d):
        super().__init__()
        self.te   = nn.Embedding(self.N, d)
        self.de   = nn.Embedding(64, d)
        self.proj = nn.Linear(d * 2, d)
        self.norm = nn.LayerNorm(d)
    def forward(self, t, dep):
        return self.norm(self.proj(torch.cat([self.te(t), self.de(dep.clamp(0, 63))], -1)))


class ASTAttention(nn.Module):
    def __init__(self, d, h, drop=0.1):
        super().__init__()
        self.h, self.hd = h, d // h
        self.scale = (d // h) ** -0.5
        self.qkv  = nn.Linear(d, d * 3)
        self.out  = nn.Linear(d, d)
        self.bias = nn.Embedding(32, h)
        self.drop = nn.Dropout(drop)

    def forward(self, x, dist=None, mask=None):
        B, N, D = x.shape
        q, k, v = self.qkv(x).chunk(3, -1)
        def sh(t): return t.view(B, N, self.h, self.hd).transpose(1, 2)
        q, k, v = sh(q), sh(k), sh(v)
        a = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if dist is not None:
            a = a + self.bias(dist.clamp(0, 31)).permute(0, 3, 1, 2)
        if mask is not None:
            a = a.masked_fill(mask[:, None, None, :], float('-inf'))
        a = self.drop(F.softmax(a.float(), dim=-1))
        return self.out(torch.matmul(a, v).transpose(1, 2).reshape(B, N, D))


class TFLayer(nn.Module):
    def __init__(self, d, h, drop=0.1):
        super().__init__()
        self.attn = ASTAttention(d, h, drop)
        self.ff   = nn.Sequential(nn.Linear(d, d*4), nn.GELU(),
                                   nn.Dropout(drop), nn.Linear(d*4, d))
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.drop = nn.Dropout(drop)
    def forward(self, x, dist=None, mask=None):
        x = x + self.drop(self.attn(self.n1(x), dist, mask))
        x = x + self.drop(self.ff(self.n2(x)))
        return x


class HGSN(nn.Module):
    """
    HybridGraphSemanticNet — M2 edition.
    AST Sparse Transformer + Detection/Severity heads.
    """
    def __init__(self, cfg: dict):
        super().__init__()
        d = cfg["hidden_dim"]
        self.embed  = ASTEmbedder(d)
        self.cls    = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.layers = nn.ModuleList([
            TFLayer(d, cfg["num_heads"])
            for _ in range(cfg["num_transformer_layers"])
        ])
        self.norm = nn.LayerNorm(d)
        self.detect = nn.Sequential(
            nn.Linear(d, d // 2), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(d // 2, 1)
        )
        self.severity = nn.Sequential(
            nn.Linear(d, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid()
        )
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, node_type_ids, depths,
                tree_distances=None, padding_mask=None, **_):
        B = node_type_ids.size(0)
        x = self.embed(node_type_ids, depths)           # (B, N, D)
        x = torch.cat([self.cls.expand(B, -1, -1), x], 1)  # (B, N+1, D)

        # ── Extend padding_mask for the CLS token (always unmasked) ──
        if padding_mask is not None:
            cls_pad = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
            padding_mask = torch.cat([cls_pad, padding_mask], 1)  # (B, N+1)

        # ── Extend tree_distances for the CLS token ───────────────────
        # CLS has distance=0 to itself and distance=1 to every AST node
        if tree_distances is not None:
            N = tree_distances.size(1)                  # original N (128)
            # Row of 1s for CLS→nodes, col of 1s for nodes→CLS
            cls_row = torch.ones(B, 1, N,   dtype=torch.long, device=x.device)
            cls_col = torch.ones(B, N+1, 1, dtype=torch.long, device=x.device)
            cls_col[:, 0, :] = 0                        # CLS→CLS distance = 0
            dist_with_cls = torch.cat([cls_row, tree_distances], dim=1)  # (B,N+1,N)
            tree_distances = torch.cat([cls_col, dist_with_cls], dim=2)  # (B,N+1,N+1)

        for layer in self.layers:
            x = layer(x, tree_distances, padding_mask)
        cls = self.norm(x[:, 0])
        return {
            "detect_logit": self.detect(cls).squeeze(-1),
            "severity"    : self.severity(cls).squeeze(-1),
            "embedding"   : cls,
        }

    @property
    def num_params(self): return sum(p.numel() for p in self.parameters())


# ── Loss ──────────────────────────────────────────────────────────────────

class Loss(nn.Module):
    def forward(self, out, batch):
        bce = F.binary_cross_entropy_with_logits(
            out["detect_logit"], batch["bug_label"]
        )
        mse = F.mse_loss(out["severity"], batch["severity"])
        return bce + 0.2 * mse, bce.item()


# ── Helpers ───────────────────────────────────────────────────────────────

def to(batch, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()}


# ── Training loop ─────────────────────────────────────────────────────────

def train(cfg=M2_CONFIG, save_dir="checkpoints"):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device(cfg["device"])

    print(f"\n{'='*54}")
    print(f"  HGSN Training — Apple M2 MacBook Air")
    print(f"  Device  : {device}")
    print(f"  Epochs  : {cfg['num_epochs']}")
    print(f"  Batch   : {cfg['batch_size']} × {cfg['grad_accum_steps']} = "
          f"{cfg['batch_size']*cfg['grad_accum_steps']} effective")
    print(f"{'='*54}\n")

    train_l, val_l, test_l = build_loaders(
        max_nodes   = cfg["max_ast_nodes"],
        batch_size  = cfg["batch_size"],
        num_workers = cfg["num_workers"],
    )

    model = HGSN(cfg).to(device)
    print(f"Model parameters: {model.num_params:,}")

    criterion  = Loss()
    optimizer  = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                                    weight_decay=1e-2, eps=1e-8)
    scheduler  = OneCycleLR(
        optimizer,
        max_lr      = cfg["lr"],
        steps_per_epoch = max(len(train_l) // cfg["grad_accum_steps"], 1),
        epochs      = cfg["num_epochs"],
        pct_start   = 0.1,
    )

    history   = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_acc  = 0.0
    accum     = cfg["grad_accum_steps"]

    for epoch in range(1, cfg["num_epochs"] + 1):
        # ── Train ───────────────────────────────────
        model.train()
        t0 = time.time()
        tloss, tsteps = 0.0, 0
        optimizer.zero_grad()

        for i, batch in enumerate(train_l):
            batch = to(batch, device)
            out   = model(**batch)
            loss, _ = criterion(out, batch)
            (loss / accum).backward()

            if (i + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                optimizer.step(); scheduler.step(); optimizer.zero_grad()

            tloss  += loss.item(); tsteps += 1

        # Free MPS memory
        if device.type == "mps":
            torch.mps.empty_cache()

        # ── Validate ─────────────────────────────────
        model.eval()
        vloss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for batch in val_l:
                batch = to(batch, device)
                out   = model(**batch)
                l, _  = criterion(out, batch)
                vloss += l.item()
                preds  = (torch.sigmoid(out["detect_logit"]) > 0.5).long()
                correct += (preds == batch["bug_label"].long()).sum().item()
                total   += batch["bug_label"].size(0)

        tr_avg = tloss / max(tsteps, 1)
        vl_avg = vloss / max(len(val_l), 1)
        acc    = correct / max(total, 1)
        elapsed = time.time() - t0

        history["train_loss"].append(tr_avg)
        history["val_loss"].append(vl_avg)
        history["val_acc"].append(acc)

        print(f"Epoch {epoch:02d}/{cfg['num_epochs']}  "
              f"train={tr_avg:.4f}  val={vl_avg:.4f}  "
              f"acc={acc:.3f}  {elapsed:.0f}s")

        if acc > best_acc:
            best_acc = acc
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "val_acc": acc, "cfg": cfg},
                       f"{save_dir}/hgsn_best.pt")
            print(f"  ✅ Best saved (acc={acc:.3f})")

        if epoch % 5 == 0:
            torch.save(model.state_dict(), f"{save_dir}/epoch_{epoch}.pt")

    with open(f"{save_dir}/history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n🏁 Done. Best val acc: {best_acc:.3f}")
    print(f"   Checkpoint: {save_dir}/hgsn_best.pt")
    return model, history


# ── Inference ─────────────────────────────────────────────────────────────

@torch.no_grad()
def predict(model: HGSN, code: str, device=None) -> dict:
    """Analyze a single code snippet."""
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.eval()

    feats = code_to_ast(code, max_nodes=128)
    if feats is None:
        return {"error": "unparseable code"}

    out = model(
        node_type_ids  = feats["node_type_ids"].unsqueeze(0).to(device),
        depths         = feats["depths"].unsqueeze(0).to(device),
        tree_distances = feats["tree_distances"].unsqueeze(0).to(device),
        padding_mask   = feats["padding_mask"].unsqueeze(0).to(device),
    )
    prob     = torch.sigmoid(out["detect_logit"]).item()
    severity = out["severity"].item()
    vuln     = prob > 0.5

    return {
        "is_vulnerable"  : vuln,
        "bug_probability": round(prob, 4),
        "severity"       : round(severity, 4),
        "confidence"     : round(prob if vuln else 1 - prob, 4),
        "label"          : "Vulnerable ⚠️" if vuln else "Secure ✅",
    }


def load_model(checkpoint="checkpoints/hgsn_best.pt") -> HGSN | None:
    """Load a saved model."""
    if not os.path.exists(checkpoint):
        print(f"No checkpoint at {checkpoint}. Run train() first.")
        return None
    ckpt  = torch.load(checkpoint, map_location=DEVICE)
    model = HGSN(ckpt.get("cfg", M2_CONFIG)).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"✅ Model loaded (val_acc={ckpt.get('val_acc', '?'):.3f})")
    return model


if __name__ == "__main__":
    model, hist = train()
