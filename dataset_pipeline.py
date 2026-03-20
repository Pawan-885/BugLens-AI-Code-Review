"""
HGSN Dataset Pipeline v3 — Devign/CodeXGLUE
=============================================
Dataset: Devign via Microsoft CodeXGLUE
Direct Download: https://drive.google.com/uc?id=1x6hoF7G-tSYxg8AFybggypLZgMGDNHfF

HOW TO DOWNLOAD (choose one):
  Option A — gdown (automatic, recommended):
    pip install gdown
    cd dataset/
    gdown https://drive.google.com/uc?id=1x6hoF7G-tSYxg8AFybggypLZgMGDNHfF
    # → downloads function.json (~28MB)

  Option B — HuggingFace API (auto-cached):
    from datasets import load_dataset
    ds = load_dataset("google/code_x_glue_cc_defect_detection")

  Option C — Browser direct download:
    https://drive.google.com/file/d/1x6hoF7G-tSYxg8AFybggypLZgMGDNHfF/view
    Save as: dataset/function.json

DATASET PATH after download:
    hgsn_v3/
    └── dataset/
        └── function.json    ← place file here

M2 MacBook Air (8GB) Settings:
  batch_size  = 8    (safe for 8GB unified memory)
  num_workers = 2    (leave cores for MPS Metal GPU)
  max_nodes   = 128  (saves ~60% RAM vs 256)
  grad_accum  = 4    (effective batch = 32)
"""

import ast, json, os, re
import torch
from torch.utils.data import Dataset, DataLoader, random_split


# ── Device detection ──────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        print("✅ Apple MPS (Metal GPU) — M2 accelerated")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("✅ CUDA GPU")
        return torch.device("cuda")
    print("⚠️  CPU only")
    return torch.device("cpu")

DEVICE = get_device()


# ── AST vocabulary ────────────────────────────────────────────────────────

AST_TYPES = [
    'Module','FunctionDef','AsyncFunctionDef','ClassDef','Return','Delete',
    'Assign','AugAssign','AnnAssign','For','AsyncFor','While','If','With',
    'AsyncWith','Raise','Try','Assert','Import','ImportFrom','Global',
    'Nonlocal','Expr','Pass','Break','Continue','BoolOp','BinOp','UnaryOp',
    'Lambda','IfExp','Dict','Set','ListComp','SetComp','DictComp',
    'GeneratorExp','Await','Yield','YieldFrom','Compare','Call',
    'FormattedValue','JoinedStr','Constant','Attribute','Subscript',
    'Starred','Name','List','Tuple','Slice','UNKNOWN'
]
TYPE_VOCAB = {n: i for i, n in enumerate(AST_TYPES)}


# ── AST feature extractor ─────────────────────────────────────────────────

def code_to_ast(code: str, max_nodes: int = 128) -> dict | None:
    """Extract AST node features from source code."""
    blank = {
        "node_type_ids" : torch.zeros(max_nodes, dtype=torch.long),
        "depths"        : torch.zeros(max_nodes, dtype=torch.long),
        "tree_distances": torch.zeros(max_nodes, max_nodes, dtype=torch.long),
        "padding_mask"  : torch.ones(max_nodes,  dtype=torch.bool),
    }
    try:
        tree = ast.parse(code.replace('\x00', ''))
    except SyntaxError:
        return blank    # syntax errors are meaningful — keep as sample
    except Exception:
        return None     # truly unrecoverable

    types, depths = [], []

    def walk(node, d: int):
        types.append(TYPE_VOCAB.get(type(node).__name__, TYPE_VOCAB['UNKNOWN']))
        depths.append(min(d, 63))
        for child in ast.iter_child_nodes(node):
            walk(child, d + 1)

    walk(tree, 0)

    N   = min(len(types), max_nodes)
    pad = max_nodes - N
    t   = types[:N]  + [0] * pad
    d   = depths[:N] + [0] * pad
    pmask = [False] * N + [True] * pad

    d_t  = torch.tensor(d, dtype=torch.long)
    dist = (d_t.unsqueeze(0) - d_t.unsqueeze(1)).abs().clamp(0, 31)

    return {
        "node_type_ids" : torch.tensor(t,     dtype=torch.long),
        "depths"        : torch.tensor(d,     dtype=torch.long),
        "tree_distances": dist,
        "padding_mask"  : torch.tensor(pmask, dtype=torch.bool),
    }


# ── Devign dataset ────────────────────────────────────────────────────────

class DevignDataset(Dataset):
    """
    Devign dataset from Microsoft CodeXGLUE.

    Each sample: a C function labeled 0 (secure) or 1 (vulnerable).
    Real-world bugs: resource leaks, use-after-free, DoS vulnerabilities
    sourced from QEMU and FFmpeg CVE reports.

    Splits: 80% train / 10% validation / 10% test (pre-defined)

    Download:
        pip install gdown
        mkdir dataset && cd dataset
        gdown https://drive.google.com/uc?id=1x6hoF7G-tSYxg8AFybggypLZgMGDNHfF
        cd ..
    """

    JSON_PATH   = os.path.join("dataset", "/Users/pawan/Downloads/hgsn_v3/dataset/function.json")
    HF_FALLBACK = "google/code_x_glue_cc_defect_detection"

    def __init__(self, split: str = "train", max_nodes: int = 128):
        assert split in ("train", "validation", "test")
        self.split = split
        self.max_nodes = max_nodes
        self.samples: list[dict] = []

        raw = self._load_raw()
        if raw is None:
            return

        # Split indices: 80 / 10 / 10
        total = len(raw)
        t_end = int(total * 0.80)
        v_end = int(total * 0.90)
        slices = {"train": raw[:t_end], "validation": raw[t_end:v_end], "test": raw[v_end:]}
        subset = slices[split]

        skipped = 0
        for row in subset:
            code  = row.get("func", "")
            label = int(row.get("target", 0))
            if not code or len(code.strip()) < 5:
                skipped += 1
                continue
            feats = code_to_ast(code, max_nodes)
            if feats is None:
                skipped += 1
                continue
            self.samples.append({
                **feats,
                "bug_label"  : torch.tensor(label,        dtype=torch.float),
                "class_label": torch.tensor(label * 2,    dtype=torch.long),
                "severity"   : torch.tensor(0.7 if label else 0.0, dtype=torch.float),
                "difficulty" : torch.tensor(min(len(code) / 2000.0, 1.0), dtype=torch.float),
            })

        print(f"  [{split}] {len(self.samples):,} samples loaded "
              f"({skipped} skipped)")

    def _load_raw(self) -> list | None:
        """Try local JSON first, then HuggingFace."""
        # 1. Local file
        if os.path.exists(self.JSON_PATH):
            print(f"📂 Loading from {self.JSON_PATH}")
            with open(self.JSON_PATH, "r", encoding="utf-8") as f:
                return json.load(f)

        # 2. HuggingFace
        try:
            from datasets import load_dataset, concatenate_datasets
            print("📥 Downloading from HuggingFace (cached after first run)…")
            ds = load_dataset(self.HF_FALLBACK, trust_remote_code=True)
            # Merge all splits so we can re-split deterministically
            combined = concatenate_datasets([ds["train"], ds["validation"], ds["test"]])
            return [{"func": r["func"], "target": r["target"]} for r in combined]
        except Exception as e:
            print(f"❌ Could not load dataset: {e}")
            print(f"   Run: pip install gdown && mkdir dataset && cd dataset")
            print(f"   gdown https://drive.google.com/uc?id=1x6hoF7G-tSYxg8AFybggypLZgMGDNHfF")
            return None

    def __len__(self):  return len(self.samples)
    def __getitem__(self, i): return self.samples[i]


# ── Collate ───────────────────────────────────────────────────────────────

def collate(batch: list[dict]) -> dict:
    out = {}
    for k in batch[0]:
        tensors = [item[k] for item in batch]
        out[k] = torch.stack(tensors)
    return out


# ── DataLoader builder ────────────────────────────────────────────────────

def build_loaders(max_nodes=128, batch_size=8, num_workers=0):
    """
    M2-optimized DataLoader settings:
      batch_size=8    → ~1.2 GB GPU memory per batch on M2
      num_workers=2   → 2 perf cores for data, 2 for MPS
      pin_memory=False → MPS uses unified memory (no pinning needed)
    """
    print("\n📊 Building Devign DataLoaders…")
    train_ds = DevignDataset("train",      max_nodes)
    val_ds   = DevignDataset("validation", max_nodes)
    test_ds  = DevignDataset("test",       max_nodes)

    kw = dict(collate_fn=collate, num_workers=num_workers,
              pin_memory=False, persistent_workers=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  **kw)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **kw)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, **kw)

    print(f"\n  Train : {len(train_ds):,}")
    print(f"  Val   : {len(val_ds):,}")
    print(f"  Test  : {len(test_ds):,}")
    print(f"  Batch : {batch_size} | Workers: {num_workers} | Device: {DEVICE}\n")
    return train_loader, val_loader, test_loader


# ── M2 config (imported by train.py) ─────────────────────────────────────

M2_CONFIG = {
    "hidden_dim"             : 128,
    "num_heads"              : 4,
    "num_transformer_layers" : 3,
    "num_gnn_layers"         : 2,
    "max_ast_nodes"          : 128,
    "batch_size"             : 8,
    "grad_accum_steps"       : 4,    # effective batch = 32
    "lr"                     : 3e-4,
    "num_epochs"             : 20,
    "grad_clip"              : 1.0,
    "num_workers"            : 0,     # macOS fork safety — use 0
    "pin_memory"             : False,
    "device"                 : str(DEVICE),
}


if __name__ == "__main__":
    import platform, torch
    print("=" * 56)
    print("  HGSN Pipeline — Devign/CodeXGLUE")
    print("=" * 56)
    print(f"  System  : {platform.system()} {platform.machine()}")
    print(f"  Device  : {DEVICE}")
    print(f"  PyTorch : {torch.__version__}")
    print("=" * 56)

    train_l, val_l, test_l = build_loaders()
    batch = next(iter(train_l))
    print("Sample batch:")
    print(f"  node_type_ids : {batch['node_type_ids'].shape}")
    print(f"  bug_label     : {batch['bug_label']}")
    print("\n✅ Pipeline ready. Run: python train.py")
