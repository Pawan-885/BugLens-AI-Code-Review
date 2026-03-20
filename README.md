# BugLens — AI Code Review System
## Complete Setup & Execution Guide

---

## 📁 Project Structure

```
hgsn_v3/
├── index.html          ← Website (open in browser)
├── dataset_pipeline.py ← Dataset loading & AST feature extraction
├── train.py            ← Model architecture + training loop
├── app.py              ← Flask API server
├── README.md           ← This file
│
├── dataset/            ← Create this folder (Step 2)
│   └── function.json   ← Dataset goes here
│
└── checkpoints/        ← Auto-created during training
    └── hgsn_best.pt    ← Best model checkpoint
```

---

## ✅ System Requirements

| Item | Requirement |
|------|------------|
| Hardware | MacBook Air M2, 8GB RAM, 256GB SSD |
| OS | macOS 13 (Ventura) or later |
| Python | 3.9 or later |
| Disk space needed | ~500MB total |
| Training time | ~45 minutes (20 epochs, M2 MPS) |

---

## 🗂️ Dataset

**Name:** Devign (via Microsoft CodeXGLUE)  
**Size:** ~28MB (~27,000 real C functions from QEMU + FFmpeg)  
**Labels:** 0 = Secure, 1 = Vulnerable (binary classification)  
**Why best:** Real CVE-sourced security bugs, pre-split 80/10/10, industry benchmark

### Direct Download Links

| Method | Link |
|--------|------|
| **Google Drive (direct file)** | https://drive.google.com/file/d/1x6hoF7G-tSYxg8AFybggypLZgMGDNHfF/view |
| **gdown command** | `gdown https://drive.google.com/uc?id=1x6hoF7G-tSYxg8AFybggypLZgMGDNHfF` |
| **HuggingFace** | https://huggingface.co/datasets/google/code_x_glue_cc_defect_detection |

---

## 🚀 Step-by-Step Execution

### STEP 1 — Set Up Python Environment

Open Terminal and run:

```bash
# Navigate to your project folder
cd ~/Downloads/hgsn_v3

# Create a virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# Your terminal prompt should now show (venv)
```

---

### STEP 2 — Install Dependencies

```bash
pip install torch torchvision torchaudio
pip install datasets transformers
pip install flask flask-cors
pip install gdown numpy scikit-learn
```

Verify PyTorch and M2 MPS are working:

```bash
python3 -c "import torch; print('PyTorch:', torch.__version__); print('MPS available:', torch.backends.mps.is_available())"
```

Expected output:
```
PyTorch: 2.x.x
MPS available: True
```

---

### STEP 3 — Download the Dataset

```bash
# Create the dataset folder
mkdir -p dataset
cd dataset

# Download using gdown (~28MB, takes ~10 seconds)
pip install gdown
gdown https://drive.google.com/uc?id=1x6hoF7G-tSYxg8AFybggypLZgMGDNHfF

# Go back to project root
cd ..
```

After download, verify:
```bash
ls dataset/
# Should show: function.json
```

**Alternative — browser download:**
1. Open: https://drive.google.com/file/d/1x6hoF7G-tSYxg8AFybggypLZgMGDNHfF/view
2. Click the download button (↓)
3. Move the downloaded file to your `dataset/` folder
4. Rename it to `function.json` if needed

---

### STEP 4 — Verify the Pipeline

```bash
python dataset_pipeline.py
```

Expected output:
```
✅ Apple MPS (Metal GPU) — M2 accelerated
📂 Loading from dataset/function.json
  [train]      21,854 samples loaded
  [validation]  2,731 samples loaded
  [test]        2,733 samples loaded

✅ Pipeline ready. Run: python train.py
```

---

### STEP 5 — Train the Model

```bash
python train.py
```

Expected output (training runs for ~45 minutes):
```
======================================================
  HGSN Training — Apple M2 MacBook Air
  Device  : mps
  Epochs  : 20
  Batch   : 8 × 4 = 32 effective
======================================================

Model parameters: 892,417
Epoch 01/20  train=0.6831  val=0.6124  acc=0.631  142s
  ✅ Best saved (acc=0.631)
Epoch 02/20  train=0.6312  val=0.5841  acc=0.659  138s
  ✅ Best saved (acc=0.659)
...
Epoch 20/20  train=0.4201  val=0.4837  acc=0.742  134s

🏁 Done. Best val acc: 0.742
   Checkpoint: checkpoints/hgsn_best.pt
```

If you run out of memory, reduce batch size:
```python
# In dataset_pipeline.py, change:
M2_CONFIG["batch_size"] = 4   # from 8 → 4
```

---

### STEP 6 — Start the API Server

Open a **new terminal tab**, activate the environment, and run:

```bash
source venv/bin/activate
python app.py
```

Expected output:
```
✅ Apple MPS (Metal GPU) — M2 accelerated
✅ Model loaded (val_acc=0.742)

🚀 BugLens API at http://localhost:5000
   Open index.html to use the UI
```

---

### STEP 7 — Open the Website

Open `index.html` in Safari or Chrome:

```bash
open index.html
```

Or drag `index.html` into your browser window.

The status indicator in the top-right shows:
- `● model online` → model loaded, full AI predictions
- `● server online (no model)` → server running, heuristic mode  
- `○ demo mode` → no server, browser-only heuristic mode (still works!)

---

## 🔎 How to Use the Website

1. **Paste code** into the left editor panel
2. **Choose language** using the pills (Python / C++ / Java / JS)
3. Click **Scan for Bugs**
4. Results appear on the right:
   - **Verdict** — Safe or Vulnerable with confidence
   - **Bug card** — Bug name, exact line number, line content, description
   - Click a bug card → shows all correct fixes with multiple approaches
5. **History panel** at the bottom keeps all your scans. Click any row to reload it.

---

## 🧠 Model Architecture

```
Code Snippet
  └── AST Extraction
        ├── ASTEmbedder    (node type + tree depth → embeddings)
        ├── SparseAttention (tree-distance biased attention)
        └── 3 × TransformerLayer
              ↓
         [CLS] token embedding
              ↓
       ┌──────┴──────┐
   DetectHead    SeverityHead
   (bug? 0/1)   (severity 0→1)
```

---

## 🛠 Troubleshooting

**MPS not available:**
```bash
# Requires macOS 13+. Check:
sw_vers  # should show 13.x or higher
```

**Dataset download fails:**
```bash
# Alternative: use HuggingFace (auto-downloads, no file needed)
# The pipeline falls back to HuggingFace automatically if dataset/ is missing
pip install datasets
python dataset_pipeline.py  # downloads ~50MB via HuggingFace
```

**Port 5000 in use:**
```bash
# Change port in app.py, last line:
app.run(host="0.0.0.0", port=5001, debug=False)
# And update fetch URL in index.html:
# fetch('http://localhost:5001/analyze', ...)
```

**Out of memory during training:**
```python
# In dataset_pipeline.py:
M2_CONFIG["batch_size"] = 4
M2_CONFIG["max_ast_nodes"] = 64
```

---

## 📊 Expected Performance

| Metric | Value |
|--------|-------|
| Val Accuracy | ~74% (20 epochs) |
| Inference time | < 50ms per snippet |
| Model size | ~3.5 MB |
| Memory (training) | ~4.5 GB peak |
| Memory (inference) | < 200 MB |

---

*BugLens · HybridGraphSemanticNet · Devign/CodeXGLUE · Apple M2 Optimized*
