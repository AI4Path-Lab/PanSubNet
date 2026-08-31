# PanSubNet

**Cell + patch cross-attention multiple-instance learning for pancreatic-cancer
whole-slide images.**

PanSubNet predicts slide-level labels (primarily **PDAC molecular subtype**,
basal-like vs. classical) by combining two views of each tissue patch:

1. **Cellular composition** — a spatially-aware self-attention module over the
   CellViT/PanNuke nuclei inside the patch, producing a patch-level "CLS" token.
2. **Morphology** — a patch embedding from a pathology foundation model
   (e.g. UNI v2, 1536-d).

The two are fused per patch (bilinear tensor fusion), and a gated-attention MIL
head aggregates all patches of a slide into a single logit. Patches are streamed
to the GPU one at a time, so slides with >100k patches fit in modest memory.

```
cells (1280-d) ──▶ SpatialAwareAttention ──▶ CLS token ┐
                                                        ├─ TensorFusion ─▶ patch repr ─┐
patch embedding (1536-d) ──▶ Linear ────────────────────┘                              │
                                                                                       ▼
                                              AttentionMIL over all patches ──▶ slide logit
```

---

## Repository layout

```
PanSubNet/
├── pansubnet/                 importable package
│   ├── model.py               SpatialAwareAttention, TensorFusion, AttentionMIL, WSIClassifier
│   ├── dataset.py             WSIDataset — loads per-slide patch dicts
│   ├── folds.py               fold-CSV parsing + per-task label logic
│   ├── utils.py               metrics, meters, EarlyStopping
│   └── data/excluded_patients.txt
├── scripts/
│   ├── train.py                   train / cross-validate / evaluate one fold
│   ├── extract_patch_embeddings.py  dump fused per-patch embeddings from a checkpoint
│   ├── eval_attention_maps.py       score a checkpoint + export per-patch attention (.npz)
│   └── draw_attention_maps.py        render publication attention figures
├── pyproject.toml
├── requirements.txt
└── LICENSE                    Apache-2.0
```

---

## Installation

```bash
git clone <your-remote> PanSubNet && cd PanSubNet
python -m venv .venv && source .venv/bin/activate      # optional
pip install -e ".[viz]"                                # core + visualisation deps
```

Install PyTorch with the CUDA build appropriate for your machine
(<https://pytorch.org/get-started/locally/>). The `scripts/` can also be run
directly without installing the package — each adds the repo root to `sys.path`
via `scripts/_bootstrap.py`.

---

## Data format

### Per-slide embedding files

One pickled `.npy` **dict** per slide, keyed by the patch's top-left pixel
coordinate at level 0:

```python
{
  (x, y): {
     "patch_embeddings": np.ndarray,   # [patch_dim]      foundation-model embedding
     "cell_embeddings":  np.ndarray,   # [n_cells, cell_dim] CellViT embeddings
     "cell_centroids":   np.ndarray,   # [n_cells, 2]      (x, y) in level-0 px
     "cell_types":       np.ndarray,   # [5]  PanNuke class ratios   (only if --use-cell-ratios)
     "cells_in_patch":   np.ndarray,   # [n_cells] per-cell class id (only if --use-single-cell)
  },
  ...
}
```

Save as `np.save(path, slide_dict)` and load with
`np.load(path, allow_pickle=True).item()`. Patches with zero cells are dropped.

### Fold CSVs

`--folds-dir` must contain `fold{N}_{train,val,test}.csv` (and `fold{N}_ext.csv`
for the external cohort) for `N = 0 … nfold-1`. Required columns depend on the
task:

| task          | label source                                             | other columns |
|---------------|----------------------------------------------------------|---------------|
| `subtype`     | integer `label` column (0 = basal-like, 1 = classical)   | `slide_filename`, `patientid` |
| `subtype_low` | `gata6` (train/val/test), `moffitt_type` (ext); low-confidence only | `slide_filename`, `patientid`, `confidence` |
| `gata6`       | `gata6` (train/val/ext), `final` (test)                   | `slide_filename`, `patientid`, `confidence` |
| `confidence`  | `confidence == "High"`                                    | `slide_filename` |
| `1ydfs`       | `event` column                                            | `embeddings_filename`, `patient_id` |

`slide_filename` is matched to an embedding file by name prefix inside
`--embd-dir`. `1ydfs` instead expects the CSV to carry an
`embeddings_filename` column directly.

Patient IDs listed in `pansubnet/data/excluded_patients.txt` are removed from the
**test** split (override with `--exclude-file`, or point it at an empty file).

---

## Usage

### 1. Train / cross-validate

Run once per fold:

```bash
for f in 0 1 2 3 4; do
  python scripts/train.py \
    --task subtype --fold-idx $f --nfold 5 \
    --folds-dir /data/pancan/folds \
    --embd-dir  /data/pancan/embeddings \
    --save-dir  runs/subtype \
    --epochs 60 --lr 5e-5 --monitor auc \
    --use-spatial-bias --use-patch-embeddings --projection-dim 2
done
```

Outputs, under `runs/subtype/fold_<f>/`:

* `<epoch>ep_checkpoint.pth.tar` — checkpoint each epoch
* `model_best.pth.tar`          — best epoch by `--monitor`
* TensorBoard event files
* `runs/subtype/logs/train_*.log`

Resume an interrupted fold with `--resume`.

### 2. Evaluate a checkpoint

```bash
python scripts/train.py --evaluate \
  --task subtype --fold-idx 0 \
  --folds-dir /data/pancan/folds --embd-dir /data/pancan/embeddings \
  --save-dir runs/subtype \
  --pretrained runs/subtype/fold_0/model_best.pth.tar \
  --use-spatial-bias --projection-dim 2
```

Writes `runs/subtype/fold_0/fold_0_test_predictions.csv` (per-slide logit,
probability, prediction and the serialised slide embedding).

### 3. Export fused patch embeddings

```bash
python scripts/extract_patch_embeddings.py \
  --pretrained runs/subtype/fold_0/model_best.pth.tar \
  --fold-csv   /data/pancan/folds/fold0_test.csv \
  --embd-dir   /data/pancan/embeddings \
  --out-dir    runs/subtype/patch_embeddings/fold_0 \
  --use-spatial-bias --projection-dim 2
```

One `<slide>.npy` per slide: `{(x, y): np.ndarray[hidden_dim1]}`.

### 4. Export attention maps

```bash
python scripts/eval_attention_maps.py \
  --pretrained runs/subtype/fold_0/model_best.pth.tar \
  --folds-dir  /data/pancan/folds \
  --embd-dir   /data/pancan/embeddings \
  --task subtype --fold-idx 0 \
  --out-dir    attention_weights \
  --use-spatial-bias --projection-dim 2
```

By default only correctly-classified slides are written (`--all` for every
slide). Each `<slide>_att.npz` holds `coords [n, 2]` and `attention [1, 1, n]`.

### 5. Render figures

```bash
pip install -e ".[viz]"        # matplotlib, scipy, tiffslide

python scripts/draw_attention_maps.py \
  --att-dir   attention_weights \
  --ds-dir    /data/pancan/embeddings \
  --wsi-dir   /data/pancan/slides \
  --folds-dir /data/pancan/folds \
  --out-dir   attn_figures \
  --all
```

Produces a multi-panel PNG per case: whole-slide H&E, attention overlay, and the
top/bottom attention regions with CellViT nuclei overlaid and their cell-type
composition.

---

## Key hyper-parameters

| flag | default | meaning |
|------|---------|---------|
| `--use-spatial-bias` | off | add `-distance` bias between cell centroids in the cell attention |
| `--use-cell-ratios` | off | append the 5 PanNuke class ratios to the patch CLS token |
| `--use-patch-embeddings` / `--no-patch-embeddings` | on | fuse the foundation-model embedding |
| `--patch-embeddings-only` | off | skip the cell branch entirely |
| `--projection-dim` | 1 | dimensionality of the MIL attention projection |
| `--aggregation-method` | `norm` | how a multi-dim attention projection is reduced to a scalar |
| `--nsamples` | 15000 | max patches sampled per training slide |
| `--pw` | 1.0 | extra multiplier on the positive-class `pos_weight` |
| `--mil` | `att` | `att` = gated-attention MIL; `casii` = cross-attention key-set MIL (experimental) |

---

## Notes

* Batch size is fixed at 1 slide; the model iterates patches internally.
* The `casii` MIL branch is retained for completeness but is not the tested path.
* `pansubnet.utils.EarlyStopping` writes a checkpoint every epoch and mirrors the
  best (by `--monitor`) to `model_best.pth.tar`.

## License

Apache-2.0 — see [LICENSE](LICENSE).
