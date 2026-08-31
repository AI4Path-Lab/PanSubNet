#!/usr/bin/env python3
"""
Run a trained PanSubNet checkpoint over slides and dump the fused per-patch
representations.

For every input slide a ``<slide>.npy`` file is written containing a dict
``{(x, y): np.ndarray[hidden_dim1]}`` - the patch embedding the MIL head sees,
one vector per patch, keyed by the patch's top-left pixel coordinate.

Input can be a fold CSV (``--fold-csv``, needs a ``slide_filename`` or
``embeddings_filename`` column) or an explicit list of embedding ``.npy`` files
(``--slides``).
"""

import argparse
import gc
import glob
import logging
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import _bootstrap  # noqa: F401
from pansubnet.dataset import WSIDataset
from pansubnet.folds import find_embedding_file
from pansubnet.model import WSIClassifier

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pretrained", required=True, help="checkpoint (.pth.tar)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--embd-dir", help="dir with per-slide .npy embedding dicts")
    p.add_argument("--fold-csv", help="CSV listing the slides to process")
    p.add_argument("--slides", nargs="*", help="explicit .npy embedding files")

    p.add_argument("--patch-dim", type=int, default=1536)
    p.add_argument("--cell-dim", type=int, default=1280)
    p.add_argument("--hidden-dim1", type=int, default=512)
    p.add_argument("--hidden-dim2", type=int, default=512)
    p.add_argument("--hidden-dim3", type=int, default=256)
    p.add_argument("--num-classes", type=int, default=1)
    p.add_argument("--distance-metric", default="euclidean", choices=["euclidean", "manhattan"])
    p.add_argument("--use-spatial-bias", action="store_true")
    p.add_argument("--use-cell-ratios", action="store_true")
    p.add_argument("--use-patch-embeddings", action="store_true", default=True)
    p.add_argument("--no-patch-embeddings", dest="use_patch_embeddings", action="store_false")
    p.add_argument("--patch-embeddings-only", action="store_true")
    p.add_argument("--projection-dim", type=int, default=1)
    p.add_argument("--aggregation-method", default="norm", choices=["norm", "learnable"])
    p.add_argument("--nsamples", type=int, default=15000)
    p.add_argument("--num-workers", type=int, default=0)
    return p


def resolve_inputs(args):
    if args.slides:
        return list(args.slides)
    if args.fold_csv:
        df = pd.read_csv(args.fold_csv)
        if "embeddings_filename" in df.columns:
            paths = df["embeddings_filename"].tolist()
        elif "slide_filename" in df.columns and args.embd_dir:
            npy_files = [f for f in os.listdir(args.embd_dir) if f.endswith(".npy")]
            paths = [find_embedding_file(s, npy_files, args.embd_dir) for s in df["slide_filename"]]
        else:
            raise ValueError("fold CSV needs 'embeddings_filename', or 'slide_filename' + --embd-dir")
        return [p for p in paths if p and os.path.exists(p)]
    if args.embd_dir:
        return sorted(glob.glob(os.path.join(args.embd_dir, "*.npy")))
    raise ValueError("provide one of --slides, --fold-csv, or --embd-dir")


def load_model(args):
    model = WSIClassifier(
        patch_dim=args.patch_dim, cell_dim=args.cell_dim,
        hidden_dim1=args.hidden_dim1, hidden_dim2=args.hidden_dim2,
        hidden_dim3=args.hidden_dim3, num_classes=args.num_classes,
        distance_metric=args.distance_metric, use_spatial_bias=args.use_spatial_bias,
        use_cell_ratios=args.use_cell_ratios, projection_dim=args.projection_dim,
        aggregation_method=args.aggregation_method,
        use_patch_embeddings=args.use_patch_embeddings,
        patch_embeddings_only=args.patch_embeddings_only, mil="att", device=DEVICE,
    ).to(DEVICE)
    ckpt = torch.load(args.pretrained, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt.get("state_dict", ckpt))
    model.eval()
    return model


@torch.no_grad()
def main():
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    os.makedirs(args.out_dir, exist_ok=True)

    paths = resolve_inputs(args)
    logging.info("Processing %d slide(s)", len(paths))

    model = load_model(args)
    dataset = WSIDataset(paths, [0] * len(paths), args.use_cell_ratios, -1,
                         seed=7, nsamples=args.nsamples, split="test")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

    for i, (wsi_dict, _label, wsi_path) in enumerate(loader):
        stem = os.path.splitext(os.path.basename(wsi_path[0]))[0]
        try:
            _logits, _attn, _emb, _w, _s, patch_emb, cellpatchattn = model(wsi_dict)
            patch_emb = patch_emb.squeeze(0).cpu().numpy()
            coords = list(cellpatchattn.keys())
            out = {tuple(np.asarray(c).tolist()): patch_emb[j] for j, c in enumerate(coords)}
            np.save(os.path.join(args.out_dir, f"{stem}.npy"), out)
            logging.info("[%d/%d] %s -> %d patches", i + 1, len(loader), stem, len(out))
        except Exception as err:  # noqa: BLE001
            logging.error("[%d/%d] %s failed: %s", i + 1, len(loader), stem, err)
        finally:
            torch.cuda.empty_cache()
            gc.collect()


if __name__ == "__main__":
    main()
