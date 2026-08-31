#!/usr/bin/env python3
"""
Score a checkpoint on a fold's test split and export patch-attention maps.

For each slide an ``<slide>_att.npz`` is written with:
    coords     [n_patches, 2]  top-left pixel coordinate of each patch
    attention  [1, 1, n_patches] MIL attention weight per patch

By default only correctly-classified slides are exported (the maps used for the
figures); pass ``--all`` to export every slide.
"""

import argparse
import gc
import logging
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

import _bootstrap  # noqa: F401
from pansubnet.dataset import WSIDataset
from pansubnet.folds import TASKS, build_split, load_excluded_patients
from pansubnet.model import WSIClassifier

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pretrained", required=True)
    p.add_argument("--folds-dir", required=True)
    p.add_argument("--embd-dir", required=True)
    p.add_argument("--out-dir", default="./attention_weights")
    p.add_argument("--task", default="subtype", choices=TASKS)
    p.add_argument("--fold-idx", type=int, default=0)
    p.add_argument("--exclude-file", default=None)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--all", action="store_true", help="export every slide, not just correct ones")

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

    excluded = load_excluded_patients(args.exclude_file)
    npy_files = [f for f in os.listdir(args.embd_dir) if f.endswith(".npy")]
    csv_path = os.path.join(args.folds_dir, f"fold{args.fold_idx}_test.csv")
    _, paths, labels = build_split(csv_path, args.task, "test", args.embd_dir,
                                   npy_files, excluded, is_eval=True)

    loader = DataLoader(
        WSIDataset(paths, labels, args.use_cell_ratios, -1, 7, args.nsamples, "test"),
        batch_size=1, shuffle=False, num_workers=args.num_workers,
    )
    model = load_model(args)

    n_saved = n_correct = 0
    for i, (images, target, wsi_path) in enumerate(loader):
        target = target.to(DEVICE).float()
        logits, attn, *_ , cellpatchattn = model(images)
        prob = torch.sigmoid(logits.view(-1)).item()
        pred = int(prob > args.threshold)
        correct = pred == int(target.item())
        n_correct += correct

        if args.all or correct:
            coords = np.array([np.asarray(c).tolist() for c in cellpatchattn])
            stem = os.path.splitext(os.path.basename(wsi_path[0]))[0]
            np.savez_compressed(
                os.path.join(args.out_dir, f"{stem}_att.npz"),
                coords=coords, attention=attn.cpu().numpy(),
            )
            n_saved += 1

        logging.info("[%d/%d] prob %.3f pred %d target %d%s",
                     i + 1, len(loader), prob, pred, int(target.item()),
                     "" if correct else "  (wrong)")
        del logits, attn, cellpatchattn
        torch.cuda.empty_cache()
        gc.collect()

    logging.info("Accuracy %.3f  |  saved %d/%d attention maps to %s",
                 n_correct / max(len(loader), 1), n_saved, len(loader), args.out_dir)


if __name__ == "__main__":
    main()
