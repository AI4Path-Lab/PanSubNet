#!/usr/bin/env python3
"""
Train / cross-validate PanSubNet on pancreatic-cancer whole-slide images.

One invocation trains a single fold (``--fold-idx``). Run it once per fold to get
a full cross-validation. Pass ``--evaluate --pretrained <ckpt>`` to score a
saved checkpoint on the fold's test split instead of training.

Example
-------
    python scripts/train.py \
        --task subtype --fold-idx 0 --nfold 5 --epochs 60 \
        --folds-dir /data/pancan/folds --embd-dir /data/pancan/embeddings \
        --save-dir runs/subtype --use-spatial-bias
"""

import argparse
import gc
import glob
import json
import logging
import os
import random
import re
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import _bootstrap  # noqa: F401  (adds repo root to sys.path)
from pansubnet import utils
from pansubnet.dataset import WSIDataset
from pansubnet.folds import (TASKS, build_split, load_excluded_patients,
                             split_stats)
from pansubnet.model import WSIClassifier

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MONITOR_INDEX = {"acc": 0, "auc": 1, "loss": 4, "f1_score": 5, "y_index": 6}


# --------------------------------------------------------------------------- args
def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)

    # task / data
    p.add_argument("--task", default="subtype", choices=TASKS)
    p.add_argument("--folds-dir", required=True,
                   help="dir with fold{N}_{train,val,test,ext}.csv")
    p.add_argument("--embd-dir", required=True,
                   help="dir with per-slide .npy embedding dicts")
    p.add_argument("--save-dir", required=True, help="output dir for checkpoints / logs")
    p.add_argument("--log-dir", default=None, help="log dir (default: <save-dir>/logs)")
    p.add_argument("--exclude-file", default=None,
                   help="newline-delimited patient IDs to drop from test (default: packaged list)")
    p.add_argument("--nfold", type=int, default=5)
    p.add_argument("--fold-idx", type=int, default=0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--num-workers", type=int, default=0)

    # model
    p.add_argument("--arch", default="WSIClassifier")
    p.add_argument("--mil", default="att", choices=["att", "casii"])
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
    p.add_argument("--use-single-cell", type=int, default=-1,
                   help="keep only this PanNuke class id (-1 = all)")
    p.add_argument("--projection-dim", type=int, default=1)
    p.add_argument("--aggregation-method", default="norm", choices=["norm", "learnable"])

    # optimisation
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--regtype", default="l2", choices=["l1", "l2"])
    p.add_argument("--reg", type=float, default=1e-5, help="weight decay / L1 strength")
    p.add_argument("--pw", type=float, default=1.0, help="extra multiplier on the positive class weight")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--nsamples", type=int, default=15000, help="max patches per training slide")
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--stop-epoch", type=int, default=5)
    p.add_argument("--monitor", default="auc", choices=list(MONITOR_INDEX))
    p.add_argument("--freq", type=int, default=100, help="log every N training slides")
    p.add_argument("--keyset-size", type=int, default=4000, help="CASii key-set length")
    p.add_argument("--t", type=int, default=100, help="CASii: keys kept per slide")

    # modes
    p.add_argument("--resume", action="store_true",
                   help="resume from the latest checkpoint in <save-dir>/fold_<idx>")
    p.add_argument("--evaluate", action="store_true", help="evaluate --pretrained, no training")
    p.add_argument("--pretrained", default="", help="checkpoint for --evaluate")
    p.add_argument("--debug", action="store_true", help="print per-slide CUDA memory")
    return p


# ---------------------------------------------------------------------- utilities
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def setup_logging(args):
    log_dir = args.log_dir or os.path.join(args.save_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"train_{args.task}_fold{args.fold_idx}_{stamp}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        force=True,
    )
    logging.info("Args: %s", json.dumps(vars(args), indent=2, default=str))
    return log_file


def find_latest_checkpoint(checkpoint_dir):
    if not os.path.isdir(checkpoint_dir):
        return None, -1
    best_epoch, best_path = -1, None
    for path in glob.glob(os.path.join(checkpoint_dir, "*ep_checkpoint.pth.tar")):
        m = re.search(r"(\d+)ep_checkpoint\.pth\.tar", os.path.basename(path))
        if m and int(m.group(1)) > best_epoch:
            best_epoch, best_path = int(m.group(1)), path
    return best_path, best_epoch


def load_checkpoint(path, model, optimizer):
    logging.info("Loading checkpoint %s", path)
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return int(ckpt.get("epoch", -1)) + 1


def save_test_predictions_csv(test_df, logits, probs, preds, slide_embeddings, fold, out_dir):
    assert len(test_df) == len(logits) == slide_embeddings.shape[0], "row / prediction mismatch"
    out = test_df.copy()
    out["logit"] = logits
    out["prob"] = probs
    out["pred"] = preds.astype(int)
    out[f"slide_embedding_{slide_embeddings.shape[1]}d"] = [
        json.dumps(v.tolist()) for v in slide_embeddings
    ]
    path = os.path.join(out_dir, f"fold_{fold}_test_predictions.csv")
    out.to_csv(path, index=False)
    logging.info("Saved test predictions -> %s", path)


# ---------------------------------------------------------------------- forward
def _forward(model, images, mil, keysets):
    if mil == "casii":
        y_hat, logits, atts, patch_emb = model((images, *keysets))
        return logits, None, None, patch_emb
    logits, attn, slide_emb, _w, _s, patch_emb, _cpa = model(images)
    return logits.view(-1).float(), attn, slide_emb, patch_emb


def train_one_epoch(loader, model, criterions, optimizer, epoch, args, writer, keysets_np):
    criterion, criterion_q = criterions
    keysets = tuple(torch.from_numpy(k).float().to(DEVICE) for k in keysets_np)
    losses = utils.AverageMeter("Loss", ":.4e")
    progress = utils.ProgressMeter(len(loader), [losses], prefix=f"Epoch [{epoch}]")
    model.train()

    high_keys = np.empty((0, args.hidden_dim1))
    low_keys = np.empty((0, args.hidden_dim1))
    outputs = targets = None

    for i, (images, target, _) in enumerate(loader):
        is_low = bool(target.item() == 0)
        target = target.to(DEVICE)
        target = target if args.mil == "casii" else target.float()

        logits, _attn, _slide_emb, patch_emb = _forward(model, images, args.mil, keysets)

        outputs = logits if outputs is None else torch.cat((outputs, logits), 0)
        targets = target if targets is None else torch.cat((targets, target), 0)

        if args.mil == "casii":
            keys = _extract_keys(patch_emb, args.t)
            bucket = low_keys if is_low else high_keys
            bucket = np.vstack([bucket, keys[: args.t]])
            if is_low:
                low_keys = bucket
            else:
                high_keys = bucket
            loss = criterion_q(logits, target)
        else:
            loss = criterion(logits, target)

        if args.regtype == "l1" and args.reg > 0:
            loss = loss + args.reg * sum(p.abs().sum() for p in model.parameters())

        losses.update(loss.item(), args.batch_size)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % args.freq == 0:
            progress.display(i)

    acc, sen, spe, auc, f1, y_index = utils.accuracy(outputs, targets, args.threshold, mil=args.mil)
    logging.info("Train[%d] acc %.3f sen %.3f spe %.3f auc %.3f loss %.3f f1 %.3f y %.3f",
                 epoch, acc, sen, spe, auc, losses.avg, f1, y_index)
    if writer:
        for name, val in zip(("loss", "acc", "sen", "spe", "auc", "f1", "y_index"),
                             (losses.avg, acc, sen, spe, auc, f1, y_index)):
            writer.add_scalar(f"{name}/train", val, epoch)

    return low_keys, high_keys


def _extract_keys(patch_emb, t):
    mat = patch_emb.squeeze(0).cpu().detach().numpy().T
    res = utils.extract_top_k_columns(mat)
    return np.transpose(np.squeeze(mat[:, res["columns"]]))


@torch.no_grad()
def evaluate(loader, model, criterions, args, split, writer=None, epoch=0,
             keysets_np=None, collect_outputs=False):
    criterion, _ = criterions
    keysets = tuple(torch.from_numpy(k).float().to(DEVICE) for k in keysets_np) if keysets_np else ()
    losses = utils.AverageMeter("Loss", ":.4e")
    model.eval()

    outputs = targets = None
    slide_embs = [] if collect_outputs else None

    for i, (images, target, _) in enumerate(loader):
        target = target.to(DEVICE).float()
        logits, _attn, slide_emb, _patch_emb = _forward(model, images, args.mil, keysets)

        outputs = logits if outputs is None else torch.cat((outputs, logits), 0)
        targets = target if targets is None else torch.cat((targets, target), 0)
        if args.mil == "att":
            losses.update(criterion(logits, target).item(), args.batch_size)

        if collect_outputs and slide_emb is not None:
            slide_embs.append(slide_emb.detach().cpu().numpy())

        if args.debug and torch.cuda.is_available():
            torch.cuda.synchronize()
            logging.info("[%s %d/%d] mem alloc %.0f MB",
                         split, i + 1, len(loader),
                         torch.cuda.memory_allocated() / 1024 ** 2)
        del logits, slide_emb
        torch.cuda.empty_cache()
        gc.collect()

    acc, sen, spe, auc, f1, y_index = utils.accuracy(outputs, targets, args.threshold, mil=args.mil)
    logging.info("%s acc %.3f sen %.3f spe %.3f auc %.3f loss %.3f f1 %.3f y %.3f",
                 split.capitalize(), acc, sen, spe, auc, losses.avg, f1, y_index)
    if writer:
        for name, val in zip(("loss", "acc", "sen", "spe", "auc", "f1", "y_index"),
                             (losses.avg, acc, sen, spe, auc, f1, y_index)):
            writer.add_scalar(f"{name}/{split}", val, epoch)

    metrics = (acc, auc, sen, spe, losses.avg, f1, y_index)
    if collect_outputs:
        logits_np = outputs.detach().cpu().numpy()
        probs = torch.sigmoid(outputs).cpu().numpy()
        preds = (probs > args.threshold).astype(int)
        embs = np.concatenate(slide_embs, axis=0) if slide_embs else np.empty((len(probs), 0))
        return metrics, (logits_np, probs, preds, embs)
    return metrics


# ---------------------------------------------------------------------- run
def build_model(args):
    return WSIClassifier(
        patch_dim=args.patch_dim, cell_dim=args.cell_dim,
        hidden_dim1=args.hidden_dim1, hidden_dim2=args.hidden_dim2,
        hidden_dim3=args.hidden_dim3, num_classes=args.num_classes,
        distance_metric=args.distance_metric, use_spatial_bias=args.use_spatial_bias,
        use_cell_ratios=args.use_cell_ratios, dropout=args.dropout,
        projection_dim=args.projection_dim, aggregation_method=args.aggregation_method,
        use_patch_embeddings=args.use_patch_embeddings,
        patch_embeddings_only=args.patch_embeddings_only, mil=args.mil, device=DEVICE,
        lowkeysetlength=args.keyset_size, highkeysetlength=args.keyset_size,
    ).to(DEVICE)


def make_loader(paths, labels, split, args, shuffle):
    ds = WSIDataset(paths, labels, args.use_cell_ratios, args.use_single_cell,
                    args.seed, args.nsamples, split)
    return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                      num_workers=args.num_workers, pin_memory=False)


def run_fold(args, fold):
    excluded = load_excluded_patients(args.exclude_file)
    npy_files = [f for f in os.listdir(args.embd_dir) if f.endswith(".npy")]

    def prep(split, is_eval=False):
        csv_path = os.path.join(args.folds_dir, f"fold{fold}_{split}.csv")
        df, paths, labels = build_split(csv_path, args.task, split, args.embd_dir,
                                        npy_files, excluded, is_eval)
        logging.info(split_stats(df, split, fold))
        return df, paths, labels

    _, train_paths, train_labels = prep("train")
    _, val_paths, val_labels = prep("val")
    test_df, test_paths, test_labels = prep("test", is_eval=args.evaluate)

    pos = max((train_labels == 1).sum(), 1)
    neg = (train_labels == 0).sum()
    pos_weight = torch.tensor([args.pw * neg / pos], dtype=torch.float32, device=DEVICE)
    criterions = (nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(DEVICE),
                  nn.CrossEntropyLoss().to(DEVICE))

    model = build_model(args)
    params = [p for p in model.parameters() if p.requires_grad]
    weight_decay = args.reg if args.regtype == "l2" else 0.0
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=weight_decay)

    ckpt_dir = os.path.join(args.save_dir, f"fold_{fold}")
    os.makedirs(ckpt_dir, exist_ok=True)

    keysets_np = (np.ones((1, args.keyset_size, args.hidden_dim1), dtype=np.float32),
                  np.ones((1, args.keyset_size, args.hidden_dim1), dtype=np.float32))

    test_loader = make_loader(test_paths, test_labels, "test", args, shuffle=False)

    # ---------------------------------------------------------- evaluate only
    if args.evaluate:
        if not args.pretrained:
            raise ValueError("--evaluate requires --pretrained")
        load_checkpoint(args.pretrained, model, optimizer=None)
        metrics, (logits, probs, preds, embs) = evaluate(
            test_loader, model, criterions, args, "test",
            keysets_np=keysets_np, collect_outputs=True,
        )
        save_test_predictions_csv(test_df, logits, probs, preds, embs, fold, ckpt_dir)
        return metrics

    # ---------------------------------------------------------- train
    train_loader = make_loader(train_paths, train_labels, "train", args, shuffle=True)
    val_loader = make_loader(val_paths, val_labels, "val", args, shuffle=False)

    start_epoch = 0
    if args.resume:
        path, _ = find_latest_checkpoint(ckpt_dir)
        if path:
            start_epoch = load_checkpoint(path, model, optimizer)
            logging.info("Resuming at epoch %d", start_epoch)

    writer = SummaryWriter(os.path.join(ckpt_dir, time.strftime("%Y%m%d-%H%M%S")))
    stopper = utils.EarlyStopping(save_dir=ckpt_dir, args=args)
    monitor_idx = MONITOR_INDEX[args.monitor]

    best_metrics, best_epoch, test_metrics = None, start_epoch, None
    for epoch in range(start_epoch, args.epochs):
        logging.info("=== epoch %d / %d ===", epoch, args.epochs)
        keysets_np = train_one_epoch(train_loader, model, criterions, optimizer,
                                     epoch, args, writer, keysets_np)
        keysets_np = tuple(np.expand_dims(k, 0) if k.ndim == 2 else k for k in keysets_np)

        val_metrics = evaluate(val_loader, model, criterions, args, "val", writer, epoch, keysets_np)
        stopper(epoch, val_metrics[monitor_idx], model, optimizer)

        test_metrics = evaluate(test_loader, model, criterions, args, "test", writer, epoch, keysets_np)
        if stopper.counter == 0:
            best_metrics, best_epoch = test_metrics, epoch
        if stopper.early_stop:
            logging.info("Early stop at epoch %d", epoch)
            break

    if best_metrics is None:
        best_metrics = test_metrics
    if best_metrics is None:
        raise RuntimeError("No epochs ran - check --resume / --epochs.")
    logging.info("Fold %d best (epoch %d): acc %.3f auc %.3f sen %.3f spe %.3f f1 %.3f y %.3f",
                 fold, best_epoch, best_metrics[0], best_metrics[1], best_metrics[2],
                 best_metrics[3], best_metrics[5], best_metrics[6])
    writer.close()
    return best_metrics


def main():
    args = build_parser().parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    seed_everything(args.seed)
    setup_logging(args)

    fold_metrics = [run_fold(args, args.fold_idx)]
    arr = np.array(fold_metrics)
    names = ["acc", "auc", "sen", "spe", "loss", "f1", "y_index"]
    logging.info("Fold %d summary: %s", args.fold_idx,
                 ", ".join(f"{n} {v:.3f}" for n, v in zip(names, arr.mean(0))))


if __name__ == "__main__":
    main()
