"""Training helpers: metrics, meters, early stopping."""

import math
import os
import shutil

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)


class EarlyStopping:
    """Checkpoint every epoch and stop when the monitored metric stalls.

    A checkpoint ``<epoch>ep_checkpoint.pth.tar`` is written every epoch and the
    best one so far is mirrored to ``model_best.pth.tar``. After ``stop_epoch``
    the counter increments on every non-improving epoch; training stops once it
    reaches ``patience``.
    """

    def __init__(self, save_dir="", args=None, verbose=False):
        self.patience = args.patience
        self.args = args
        self.stop_epoch = args.stop_epoch
        self.monitor = args.monitor
        self.verbose = verbose
        self.save_dir = save_dir
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        os.makedirs(save_dir, exist_ok=True)

    def __call__(self, epoch, value, model, optimizer):
        score = -value if self.monitor == "loss" else value

        def _save(is_best):
            self._save_checkpoint(
                {
                    "epoch": epoch,
                    "arch": self.args.arch,
                    "state_dict": model.state_dict(),
                    "best_score": self.best_score,
                    "optimizer": optimizer.state_dict(),
                },
                is_best,
                filename=os.path.join(self.save_dir, f"{epoch}ep_checkpoint.pth.tar"),
            )

        if epoch <= self.stop_epoch:
            self.best_score = score
            _save(is_best=True)
            return

        if self.best_score is not None and score <= self.best_score:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} / {self.patience}")
            _save(is_best=False)
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            _save(is_best=True)
            self.counter = 0

    def _save_checkpoint(self, state, is_best, filename):
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, os.path.join(self.save_dir, "model_best.pth.tar"))


class AverageMeter:
    """Track the running average of a scalar."""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        return f"{self.name} {self.val:{self.fmt[1:]}} ({self.avg:{self.fmt[1:]}})"


class ProgressMeter:
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(m) for m in self.meters]
        print("\t".join(entries))

    @staticmethod
    def _get_batch_fmtstr(num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def accuracy(output, target, threshold, test=True, mil="att"):
    """Return ``[acc, recall, specificity, auc, f1, youden_index]`` (percentages
    for the first four)."""
    with torch.no_grad():
        batch_size = target.size(0)

        if mil == "att":
            scores = torch.sigmoid(output).cpu().numpy()
            pred = (torch.sigmoid(output) > threshold).float().view(-1, 1).t()
        elif mil == "casii":
            pred = torch.topk(output, 1, dim=1)[1].t()
            scores = torch.softmax(output, dim=1).cpu().numpy()[:, 1]
        else:
            raise ValueError(f"Unknown mil: {mil}")

        correct = pred.eq(target.view(1, -1))
        auc = roc_auc_score(target.view(-1).cpu().numpy(), scores)

        tp = torch.sum(torch.logical_and(correct, pred == 1)).float()
        tn = torch.sum(torch.logical_and(correct, pred == 0)).float()
        fp = torch.sum(torch.logical_and(~correct, pred == 1)).float()

        acc = correct.reshape(-1).float().sum(0, keepdim=True).mul_(100.0 / batch_size)
        recall = tp / torch.sum(target == 1).float()
        specificity = tn / torch.sum(target == 0).float()
        precision = tp / (tp + fp + 1e-7)

        recall_f = float(recall.cpu().numpy())
        spec_f = float(specificity.cpu().numpy())
        prec_f = float(precision.cpu().numpy())

        f1 = 2.0 * (prec_f * recall_f) / (prec_f + recall_f + 1e-7)
        youden = recall_f + spec_f - 1

        return [
            acc.view(-1).cpu().numpy()[0],
            recall.cpu().numpy() * 100,
            specificity.cpu().numpy() * 100,
            auc * 100,
            f1,
            youden,
        ]


def eval_accuracy(output, target, threshold):
    """Sklearn-based metrics for a binary sigmoid head."""
    with torch.no_grad():
        scores = torch.sigmoid(output).cpu().numpy()
        pred = (torch.sigmoid(output) > threshold).float().view(-1).cpu().numpy()
        target = target.view(-1).cpu().numpy()
        return [
            roc_auc_score(target, scores) * 100,
            accuracy_score(target, pred) * 100,
            precision_score(target, pred) * 100,
            recall_score(target, pred) * 100,
            f1_score(target, pred) * 100,
        ]


def adjust_learning_rate(optimizer, init_lr, epoch, epochs):
    """Cosine learning-rate decay."""
    cur_lr = init_lr * 0.5 * (1.0 + math.cos(math.pi * epoch / epochs))
    for group in optimizer.param_groups:
        group["lr"] = cur_lr


def extract_top_k_columns(matrix):
    """SVD-based selection of the most representative columns of ``matrix``
    (used to build CASii key sets)."""
    score = {}
    rank = np.linalg.matrix_rank(matrix)
    _, _, vh = np.linalg.svd(matrix, full_matrices=True)
    for j in range(matrix.shape[1]):
        cscore = np.sum(np.square(vh[0:rank, j])) / rank
        score[j] = min(1, rank * cscore)
    prominent = sorted(score, key=score.get, reverse=True)[:rank]
    return {
        "columns": prominent,
        "matrix": np.squeeze(matrix[:, [prominent]]),
        "scores": sorted(score.values(), reverse=True)[:rank],
    }
