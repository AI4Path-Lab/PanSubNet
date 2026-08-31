"""Slide-level dataset.

Each slide is stored as a pickled ``.npy`` dict::

    {
        (x, y): {
            "patch_embeddings": np.ndarray [patch_dim],
            "cell_embeddings":  np.ndarray [n_cells, cell_dim],
            "cell_centroids":   np.ndarray [n_cells, 2],
            "cell_types":       np.ndarray [5],           # PanNuke class ratios (optional)
            "cells_in_patch":   np.ndarray [n_cells],      # per-cell class id (optional)
        },
        ...
    }

``__getitem__`` returns ``(patch_dict_of_tensors, label, slide_path)``.
"""

import random

import numpy as np
import torch
from torch.utils.data import Dataset

VAL_TEST_PATCH_CAP = 70_000


class WSIDataset(Dataset):
    def __init__(self, wsi_paths, labels, use_cell_ratios=True, use_single_cell=-1,
                 seed=7, nsamples=15000, split="train", verbose=False):
        random.seed(seed)
        self.wsi_paths = list(wsi_paths)
        self.labels = list(labels)
        self.use_cell_ratios = use_cell_ratios
        self.use_single_cell = use_single_cell
        self.nsamples = nsamples
        self.split = split
        self.verbose = verbose

    def __len__(self):
        return len(self.wsi_paths)

    def __getitem__(self, idx):
        wsi_dict = np.load(self.wsi_paths[idx], allow_pickle=True).item()
        label = self.labels[idx]

        processed = {}
        for coords, patch_data in wsi_dict.items():
            cell_embs = patch_data["cell_embeddings"]
            if len(cell_embs) == 0:
                continue

            entry = {
                "patch_coords": torch.from_numpy(np.array(coords)).float(),
                "patch_embeddings": torch.from_numpy(patch_data["patch_embeddings"]).float(),
            }

            if self.use_cell_ratios:
                entry["cell_embeddings"] = torch.from_numpy(np.array(patch_data["cell_embeddings"])).float()
                entry["cell_centroids"] = torch.from_numpy(np.array(patch_data["cell_centroids"])).float()
                entry["cell_types"] = torch.from_numpy(np.array(patch_data["cell_types"])).float()
            elif self.use_single_cell != -1:
                cells_in_patch = np.array(patch_data["cells_in_patch"])
                keep = np.where(cells_in_patch == self.use_single_cell)[0]
                entry["cell_embeddings"] = torch.from_numpy(np.array(patch_data["cell_embeddings"])[keep]).float()
                entry["cell_centroids"] = torch.from_numpy(np.array(patch_data["cell_centroids"])[keep]).float()
            else:
                entry["cell_embeddings"] = torch.from_numpy(np.array(patch_data["cell_embeddings"])).float()
                entry["cell_centroids"] = torch.from_numpy(np.array(patch_data["cell_centroids"])).float()

            processed[coords] = entry

        # Subsample patches to bound memory / compute.
        if self.split == "train" and len(processed) > self.nsamples:
            keys = random.sample(list(processed), self.nsamples)
            processed = {k: processed[k] for k in keys}
        elif self.split in ("val", "test", "ext") and len(processed) > VAL_TEST_PATCH_CAP:
            keys = random.sample(list(processed), VAL_TEST_PATCH_CAP)
            processed = {k: processed[k] for k in keys}

        # Largest patches first (helps surface OOM early during streaming).
        processed = dict(
            sorted(processed.items(), key=lambda kv: len(kv[1]["cell_embeddings"]), reverse=True)
        )

        if self.verbose:
            print(f"Loaded {self.wsi_paths[idx]} with {len(processed)} patches.")

        return processed, label, self.wsi_paths[idx]
