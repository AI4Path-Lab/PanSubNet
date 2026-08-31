"""Fold / split preparation.

Turns a fold CSV into ``(dataframe, embedding_paths, labels)`` for a given
prediction task. Every task ends up with an integer ``label`` column in
``{0, 1}``.

Supported tasks
---------------
``subtype``      binary molecular subtype (basal-like = 0, classical = 1) from a
                 pre-computed ``label`` column.
``subtype_low``  same, restricted to low-confidence cases; labels from ``gata6``
                 (train/val/test) and ``moffitt_type`` (external).
``gata6``        labels from ``gata6`` for train/val/external and from the
                 consensus ``final`` call for test.
``confidence``   predict whether the subtype call is high-confidence.
``1ydfs``        1-year disease-free-survival event; expects an
                 ``embeddings_filename`` column already in the CSV.
"""

import os

import pandas as pd

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_EXCLUDED_PATIENTS_FILE = os.path.join(_PKG_DIR, "data", "excluded_patients.txt")

GATA6_MAP = {"likely_classical": 1, "likely_basal": 0}
MOFFITT_MAP = {"classical": 1, "basal": 0}
FINAL_MAP = {
    "Classical": 1, "intermediate_likely_classical": 1,
    "Basal": 0, "intermediate_likely_basal": 0,
}

TASKS = ("subtype", "subtype_low", "gata6", "confidence", "1ydfs")


def load_excluded_patients(path=None):
    """Read a newline-delimited list of patient IDs to hold out."""
    path = path or DEFAULT_EXCLUDED_PATIENTS_FILE
    if not path or not os.path.exists(path):
        return set()
    with open(path) as fh:
        return {
            line.strip() for line in fh
            if line.strip() and not line.startswith("#")
        }


def find_embedding_file(slide_filename, npy_files, embd_dir):
    """Match a slide filename to its ``.npy`` embedding file by name prefix."""
    stem = str(slide_filename).replace(".svs", "")
    for name in npy_files:
        if name.startswith(stem):
            return os.path.join(embd_dir, name)
    return None


def _resolve_labels(df, task, split, is_eval):
    df = df.copy()

    if task == "subtype":
        pass  # `label` column already present

    elif task == "confidence":
        df["label"] = (df["confidence"] == "High").astype(int)

    elif task == "gata6":
        if split == "test":
            if not is_eval:
                df = df[df["confidence"] == "Low"]
            df["label"] = df["final"].map(FINAL_MAP)
        else:  # train / val / ext
            df["label"] = df["gata6"].map(GATA6_MAP)

    elif task == "subtype_low":
        if split in ("train", "val"):
            df = df[df["confidence"] == "Low"]
            df["label"] = df["gata6"].map(GATA6_MAP)
        elif split == "test":
            if not is_eval:
                df = df[df["confidence"] == "Low"]
            df["label"] = df["gata6"].map(GATA6_MAP)
        else:  # ext
            df = df[df["confidence"] == "Low"]
            df["label"] = df["moffitt_type"].map(MOFFITT_MAP)

    elif task == "1ydfs":
        df["label"] = df["event"]

    else:
        raise ValueError(f"Unknown task '{task}'. Choose from {TASKS}.")

    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    return df[df["label"].isin([0, 1])].reset_index(drop=True)


def build_split(csv_path, task, split, embd_dir, npy_files, excluded=None,
                is_eval=False):
    """Return ``(df, paths, labels)`` for one fold split.

    ``df`` carries an ``embeddings_filename`` column pointing at existing files
    and an integer ``label`` column.
    """
    excluded = excluded or set()
    df = pd.read_csv(csv_path)

    if split == "test" and task != "confidence":
        pid_col = "patient_id" if task == "1ydfs" else "patientid"
        if pid_col in df.columns:
            df = df[~df[pid_col].isin(excluded)]

    df = _resolve_labels(df, task, split, is_eval)
    df["label"] = df["label"].astype(int)

    if task == "1ydfs":
        if "embeddings_filename" not in df.columns:
            raise KeyError("task '1ydfs' expects an 'embeddings_filename' column in the fold CSV")
    else:
        df["embeddings_filename"] = df["slide_filename"].apply(
            find_embedding_file, args=(npy_files, embd_dir)
        )
        df = df[df["embeddings_filename"].notna() & (df["embeddings_filename"] != "")]

    df = df[df["embeddings_filename"].apply(os.path.exists)].reset_index(drop=True)
    return df, df["embeddings_filename"], df["label"]


def split_stats(df, split, fold):
    pos = int((df["label"] == 1).sum())
    neg = int((df["label"] == 0).sum())
    return (
        f"\nSplit: {split}, Fold: {fold}\n"
        f"Total slides: {len(df)}\n"
        f"Positive slides: {pos}\n"
        f"Negative slides: {neg}\n"
    )
