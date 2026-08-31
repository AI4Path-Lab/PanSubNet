#!/usr/bin/env python3
"""
Publication attention figure, one per case.

  A   whole-slide H&E thumbnail (magnification + scale bar)
  B   the same H&E with the attention overlay (percentile colourbar, inset boxes)
  4x  highest-attention regions cropped at native magnification, CellViT nuclei
      overlaid (5 PanNuke classes) with a composition caption
  4x  low-attention regions for contrast, same layout

Inputs
------
  --att-dir    directory of ``*_att.npz`` (coords + attention), from
               ``scripts/eval_attention_maps.py``
  --ds-dir     directory of per-slide dataset ``*.npy`` (cell_types / centroids)
  --wsi-dir    directory tree containing the ``*.svs`` slides
  --folds-dir  directory of ``*_{test,ext}.csv`` for subtype / label metadata
  --out-dir    where to write the figures
"""

import argparse
import glob
import json
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import tiffslide  # noqa: E402
from matplotlib import colormaps  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402

PATCH = 256          # level-0 px per patch (npz coord spacing)
INSET_UM = 500       # inset field of view (micrometres)
N_HOTSPOTS = 4

CELL_CLASSES = {1: "Neoplastic", 2: "Inflammatory", 3: "Connective", 4: "Dead", 5: "Epithelial"}
CELL_COLORS = {1: "#e6194B", 2: "#3cb44b", 3: "#4363d8", 4: "#f032e6", 5: "#f58231"}


# --------------------------------------------------------------------- metadata
def subtype_map(folds_dir):
    m = {}
    for split in ("test", "ext"):
        for f in glob.glob(os.path.join(folds_dir, f"*_{split}.csv")):
            for _, r in pd.read_csv(f).iterrows():
                m[str(r["slide_filename"])] = dict(
                    subtype=str(r.get("subtype_norm", r.get("subtype", "?"))),
                    label=int(r["label"]) if not pd.isna(r.get("label")) else -1,
                )
    return m


def build_svs_index(wsi_dir, cache):
    if os.path.exists(cache):
        return json.load(open(cache))
    idx = {
        os.path.splitext(os.path.basename(f))[0]: f
        for f in glob.glob(os.path.join(wsi_dir, "**", "*.svs"), recursive=True)
    }
    json.dump(idx, open(cache, "w"))
    return idx


# -------------------------------------------------------------------- attention
def att_grid(coords, att):
    x0, y0 = coords[:, 0].min(), coords[:, 1].min()
    gx = ((coords[:, 0] - x0) // PATCH).astype(int)
    gy = ((coords[:, 1] - y0) // PATCH).astype(int)
    grid = np.full((gy.max() + 1, gx.max() + 1), np.nan, np.float32)
    grid[gy, gx] = att
    mask = ~np.isnan(grid)
    vals = grid[mask]
    order = vals.argsort()
    pct = np.empty_like(vals)
    pct[order] = np.linspace(0, 100, len(vals))
    out = np.full_like(grid, np.nan)
    out[mask] = pct
    return out, (int(x0), int(y0))


def density_grid(shape, origin, cells):
    d = np.zeros(shape, np.float32)
    if not cells:
        return d
    ox, oy = origin
    for (px, py), rec in cells.items():
        gx = int((px - ox) // PATCH)
        gy = int((py - oy) // PATCH)
        if 0 <= gy < shape[0] and 0 <= gx < shape[1]:
            d[gy, gx] += rec["counts"].sum()
    return d


def pick_spots(P, D, k=N_HOTSPOTS, min_sep=14):
    S = gaussian_filter(np.where(np.isnan(P), 0, P), 2)

    hot, Sw = [], S.copy()
    for _ in range(k):
        iy, ix = np.unravel_index(np.argmax(Sw), Sw.shape)
        hot.append((iy, ix))
        Sw[max(0, iy - min_sep):iy + min_sep, max(0, ix - min_sep):ix + min_sep] = -1

    tissue = D > np.percentile(D[D > 0], 40) if np.any(D > 0) else ~np.isnan(P)
    Slw = np.where(tissue, S, np.inf)
    low = []
    for _ in range(k):
        iy, ix = np.unravel_index(np.argmin(Slw), Slw.shape)
        low.append((iy, ix))
        Slw[max(0, iy - min_sep):iy + min_sep, max(0, ix - min_sep):ix + min_sep] = np.inf
    return hot, low


# ------------------------------------------------------------------------ cells
def load_cells(base, ds_dir):
    cands = glob.glob(os.path.join(ds_dir, base + "*.npy"))
    if not cands:
        return None
    d = np.load(cands[0], allow_pickle=True).item()
    out = {}
    for (x, y), v in d.items():
        out[(int(x), int(y))] = dict(
            counts=np.asarray(v["cell_types"], float),
            centroids=np.asarray(v.get("cell_centroids", np.empty((0, 2))), float),
            types=np.asarray(v.get("cells_in_patch", []), int),
        )
    return out


def comp_string(counts):
    total = counts.sum()
    if total == 0:
        return "no nuclei detected"
    p = 100 * counts / total
    return "  ".join(f"{CELL_CLASSES[i + 1][:4]} {p[i]:.0f}%" for i in range(5)) + f"   (n={int(total)})"


# ----------------------------------------------------------------------- drawing
def mag_label(slide, downsample):
    obj = float(slide.properties.get("tiffslide.objective-power", 40) or 40)
    return f"{obj / downsample:.1f}×"


def scalebar(ax, um_per_px, length_um, dark=True):
    x0, x1 = ax.get_xlim()
    y1, y0 = ax.get_ylim()
    w = length_um / um_per_px
    xs = x0 + 0.02 * (x1 - x0)
    ys = y0 + 0.97 * (y1 - y0)
    col = "k" if dark else "w"
    ln, = ax.plot([xs, xs + w], [ys, ys], "-", lw=3.5, color=col, solid_capstyle="butt")
    ln.set_path_effects([pe.Stroke(linewidth=6, foreground="w" if dark else "k"), pe.Normal()])
    txt = ax.text(xs + w / 2, ys - 0.03 * (y1 - y0),
                  f"{length_um:g} um" if length_um < 1000 else f"{length_um / 1000:g} mm",
                  ha="center", va="bottom", fontsize=7, color=col)
    txt.set_path_effects([pe.Stroke(linewidth=2, foreground="w" if dark else "k"), pe.Normal()])


def corner_tag(ax, text):
    ax.text(0.03, 0.97, text, transform=ax.transAxes, fontsize=7.5, va="top", ha="left",
            color="k", bbox=dict(fc="w", ec="none", alpha=0.75, pad=1.5))


def render(base, meta, npz_path, slide_path, ds_dir, out_png, overlay_floor=55):
    d = np.load(npz_path, allow_pickle=True)
    coords, att = d["coords"].astype(int), d["attention"].astype(float).ravel()
    P, origin = att_grid(coords, att)
    ox, oy = origin
    cells = load_cells(base, ds_dir)
    D = density_grid(P.shape, origin, cells)
    hot, low = pick_spots(P, D, k=N_HOTSPOTS)

    sl = tiffslide.TiffSlide(slide_path)
    mpp = float(sl.properties.get("tiffslide.mpp-x", 0.2627) or 0.2627)
    W, H = sl.dimensions
    tw = 2200
    ds_t = W / tw
    thumb = np.asarray(sl.get_thumbnail((tw, int(H / ds_t))))
    ds_ty = H / thumb.shape[0]

    gh, gw = P.shape
    mx = 0.04 * gw * PATCH / ds_t
    my = 0.04 * gh * PATCH / ds_ty
    cx0 = int(np.clip(ox / ds_t - mx, 0, thumb.shape[1] - 2))
    cx1 = int(np.clip((ox + gw * PATCH) / ds_t + mx, cx0 + 2, thumb.shape[1]))
    cy0 = int(np.clip(oy / ds_ty - my, 0, thumb.shape[0] - 2))
    cy1 = int(np.clip((oy + gh * PATCH) / ds_ty + my, cy0 + 2, thumb.shape[0]))
    thumb = thumb[cy0:cy1, cx0:cx1]

    def tx(x_l0):
        return x_l0 / ds_t - cx0

    def ty(y_l0):
        return y_l0 / ds_ty - cy0

    ext = [tx(ox), tx(ox + gw * PATCH), ty(oy + gh * PATCH), ty(oy)]

    fig = plt.figure(figsize=(17, 17), dpi=300)
    gs = fig.add_gridspec(4, 4, top=0.95, bottom=0.045, left=0.035, right=0.98,
                          hspace=0.34, wspace=0.20)

    axA = fig.add_subplot(gs[0:2, 0:2])
    axA.imshow(thumb)
    axA.set_xticks([])
    axA.set_yticks([])
    axA.set_title("A   H&E - whole section", fontsize=12, loc="left")
    corner_tag(axA, mag_label(sl, ds_t) + "   .   H&E")
    scalebar(axA, ds_t * mpp, 5000)

    axB = fig.add_subplot(gs[2:4, 0:2])
    axB.imshow(thumb)
    axB.set_xticks([])
    axB.set_yticks([])
    cmap = colormaps["magma"].copy()
    cmap.set_bad(alpha=0)
    Pshow = np.ma.masked_where(np.isnan(P) | (P < overlay_floor), P)
    im = axB.imshow(Pshow, cmap=cmap, vmin=overlay_floor, vmax=100, alpha=0.6,
                    extent=ext, interpolation="nearest")
    axB.set_title("B   attention overlay", fontsize=12, loc="left")
    cb = fig.colorbar(im, ax=axB, fraction=0.045, pad=0.02)
    cb.set_label("attention percentile within slide", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    scalebar(axB, ds_t * mpp, 5000)

    win = int(INSET_UM / mpp)
    spots = ([("hotspot %d" % (i + 1), s, "cyan") for i, s in enumerate(hot)]
             + [("low %d" % (i + 1), s, "goldenrod") for i, s in enumerate(low)])
    slots = [gs[0, 2], gs[0, 3], gs[1, 2], gs[1, 3], gs[2, 2], gs[2, 3], gs[3, 2], gs[3, 3]]

    for (name, (iy, ix), boxcol), slot in zip(spots, slots):
        cx = ox + ix * PATCH + PATCH // 2
        cy = oy + iy * PATCH + PATCH // 2
        x0 = int(np.clip(cx - win // 2, 0, W - win))
        y0 = int(np.clip(cy - win // 2, 0, H - win))
        reg = np.asarray(sl.read_region((x0, y0), 0, (win, win)).convert("RGB"))
        ax = fig.add_subplot(slot)
        ax.imshow(reg, extent=[0, win, win, 0])
        ax.set_xticks([])
        ax.set_yticks([])

        comp = np.zeros(5)
        if cells:
            for (px, py), rec in cells.items():
                if x0 <= px < x0 + win and y0 <= py < y0 + win:
                    comp += rec["counts"]
                    for (mcx, mcy), t in zip(rec["centroids"], rec["types"]):
                        if x0 <= mcx < x0 + win and y0 <= mcy < y0 + win:
                            ax.plot(mcx - x0, mcy - y0, ".", ms=2.2,
                                    color=CELL_COLORS.get(int(t), "#888"))
        ax.set_title(name, fontsize=9.5, loc="left")
        ax.set_xlabel(comp_string(comp), fontsize=6.8)
        corner_tag(ax, mag_label(sl, 1))
        scalebar(ax, mpp, 100, dark=False)

        axB.add_patch(Rectangle((tx(x0), ty(y0)), win / ds_t, win / ds_ty,
                                fill=False, ec=boxcol, lw=1.4))
        tag = ("H" if "hotspot" in name else "L") + name.split()[-1]
        axB.text(tx(x0), ty(y0) - 5, tag, color=boxcol, fontsize=7.5, va="bottom",
                 path_effects=[pe.Stroke(linewidth=2, foreground="k"), pe.Normal()])

    handles = [Patch(fc=CELL_COLORS[i], ec="none", label=CELL_CLASSES[i]) for i in range(1, 6)]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8.5, frameon=False,
               bbox_to_anchor=(0.5, 0.005), title="CellViT nuclei classes")
    fig.suptitle(f"Subtype: {meta['subtype']}", fontsize=9.5, y=0.97)
    fig.savefig(out_png, pad_inches=0.2)
    plt.close(fig)
    sl.close()
    print("[saved]", out_png)


# -------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--att-dir", required=True)
    ap.add_argument("--ds-dir", required=True)
    ap.add_argument("--wsi-dir", required=True)
    ap.add_argument("--folds-dir", required=True)
    ap.add_argument("--out-dir", default="./attn_figures")
    ap.add_argument("--n-per-class", type=int, default=1)
    ap.add_argument("--bases", nargs="*", default=None, help="explicit slide bases to render")
    ap.add_argument("--all", action="store_true", help="render every slide with metadata + a matching svs")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    sm = subtype_map(a.folds_dir)
    idx = build_svs_index(a.wsi_dir, os.path.join(a.out_dir, "svs_index.json"))

    if a.bases:
        targets = a.bases
    elif a.all:
        targets = [
            os.path.basename(f).replace("_att.npz", "")
            for f in sorted(glob.glob(os.path.join(a.att_dir, "*_att.npz")))
            if os.path.basename(f).replace("_att.npz", "") in sm
            and os.path.basename(f).replace("_att.npz", "") in idx
        ]
    else:
        want = {0: a.n_per_class, 1: a.n_per_class}
        targets = []
        for f in sorted(glob.glob(os.path.join(a.att_dir, "*_att.npz"))):
            b = os.path.basename(f).replace("_att.npz", "")
            meta = sm.get(b)
            if not meta or b not in idx or want.get(meta["label"], 0) <= 0:
                continue
            want[meta["label"]] -= 1
            targets.append(b)
            if all(v <= 0 for v in want.values()):
                break

    print(f"[plan] {len(targets)} slide(s) to render")
    for n, b in enumerate(targets, 1):
        meta = sm.get(b, dict(subtype="?", label=-1))
        key = {1: "classical", 0: "basal"}.get(meta["label"], "case")
        out_png = os.path.join(a.out_dir, f"attn_{key}_{b.split('_')[0][:12]}.png")
        if os.path.exists(out_png) and not a.overwrite:
            print(f"[{n}/{len(targets)}] skip (exists) {out_png}")
            continue
        print(f"[{n}/{len(targets)}] rendering {b} ...")
        try:
            render(b, meta, os.path.join(a.att_dir, b + "_att.npz"), idx[b], a.ds_dir, out_png)
        except Exception:  # noqa: BLE001
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
