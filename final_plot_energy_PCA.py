#!/usr/bin/env python3
# plot_wphi_three_variants_parallel.py
#
# For each volume, read:
#   <base>_coeff_energy_uncentered.csv
#   <base>_coeff_energy_centered_global.csv
#   <base>_coeff_energy_centered_pervol.csv
# and save a PNG with the three curves w^V_phi(d), d=1..FIRST_K.
#
# Output dir: mat_converted_N=<NN>_plots_PCA
# Parallel with a progress bar.

import os, re, glob
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

# (optional) avoid BLAS oversubscription when plotting/IO
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from tqdm.auto import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------- CONFIG (edit here) ---------------------
NN        = 64                          # Resolution used by your pipeline
FIRST_K   = 100                         # Plot d = 1..FIRST_K
FIG_DPI   = 150
CSV_DIR   = f"mat_converted_N={NN}_matrix"
OUT_DIR   = f"mat_converted_N={NN}_plots_PCA"
WORKERS   = os.cpu_count() or 1
LINE_STY  = {
    "uncentered":        dict(linestyle="-",  linewidth=1.8, label="Uncentered"),
    "centered_global":   dict(linestyle="--", linewidth=1.8, label="Centered (global)"),
    "centered_pervol":   dict(linestyle=":",  linewidth=1.8, label="Centered (per-volume)"),
}
# --------------------------------------------------------------

CSV_RE = re.compile(r"^(?P<base>.+)_coeff_energy_(?P<tag>uncentered|centered_global|centered_pervol)\.csv$")

def find_triplets(csv_dir):
    """Map base -> {'uncentered':path, 'centered_global':path, 'centered_pervol':path} (only complete triplets)."""
    mapping = {}
    for fp in glob.glob(os.path.join(csv_dir, "*_coeff_energy_*.csv")):
        m = CSV_RE.match(os.path.basename(fp))
        if not m:
            continue
        base = m.group("base"); tag = m.group("tag")
        mapping.setdefault(base, {})[tag] = fp
    need = {"uncentered", "centered_global", "centered_pervol"}
    return {b: d for b, d in mapping.items() if set(d) >= need}

def load_ratio(csv_path, first_k):
    """CSV is 'k,ratio'. Return x=1..K, y (monotone non-decreasing)."""
    arr = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    if arr.ndim == 1:  # single line edge-case
        arr = arr.reshape(1, -1)
    ratios = np.asarray(arr[:, 1], dtype=float)
    ratios = np.maximum.accumulate(ratios)  # guard tiny numeric dips
    K = min(first_k, ratios.size)
    x = np.arange(1, K + 1)
    return x, ratios[:K]

def plot_one(base, files, out_dir, first_k=100, dpi=150):
    os.makedirs(out_dir, exist_ok=True)

    x_u, y_u = load_ratio(files["uncentered"], first_k)
    x_g, y_g = load_ratio(files["centered_global"], first_k)
    x_w, y_w = load_ratio(files["centered_pervol"], first_k)

    K = min(x_u.size, x_g.size, x_w.size)
    x = np.arange(1, K + 1)

    plt.figure(figsize=(6.4, 4.0))
    plt.plot(x, y_u[:K], **LINE_STY["uncentered"])
    plt.plot(x, y_g[:K], **LINE_STY["centered_global"])
    plt.plot(x, y_w[:K], **LINE_STY["centered_pervol"])

    plt.xlabel(r"$d$")
    plt.ylabel(r"$w_{\phi}^{V}(d)$")
    plt.title(f"{base} — cumulative energy (first {K})")
    plt.ylim(0.0, 1.01)
    plt.xlim(1, max(100, K))
    plt.grid(True, alpha=0.25)
    plt.legend(loc="lower right", frameon=True)

    out_path = os.path.join(out_dir, f"{base}_wphi_N={NN}_d={K}_3curves.png")
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    return out_path

def main():
    if not os.path.isdir(CSV_DIR):
        raise SystemExit(f"[error] CSV_DIR not found: {CSV_DIR}")

    groups = find_triplets(CSV_DIR)
    if not groups:
        raise SystemExit(f"[error] No complete triplets in {CSV_DIR}")

    bases = sorted(groups.keys())
    total = len(bases)

    print(f"[plot] volumes with all three variants: {total}")
    print(f"[plot] saving to: {OUT_DIR}  |  first_k={FIRST_K}  |  workers={WORKERS}")

    done = 0
    errors = 0
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(plot_one, b, groups[b], OUT_DIR, FIRST_K, FIG_DPI) for b in bases]
        for fut in tqdm(as_completed(futs), total=total, desc="Plotting", unit="vol"):
            try:
                _ = fut.result()
                done += 1
            except Exception as e:
                errors += 1
                print(f"[error] {e}")

    print(f"[done] wrote {done} plot(s) to {OUT_DIR}  |  errors: {errors}")

if __name__ == "__main__":
    main()
