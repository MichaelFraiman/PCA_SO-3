#!/usr/bin/env python3
# plot_first100_raw_vs_shrunk.py
#
# For each volume, plot cumulative approximation energy using the first
# 100 coefficients, comparing RAW vs SHRUNK orderings on the same figure.
# Y-axis is zoomed to [0.85, 1.0].
#
# Expects per-volume CSVs created by final_shrinkage_parameters.py:
#   <basename>_coeff_energy_raw.csv
#   <basename>_coeff_energy_shrunk.csv

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# ----------------------- CONSTANTS -----------------------
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
IN_DIR      = os.path.join(BASE_DIR, "final_mat_vols_out_with_shrinkage")
OUT_SUBDIR  = "plots_first100"
FIRST_K     = 100
X_TICK_STEP = 10
Y_MIN, Y_MAX = 0.85, 1.00
# --------------------------------------------------------

def load_first_k(csv_path: str, first_k: int = 100):
    """Load k, ratio from CSV and keep only rows with k <= first_k."""
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    if data.ndim == 1:
        # single row edge case
        data = data.reshape(1, -1)
    k = data[:, 0].astype(int)
    ratio = data[:, 1]
    mask = k <= first_k
    return k[mask], ratio[mask]

def group_csvs_by_volume(in_dir: str):
    """
    Return {volume_base: {'raw': <path>|None, 'shrunk': <path>|None}}
    where volume_base is the prefix before '_coeff_energy_'.
    """
    pattern = os.path.join(in_dir, "*_coeff_energy_*.csv")
    files = glob.glob(pattern)
    grouped = defaultdict(lambda: {"raw": None, "shrunk": None})
    for f in files:
        base = os.path.basename(f)
        if base.endswith("_coeff_energy_raw.csv"):
            vol_base = base[: -len("_coeff_energy_raw.csv")]
            grouped[vol_base]["raw"] = f
        elif base.endswith("_coeff_energy_shrunk.csv"):
            vol_base = base[: -len("_coeff_energy_shrunk.csv")]
            grouped[vol_base]["shrunk"] = f
    return grouped

def plot_pair(volume: str, raw_csv: str, shr_csv: str, out_png: str):
    """Plot first FIRST_K points for RAW vs SHRUNK on the same axes."""
    # Load
    k_raw, r_raw = load_first_k(raw_csv, first_k=FIRST_K)
    k_shr, r_shr = load_first_k(shr_csv, first_k=FIRST_K)

    if k_raw.size == 0 and k_shr.size == 0:
        print(f"[skip] No k<= {FIRST_K} rows in either CSV for {volume}")
        return

    # Plot
    plt.figure(figsize=(8, 5))
    if k_raw.size:
        plt.plot(k_raw, r_raw, linewidth=2, label="raw ordering")
    if k_shr.size:
        plt.plot(k_shr, r_shr, linewidth=2, label="shrunk ordering")

    # Axes/labels
    max_k_seen = 0
    if k_raw.size: max_k_seen = max(max_k_seen, int(k_raw.max()))
    if k_shr.size: max_k_seen = max(max_k_seen, int(k_shr.max()))
    plt.xlabel(r'Number of modes $k$', fontsize=12)
    plt.ylabel('Cumulative explained ratio', fontsize=12)
    plt.title(f'First {FIRST_K} coefficients — {volume}', fontsize=14)
    plt.grid(which='major', linestyle='--', alpha=0.5)
    plt.minorticks_on()
    plt.grid(which='minor', linestyle=':', alpha=0.3)
    plt.ylim(Y_MIN, Y_MAX)
    if max_k_seen > 0:
        plt.locator_params(axis='x', nbins=int(np.ceil(max_k_seen / X_TICK_STEP)) + 1)

    # Reference lines (optional)
    for y in (0.95, 0.99):
        if Y_MIN < y < Y_MAX:
            plt.axhline(y, linestyle="--", linewidth=1, alpha=0.3)

    plt.legend(loc='best', fontsize=9)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"Saved: {out_png}  (ylim=({Y_MIN:.3f}, {Y_MAX:.3f}))")

def main():
    if not os.path.isdir(IN_DIR):
        raise SystemExit(f"Directory not found: {IN_DIR}")

    grouped = group_csvs_by_volume(IN_DIR)
    if not grouped:
        raise SystemExit(f"No CSVs found in: {IN_DIR}")

    out_dir = os.path.join(IN_DIR, OUT_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)

    for volume, paths in sorted(grouped.items()):
        raw_csv = paths["raw"]
        shr_csv = paths["shrunk"]
        if not raw_csv or not shr_csv:
            print(f"[warn] Missing pair for {volume}: raw={bool(raw_csv)} shrunk={bool(shr_csv)} (skipping)")
            continue
        out_png = os.path.join(out_dir, f"{volume}_first{FIRST_K}_raw_vs_shrunk.png")
        plot_pair(volume, raw_csv, shr_csv, out_png)

    print("[done] All plots written.")

if __name__ == "__main__":
    main()
