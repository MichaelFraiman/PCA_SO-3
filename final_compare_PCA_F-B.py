#!/usr/bin/env python3
"""
pca_vs_fb_sorted_usort.py

Compares, per volume:
  - PCA cumulative energy ratio (from: mat_converted_N={NN}_matrix/*_coeff_energy_centered_global.csv)
  - FB magnitude-sorted cumulative energy ratio (from: mat_converted_N={NN}_energy_fle/*.csv)
  - FB u-sorted cumulative energy ratio (from: mat_converted_N={NN}_FBexpansions/*_coeff_energy.csv, column 'w_ratio_usort')

Saves: OUT_DIR/<basename>_pca_vs_fb3_N={NN}_first{FIRST_K}.png
"""

import os
import re
import glob
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless / nohup-safe
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional

# ----------------------- CONFIG -----------------------
NN = 256
FIRST_K = 100

PCA_DIR        = f"mat_converted_N={NN}_matrix"         # *_coeff_energy_centered_global.csv
FB_SORTED_DIR  = f"mat_converted_N={NN}_energy_fle"     # magnitude-sorted FB (|a| sorting)
FB_USORT_DIR   = f"mat_converted_N={NN}_FBexpansions"   # u-sorted (expects 'w_ratio_usort' column)

OUT_DIR        = f"plots_pca_vs_fb_first{FIRST_K}_{NN}_with_usort"
FIG_DPI        = 150
NUM_WORKERS    = None  # None -> use os.cpu_count()-1
LINE_WIDTH     = 2.0
# ------------------------------------------------------

# ---------- NAME NORMALIZATION ----------
_SUFFIX_TOKENS = [
    r"coeffs_sorted", r"coeffs?", r"coeff",
    r"coeff_energy(?:_(?:raw|shrunk|centered_global|centered_pervol|uncentered))?",
    r"energy_fle", r"energy", r"cumulative", r"cum(?:_ratio)?",
    r"abs", r"mag(?:nitude)?",
    r"fb", r"fourier(?:_?bessel)?",
    r"pca",
    r"usort", r"u[_\-]?sorted",
    r"first\d+", r"k\d+", r"top\d+",
    r"v\d+", r"ver\d+", r"run\d+",
    r"converted", r"nosort",
]
_EXTRA_SUFFIXES = [
    r"\s*\(\d+\)", r"\s*\(copy\)", r"copy", r"dup", r"final", r"clean",
    r"rev\d+", r"draft\d*"
]
_SUFFIX_PATTERNS = (
    [rf"(?:[_\-\s]+(?:{tok}))$" for tok in _SUFFIX_TOKENS] +
    [rf"(?:[_\-\s]+(?:{tok}))$" for tok in _EXTRA_SUFFIXES] +
    [r"(?:[_\-\s]+)$"]
)

def _normalize_once(stem: str) -> str:
    for pat in _SUFFIX_PATTERNS:
        new = re.sub(pat, "", stem, flags=re.IGNORECASE)
        if new != stem:
            return new
    return stem

def normalize_key(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    s = stem.strip().lower()
    while True:
        new = _normalize_once(s)
        if new == s:
            break
        s = new
    s = re.sub(r"[_\-\s]+", "_", s).strip("_")
    return s

def _suffix_score(stem: str) -> int:
    s = stem.lower()
    score = 0
    penalties = [
        "raw", "shrunk", "centered_pervol", "uncentered",
        "coeffs_sorted", "coeffs", "coeff", "energy_fle", "energy",
        "cumulative", "cum", "abs", "mag", "fb", "fourier", "bessel", "pca", "nosort",
        "usort", "u-sorted", "u_sorted",
        "first", "k", "top", "v", "ver", "run", "copy", "dup", "final", "clean", "rev", "draft"
    ]
    for tok in penalties:
        if re.search(rf"(?:^|[_\-\s]){tok}\d*(?:$|[_\-\s])", s):
            score += 1
    return score

def pick_best(paths: List[str]) -> str:
    def key_fn(p):
        stem = os.path.splitext(os.path.basename(p))[0]
        return (_suffix_score(stem), len(stem), p)
    return sorted(paths, key=key_fn)[0]

def map_key_to_file_dict(folder: str, pattern: str = "*.csv") -> Dict[str, str]:
    candidates: Dict[str, List[str]] = {}
    for path in glob.glob(os.path.join(folder, pattern)):
        k = normalize_key(path)
        candidates.setdefault(k, []).append(path)
    chosen: Dict[str, str] = {}
    for k, paths in candidates.items():
        chosen[k] = pick_best(paths) if len(paths) > 1 else paths[0]
    return chosen

# ---------- CSV / SERIES LOADING ----------
def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan

def looks_like_header(row0):
    for cell in row0:
        if re.search(r"[A-Za-z]", (cell or "")):
            return True
    return False

def compute_running_ratio_from_vector(vec: np.ndarray) -> np.ndarray:
    a = np.asarray(vec, dtype=float).copy()
    if a.size:
        nz = ~np.isnan(a)
        if nz.any():
            # already a (nearly) nondecreasing ratio in [0,1]?
            if np.all(np.diff(a[nz]) >= -1e-12) and np.nanmin(a) >= -1e-6 and np.nanmax(a) <= 1.000001:
                return np.clip(a, 0.0, 1.0)
    # treat negatives as coeffs → square (fallback)
    if a.size and np.nanmin(a) < 0:
        energies = np.square(a)
    else:
        energies = a
    energies = np.nan_to_num(energies, nan=0.0, posinf=0.0, neginf=0.0)
    energies = np.maximum(energies, 0.0)
    total = energies.sum()
    if total <= 0:
        return np.zeros_like(energies, dtype=float)
    return np.cumsum(energies) / total

def load_running_ratio(path: str, prefer_cols: Optional[List[str]] = None) -> np.ndarray:
    """
    prefer_cols: try in order (e.g., ["w_ratio_usort"]) before falling back to common names
                 or the last numeric column.
    """
    prefer_cols = prefer_cols or []
    with open(path, "r", newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return np.zeros(0, dtype=float)

    if looks_like_header(rows[0]):
        header = [h.strip().lower() for h in rows[0]]
        data_rows = rows[1:]
        cols = {h: [] for h in header}
        for r in data_rows:
            if not r or all((c or "").strip() == "" for c in r):
                continue
            for i, h in enumerate(header):
                val = safe_float(r[i]) if i < len(r) else np.nan
                cols[h].append(val)
        cols = {h: np.array(v, dtype=float) for h, v in cols.items()}

        # 1) explicit preference list
        for cname in prefer_cols:
            if cname in cols and cols[cname].size:
                return compute_running_ratio_from_vector(cols[cname])

        # 2) common names
        for cname in ["w_ratio", "ratio", "cumulative", "w", "w_cum", "cum_ratio"]:
            if cname in cols and cols[cname].size:
                return compute_running_ratio_from_vector(cols[cname])

        # 3) energy or abs/real+imag fallbacks
        if "energy" in cols and cols["energy"].size:
            return compute_running_ratio_from_vector(cols["energy"])
        if "abs" in cols and "abs" in cols and cols["abs"].size:
            return compute_running_ratio_from_vector(np.square(cols["abs"]))
        if "real" in cols and "imag" in cols and cols["real"].size:
            return compute_running_ratio_from_vector(cols["real"]**2 + cols["imag"]**2)

        # 4) last column fallback
        return compute_running_ratio_from_vector(cols[header[-1]])

    # No header -> numeric fallback
    numeric = []
    for r in rows:
        vals = [safe_float(x) for x in r if (x or "").strip() != ""]
        if vals:
            numeric.append(vals)
    if not numeric:
        return np.zeros(0, dtype=float)
    arr = np.array(numeric, dtype=float)
    series = arr if arr.ndim == 1 else arr[:, -1]
    return compute_running_ratio_from_vector(series)

# ---------- PLOTTING WORKERS ----------
def ensure_outdir():
    os.makedirs(OUT_DIR, exist_ok=True)

def build_maps() -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
    # PCA: only centered_global CSVs
    pca_map = map_key_to_file_dict(PCA_DIR, pattern="*_coeff_energy_centered_global.csv")
    # FB magnitude-sorted: all CSVs in energy_fle dir
    fbs_map = map_key_to_file_dict(FB_SORTED_DIR, pattern="*.csv")
    # FB u-sorted: all CSVs in FBexpansions dir (expects 'w_ratio_usort' in header)
    fbu_map = map_key_to_file_dict(FB_USORT_DIR, pattern="*.csv")
    return pca_map, fbs_map, fbu_map

def build_work_items(pca_map: Dict[str, str], fbs_map: Dict[str, str], fbu_map: Dict[str, str]) -> List[Tuple[str, str, str, str]]:
    common = sorted(set(pca_map.keys()) & set(fbs_map.keys()) & set(fbu_map.keys()))
    return [(k, pca_map[k], fbs_map[k], fbu_map[k]) for k in common]

def process_one(key: str,
                pca_path: str,
                fb_sorted_path: str,
                fb_usort_path: str,
                first_k: int,
                out_dir: str,
                fig_dpi: int) -> Tuple[str, Optional[str], bool, str]:
    try:
        # PCA (centered_global)
        r_pca = load_running_ratio(pca_path, prefer_cols=["w_ratio", "ratio", "cumulative", "w", "cum_ratio"])

        # FB magnitude-sorted
        r_fbs = load_running_ratio(fb_sorted_path, prefer_cols=["w_ratio_sorted", "w_ratio", "ratio", "cumulative"])

        # FB u-sorted (explicit column)
        r_fbu = load_running_ratio(fb_usort_path, prefer_cols=["w_ratio_usort", "usort", "ratio_usort", "w_usort"])

        # Require all three present
        if r_pca.size == 0 or r_fbs.size == 0 or r_fbu.size == 0:
            return (key, None, False, "missing/empty series")

        K = min(first_k, r_pca.size, r_fbs.size, r_fbu.size)
        if K <= 0:
            return (key, None, False, f"no data within first {first_k} coefficients")

        x = np.arange(1, K+1)
        plt.figure(figsize=(6.8, 4.6))
        #plt.plot(x, r_pca[:K], linewidth=LINE_WIDTH, label="PCA (centered_global)")
        plt.plot(x, r_pca[:K], linewidth=LINE_WIDTH)
        #plt.plot(x, r_fbs[:K], linewidth=LINE_WIDTH, linestyle="--", label="FB (|a|-sorted)")
        plt.plot(x, r_fbs[:K], linewidth=LINE_WIDTH, linestyle="--")
        #plt.plot(x, r_fbu[:K], linewidth=LINE_WIDTH, linestyle=":", label="FB (u-sorted)")
        plt.plot(x, r_fbu[:K], linewidth=LINE_WIDTH, linestyle=":")
        # Uncomment labels if you want them on the figure:
        # plt.xlabel(r"$k$")
        # plt.ylabel(r"$w_f(k)$")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()

        out_png = os.path.join(out_dir, f"{key}_pca_vs_fb3_N={NN}_first{first_k}.png")
        plt.savefig(out_png, dpi=fig_dpi)
        plt.close()
        return (key, out_png, True, "ok")
    except Exception as e:
        return (key, None, False, f"error: {e}")

def main():
    ensure_outdir()

    # Sanity: directories must exist
    for label, d in [("PCA_DIR", PCA_DIR), ("FB_SORTED_DIR", FB_SORTED_DIR), ("FB_USORT_DIR", FB_USORT_DIR)]:
        if not os.path.isdir(d):
            print(f"[error] {label} not found: {d}")
            return

    pca_map, fbs_map, fbu_map = build_maps()
    print(f"[scan] PCA CSVs (centered_global): {len(pca_map)} in '{PCA_DIR}'")
    print(f"[scan] FB (|a|-sorted) CSVs:       {len(fbs_map)} in '{FB_SORTED_DIR}'")
    print(f"[scan] FB (u-sorted) CSVs:         {len(fbu_map)} in '{FB_USORT_DIR}'")

    work = build_work_items(pca_map, fbs_map, fbu_map)
    if not work:
        print("[stop] No common volumes found across PCA, FB(|a|-sorted), and FB(u-sorted).")
        return
    print(f"[match] Common volumes: {len(work)}")

    # worker count
    if NUM_WORKERS is None:
        cpu = os.cpu_count() or 2
        workers = max(1, cpu - 1)
    else:
        workers = int(NUM_WORKERS)
    print(f"[mp] Using {workers} worker processes.")

    submitted = successes = failures = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = []
        for key, pca_path, fbs_path, fbu_path in work:
            fut = ex.submit(process_one, key, pca_path, fbs_path, fbu_path, FIRST_K, OUT_DIR, FIG_DPI)
            futures.append(fut)
            submitted += 1

        for fut in as_completed(futures):
            key, out_path, ok, msg = fut.result()
            if ok:
                successes += 1
                print(f"[OK]   {key} -> {out_path}")
            else:
                failures += 1
                print(f"[SKIP] {key}: {msg}")

    print(f"[done] Submitted={submitted}, Successes={successes}, Failures={failures}")
    print(f"[out]  Figures in: {OUT_DIR}")

if __name__ == "__main__":
    main()
