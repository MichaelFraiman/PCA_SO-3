#!/usr/bin/env python3
"""
0_3.compare_PCA_FB.py

Per-volume comparison plots of cumulative energy concentration between:
  • PCA coefficients (from per-volume coeff NPZ exported by 0_1.covariance_matrix.py)
  • Fourier–Bessel (FB) energy curves (from 0_2.FB_expand.py CSVs and/or optional legacy CSVs):
      - |a|-sorted (coefficients sorted by decreasing magnitude; fastest way to get the FB energy curve)
      - u-sorted
      - (optional) lexicographic

Filename expectations (matches the newer 0_1/0_2 scripts, with legacy fallbacks):

PCA coeff NPZ (preferred):
  <base>_coeffs_L=<L>_eps=<eps_tag>_centered_global_all.npz
  <base>_coeffs_L=<L>_eps=<eps_tag>_centered_global_topK.npz

Legacy PCA coeff NPZ (still supported):
  <base>_coeffs_centered_global_all.npz
  <base>_coeffs_centered_global_topK.npz

FB magnitude-sorted energy CSV (optional, legacy / precomputed):
  <base>_coeff_energy.csv        (expects a column like 'w_ratio' or already-cumulative ratios)

If the magnitude-sorted CSV is not available, this script derives the |a|-sorted
curve from the FBexpansions CSV by taking the per-coefficient energy increments
from the lex curve and sorting them descending.

FB u-sorted / lex energy CSV (optional, from 0_2.FB_expand.py):
  <base>_coeff_energy.csv        (expects columns 'w_ratio_usort' and/or 'w_ratio_lex')

The PCA curve is computed from the NPZ coefficients using the stored global ordering
(order_flat) over (ell, r). For each (ell, r) row, all m are appended as separate steps.

Notes:
  • If only a *_topK.npz exists (row-trimmed export), the PCA curve is incomplete.
    By default those files are skipped; use --allow-trimmed-pca to plot anyway.
  • FB |a|-sorted inputs are optional; if fb-sorted-dir doesn't exist, the script
    derives the |a|-sorted curve from the FBexpansions lex curve (no extra files).

Example:
  python3 0_3.compare_PCA_FB.py --expect-n 20 --L 10 --eps 1e-6 --first-k 200 \
    --pca-dir mat_converted_N=20_matrix \
    --fb-usort-dir mat_converted_N=20_FBexpansions \
    --fb-sorted-dir mat_converted_N=20_energy_fle
"""

import os
import sys

# Avoid BLAS/OpenMP oversubscription on macOS / in multiprocessing.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import re
import glob
import csv
import fnmatch
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tqdm.auto import tqdm


# ============================== CLI ==============================

def eps_to_tag(eps: float) -> str:
    return f"{eps:.0e}".replace("+", "")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="0_3.compare_PCA_FB.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Plot per-volume cumulative energy comparison between PCA (NPZ coeffs) and FB (CSV energy curves).",
    )

    p.add_argument("--expect-n", type=int, default=22, help="Used to form default folder names.")
    p.add_argument("--L", type=int, default=20, help="Bandlimit used for PCA coeff filenames (if present).")
    p.add_argument("--eps", type=float, default=1e-6, help="Eps used for PCA coeff filenames (if present).")

    p.add_argument("--first-k", type=int, default=100, help="Plot only the first K steps (after any u-sort zero filtering).")
    p.add_argument("--dpi", type=int, default=150, help="PNG DPI.")
    p.add_argument("--line-width", type=float, default=2.0, help="Plot line width.")

    # Input directories
    p.add_argument(
        "--pca-dir",
        default=None,
        help="Directory containing per-volume PCA coeff NPZs. Default: mat_converted_N=<expect-n>_matrix",
    )
    p.add_argument(
        "--fb-sorted-dir",
        default=None,
        help="Directory containing FB magnitude-sorted energy CSVs (optional). Default: mat_converted_N=<expect-n>_energy_fle",
    )
    p.add_argument(
        "--fb-usort-dir",
        default=None,
        help="Directory containing FB u-sorted/lex CSVs from 0_2.FB_expand.py (optional). Default: mat_converted_N=<expect-n>_FBexpansions",
    )

    # Output directory
    p.add_argument(
        "--out-dir",
        default=None,
        help=(
            "Output directory for PNGs. Default: mat_converted_N=<expect-n>_plots/"
            "compare_PCA_FB_L=<L>_eps=<eps_tag>_first<first-k>"
        ),
    )

    # Matching / filtering
    p.add_argument(
        "--targets",
        nargs="*",
        default=[],
        help=(
            "Restrict which basenames to plot. Each entry may be an exact basename (no extension) "
            "or a glob pattern (e.g. 1f*, *foo*). Empty means all matched volumes."
        ),
    )

    # Curves to include
    p.add_argument(
        "--include-fb-lex",
        action="store_true",
        default=False,
        help="If FB u-sort CSVs are available, also plot the lexicographic cumulative ratio (w_ratio_lex).",
    )

    # u-sort zero skipping (matches the old final_compare behavior)
    p.add_argument(
        "--usort-skip-zeros",
        dest="usort_skip_zeros",
        action="store_true",
        default=True,
        help="For u-sorted FB curve: drop increments below thresholds and re-cumulate.",
    )
    p.add_argument(
        "--no-usort-skip-zeros",
        dest="usort_skip_zeros",
        action="store_false",
        help="Do not filter zero/near-zero increments in the u-sorted FB curve.",
    )
    p.add_argument("--usort-zero-abs-tol", type=float, default=1e-5, help="Absolute threshold for u-sort increment filtering.")
    p.add_argument("--usort-zero-rel-tol", type=float, default=1e-5, help="Relative threshold for u-sort increment filtering (fraction of max increment).")

    # PCA trimmed handling
    p.add_argument(
        "--allow-trimmed-pca",
        action="store_true",
        default=False,
        help="Allow plotting from *_topK.npz row-trimmed coefficient packs (curve is incomplete).",
    )

    # Execution
    p.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Worker processes for plotting (-1 means all cores minus 1).",
    )
    p.add_argument("--pbar", dest="pbar", action="store_true", default=True, help="Enable progress bar.")
    p.add_argument("--no-pbar", dest="pbar", action="store_false", help="Disable progress bar.")

    p.add_argument("--legend", dest="legend", action="store_true", default=True, help="Show legend in plots.")
    p.add_argument("--no-legend", dest="legend", action="store_false", help="Hide legend in plots.")
    p.add_argument("--dry-run", action="store_true", default=False, help="Scan and report matches, but do not render plots.")

    return p.parse_args()


# ============================== filename parsing / mapping ==============================

def _stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def base_from_pca_npz(path: str) -> str:
    """Extract <base> from <base>_coeffs_...npz"""
    s = _stem(path)
    m = re.match(r"^(?P<base>.+?)_coeffs(?:[_\-].*)?$", s, flags=re.IGNORECASE)
    return m.group("base") if m else s


def base_from_coeff_energy_csv(path: str) -> str:
    """Extract <base> from <base>_coeff_energy*.csv"""
    s = _stem(path)
    m = re.match(r"^(?P<base>.+?)_coeff_energy(?:[_\-].*)?$", s, flags=re.IGNORECASE)
    return m.group("base") if m else s


def _matches_any_target(name: str, targets: List[str]) -> bool:
    if not targets:
        return True
    for t in targets:
        t = (t or "").strip()
        if not t:
            continue
        if any(ch in t for ch in "*?[]"):
            if fnmatch.fnmatch(name, t) or fnmatch.fnmatch(name.lower(), t.lower()):
                return True
        else:
            if name == t or name.lower() == t.lower():
                return True
    return False


@dataclass(frozen=True)
class FileChoice:
    base: str
    path: str


def _pca_score(path: str, want_L: int, want_eps_tag: str) -> Tuple[int, int, str]:
    """Lower is better."""
    s = _stem(path).lower()
    score = 0

    # Prefer centered_global
    score += 0 if "centered_global" in s else 5

    # Prefer matching L/eps tags if present
    # NOTE: use explicit _/- boundaries because underscores are "word" characters for \b.
    if re.search(rf"(?:^|[_\-])l={want_L}(?:$|[_\-])", s):
        score += 0
    elif re.search(r"(?:^|[_\-])l=\d+", s):
        score += 2

    if f"eps={want_eps_tag}" in s:
        score += 0
    elif "eps=" in s:
        score += 2

    # Prefer _all
    if s.endswith("_all") or s.endswith("_all.npz") or "_all" in s:
        score += 0
    elif "topk" in s:
        score += 1
    else:
        score += 2

    # Prefer shorter stems
    return (score, len(s), path)


def scan_pca_npz(pca_dir: str, want_L: int, want_eps_tag: str) -> Dict[str, FileChoice]:
    """Map lowercased base -> FileChoice for PCA NPZs."""
    patterns = [
        os.path.join(pca_dir, "*_coeffs_*.npz"),
        os.path.join(pca_dir, "*_coeffs*.npz"),
    ]
    paths: List[str] = []
    for pat in patterns:
        paths.extend(glob.glob(pat))

    candidates: Dict[str, List[str]] = {}
    for pth in paths:
        b = base_from_pca_npz(pth)
        key = b.lower()
        candidates.setdefault(key, []).append(pth)

    chosen: Dict[str, FileChoice] = {}
    for key, ps in candidates.items():
        best = sorted(ps, key=lambda p: _pca_score(p, want_L, want_eps_tag))[0]
        chosen[key] = FileChoice(base=base_from_pca_npz(best), path=best)

    return chosen


def _csv_score(path: str) -> Tuple[int, int, str]:
    s = _stem(path).lower()
    score = 0
    # prefer plain "_coeff_energy.csv" over other variants
    score += 0 if s.endswith("_coeff_energy") else 1
    # avoid copies
    score += 2 if "copy" in s or "dup" in s else 0
    return (score, len(s), path)


def scan_energy_csv(folder: str) -> Dict[str, FileChoice]:
    """Map lowercased base -> FileChoice for *_coeff_energy*.csv."""
    paths = glob.glob(os.path.join(folder, "*.csv"))
    candidates: Dict[str, List[str]] = {}
    for pth in paths:
        base = base_from_coeff_energy_csv(pth)
        key = base.lower()
        candidates.setdefault(key, []).append(pth)

    chosen: Dict[str, FileChoice] = {}
    for key, ps in candidates.items():
        best = sorted(ps, key=_csv_score)[0]
        chosen[key] = FileChoice(base=base_from_coeff_energy_csv(best), path=best)

    return chosen


# ============================== CSV loaders ==============================

def safe_float(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def looks_like_header(row0: List[str]) -> bool:
    return any(re.search(r"[A-Za-z]", (cell or "")) for cell in row0)


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
    total = float(energies.sum())
    if total <= 0.0:
        return np.zeros_like(energies, dtype=float)
    return np.cumsum(energies) / total


def derive_mag_sorted_ratio_from_cumratio(cum_ratio: np.ndarray) -> np.ndarray:
    """Given a cumulative ratio curve for a complete coefficient list, derive the
    |a|-sorted curve by sorting per-step energy increments in decreasing order."""
    r = np.asarray(cum_ratio, dtype=float)
    if r.size == 0:
        return np.zeros(0, dtype=float)

    r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
    r = np.clip(r, 0.0, 1.0)
    # guard tiny non-monotone dips
    r = np.maximum.accumulate(r)

    inc = np.diff(r, prepend=0.0)
    inc = np.nan_to_num(inc, nan=0.0, posinf=0.0, neginf=0.0)
    inc = np.maximum(inc, 0.0)

    total = float(inc.sum())
    if total <= 0.0:
        return np.zeros_like(inc, dtype=float)

    inc_sorted = np.sort(inc)[::-1]
    return np.cumsum(inc_sorted) / total


def _read_csv_to_cols(path: str) -> Tuple[Optional[List[str]], List[List[float]]]:
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        first = next(reader, None)
        if first is None:
            return None, []
        rows = [first] + list(reader)

    if not rows:
        return None, []

    if looks_like_header(rows[0]):
        header = [h.strip().lower() for h in rows[0]]
        cols: List[List[float]] = [[] for _ in header]
        for r in rows[1:]:
            if not r or all((c or "").strip() == "" for c in r):
                continue
            for i in range(len(header)):
                val = safe_float(r[i]) if i < len(r) else float("nan")
                cols[i].append(val)
        return header, cols

    # no header
    numeric: List[List[float]] = []
    for r in rows:
        vals = [safe_float(x) for x in r if (x or "").strip() != ""]
        if vals:
            numeric.append(vals)
    return None, numeric


def load_running_ratio(path: str, prefer_cols: Optional[List[str]] = None) -> np.ndarray:
    """Generic loader for cumulative-energy CSVs."""
    prefer_cols = prefer_cols or []
    header, cols = _read_csv_to_cols(path)

    if header is None:
        if not cols:
            return np.zeros(0, dtype=float)
        arr = np.array(cols, dtype=float)
        series = arr if arr.ndim == 1 else arr[:, -1]
        return compute_running_ratio_from_vector(series)

    name_to_arr = {h: np.array(v, dtype=float) for h, v in zip(header, cols)}

    for cname in prefer_cols:
        if cname in name_to_arr and name_to_arr[cname].size:
            return compute_running_ratio_from_vector(name_to_arr[cname])

    # common column names
    for cname in ["w_ratio", "ratio", "cumulative", "w", "w_cum", "cum_ratio"]:
        if cname in name_to_arr and name_to_arr[cname].size:
            return compute_running_ratio_from_vector(name_to_arr[cname])

    if "energy" in name_to_arr and name_to_arr["energy"].size:
        return compute_running_ratio_from_vector(name_to_arr["energy"])
    if "abs" in name_to_arr and name_to_arr["abs"].size:
        return compute_running_ratio_from_vector(np.square(name_to_arr["abs"]))
    if "real" in name_to_arr and "imag" in name_to_arr and name_to_arr["real"].size:
        return compute_running_ratio_from_vector(name_to_arr["real"] ** 2 + name_to_arr["imag"] ** 2)

    return compute_running_ratio_from_vector(name_to_arr[header[-1]])


def load_usort_increments(path: str) -> np.ndarray:
    """Return increments that sum to 1 (or empty) for the u-sorted FB curve."""
    header, cols = _read_csv_to_cols(path)

    if header is None:
        if not cols:
            return np.zeros(0, dtype=float)
        arr = np.array(cols, dtype=float)
        series = arr if arr.ndim == 1 else arr[:, -1]
        r = compute_running_ratio_from_vector(series)
        inc = np.diff(np.clip(r, 0.0, 1.0), prepend=0.0)
        return np.maximum(inc, 0.0)

    name_to_arr = {h: np.array(v, dtype=float) for h, v in zip(header, cols)}

    for cname in ["w_ratio_usort", "ratio_usort", "w_usort"]:
        if cname in name_to_arr and name_to_arr[cname].size:
            r = np.clip(name_to_arr[cname].astype(float), 0.0, 1.0)
            inc = np.diff(r, prepend=0.0)
            return np.maximum(inc, 0.0)

    # fallback: treat last column as ratio/energy
    r = compute_running_ratio_from_vector(name_to_arr[header[-1]])
    inc = np.diff(np.clip(r, 0.0, 1.0), prepend=0.0)
    return np.maximum(inc, 0.0)


def filter_increments_skip_zeros(inc: np.ndarray, abs_tol: float, rel_tol: float) -> np.ndarray:
    inc = np.asarray(inc, dtype=float)
    if inc.size == 0:
        return inc
    inc = np.nan_to_num(inc, nan=0.0, posinf=0.0, neginf=0.0)
    inc = np.maximum(inc, 0.0)
    m = float(inc.max()) if inc.size else 0.0
    rel_thresh = m * float(rel_tol)
    mask = (inc > float(abs_tol)) & (inc > rel_thresh)
    return inc[mask]


# ============================== PCA curve from NPZ ==============================

def pca_curve_from_npz_all_m(npz_path: str, allow_trimmed: bool) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """Return (ratio, error_msg)."""
    try:
        d = np.load(npz_path, allow_pickle=False)
    except Exception as e:
        return None, f"failed to load NPZ: {e}"

    if "order_flat" not in d:
        return None, "NPZ missing 'order_flat'"

    order_flat = np.asarray(d["order_flat"], dtype=int)

    # detect trimmed packs (rows_l*)
    trimmed = any(k.startswith("rows_l") for k in d.keys())
    if trimmed and not allow_trimmed:
        return None, "PCA NPZ appears trimmed (*_topK, has rows_l*). Re-export *_all or pass --allow-trimmed-pca."

    # Determine L and n_rad
    L_meta = int(d["meta_L"]) if "meta_L" in d else None
    if L_meta is not None and L_meta > 0 and order_flat.size % L_meta == 0:
        n_rad = int(order_flat.size // L_meta)
        L_eff = int(L_meta)
    else:
        # fallback: infer from A_l0 shape
        a_keys = sorted([k for k in d.keys() if k.startswith("A_l")], key=lambda s: int(re.findall(r"\d+", s)[0]))
        if not a_keys:
            return None, "NPZ has no A_l* arrays"
        n_rad = int(d[a_keys[0]].shape[0])
        # infer L_eff from highest ell present
        L_eff = max(int(re.findall(r"\d+", k)[0]) for k in a_keys) + 1

    # Build row lookups if trimmed
    row_lookup: Dict[int, Dict[int, int]] = {}
    if trimmed:
        for ell in range(L_eff):
            rows_key = f"rows_l{ell}"
            if rows_key in d:
                rows = np.asarray(d[rows_key], dtype=int).tolist()
                row_lookup[ell] = {int(rad): i for i, rad in enumerate(rows)}

    pieces: List[np.ndarray] = []
    for fi in order_flat:
        ell = int(fi // n_rad)
        rad = int(fi % n_rad)
        key = f"A_l{ell}"
        if key not in d:
            continue
        A = np.asarray(d[key])
        if trimmed and ell in row_lookup:
            j = row_lookup[ell].get(rad)
            if j is None:
                continue
            coeffs = A[j, :]
        else:
            if rad >= A.shape[0]:
                continue
            coeffs = A[rad, :]
        pieces.append((np.abs(coeffs) ** 2).astype(np.float64, copy=False))

    if not pieces:
        return None, "no coefficients found after applying order_flat"

    energies = np.concatenate(pieces)
    tot = float(energies.sum())
    if tot <= 0.0:
        return np.zeros_like(energies), None

    return np.cumsum(energies) / tot, None


# ============================== Plot worker ==============================

def process_one(
    key: str,
    base: str,
    pca_npz_path: str,
    fb_sorted_csv: Optional[str],
    fb_usort_csv: Optional[str],
    first_k: int,
    out_dir: str,
    fig_dpi: int,
    line_width: float,
    legend: bool,
    include_fb_lex: bool,
    usort_skip_zeros: bool,
    usort_zero_abs_tol: float,
    usort_zero_rel_tol: float,
    allow_trimmed_pca: bool,
    expect_n: int,
    L: int,
    eps_tag: str,
) -> Tuple[str, Optional[str], bool, str]:
    try:
        # PCA
        r_pca, err = pca_curve_from_npz_all_m(pca_npz_path, allow_trimmed=allow_trimmed_pca)
        if err or r_pca is None or r_pca.size == 0:
            return key, None, False, f"PCA: {err or 'empty curve'}"

        # FB lex ratio (used both for plotting and for deriving |a|-sorted if needed)
        r_lex = None
        if fb_usort_csv is not None:
            r_lex = load_running_ratio(fb_usort_csv, prefer_cols=["w_ratio_lex", "ratio_lex", "w_lex"])
            if r_lex.size == 0:
                r_lex = None

        # FB |a|-sorted (prefer precomputed CSV; otherwise derive from lex increments)
        r_fbm = None
        if fb_sorted_csv is not None:
            r_fbm = load_running_ratio(
                fb_sorted_csv,
                prefer_cols=["w_ratio_sorted", "w_ratio", "ratio", "cumulative"],
            )
            if r_fbm.size == 0:
                r_fbm = None

        if r_fbm is None and r_lex is not None:
            r_fbm = derive_mag_sorted_ratio_from_cumratio(r_lex)
            if r_fbm.size == 0:
                r_fbm = None

        # If lex wasn't present, try deriving |a|-sorted from the raw u-sorted ratio
        if r_fbm is None and fb_usort_csv is not None:
            r_us_raw = load_running_ratio(fb_usort_csv, prefer_cols=["w_ratio_usort", "ratio_usort", "w_usort"])
            if r_us_raw.size:
                r_fbm = derive_mag_sorted_ratio_from_cumratio(r_us_raw)
                if r_fbm.size == 0:
                    r_fbm = None

        # FB u-sorted + optional lex
        r_fbu = None
        r_fbl = None
        if fb_usort_csv is not None:
            # u-sorted
            if usort_skip_zeros:
                inc = load_usort_increments(fb_usort_csv)
                inc = filter_increments_skip_zeros(inc, abs_tol=usort_zero_abs_tol, rel_tol=usort_zero_rel_tol)
                if inc.size > 0:
                    r_fbu = np.cumsum(inc)
            else:
                r_fbu = load_running_ratio(fb_usort_csv, prefer_cols=["w_ratio_usort", "ratio_usort", "w_usort"])
                if r_fbu.size == 0:
                    r_fbu = None

            # lex
            if include_fb_lex:
                if r_lex is None:
                    r_lex = load_running_ratio(fb_usort_csv, prefer_cols=["w_ratio_lex", "ratio_lex", "w_lex"])
                    if r_lex.size == 0:
                        r_lex = None
                r_fbl = r_lex

        # must have at least one FB curve to compare against
        if r_fbm is None and r_fbu is None and r_fbl is None:
            return key, None, False, "No FB curve available for this volume"

        curves = [r_pca]
        labels = ["PCA (NPZ, all m)"]
        styles = [dict(linewidth=line_width, linestyle="-")]

        if r_fbm is not None:
            curves.append(r_fbm)
            labels.append("FB (|a|-sorted)")
            styles.append(dict(linewidth=line_width, linestyle="--"))
        if r_fbu is not None:
            curves.append(r_fbu)
            labels.append("FB (u-sorted)")
            styles.append(dict(linewidth=line_width, linestyle=":"))
        if r_fbl is not None:
            curves.append(r_fbl)
            labels.append("FB (lex)")
            styles.append(dict(linewidth=line_width, linestyle="-."))

        K = min(int(first_k), *[int(c.size) for c in curves])
        if K <= 0:
            return key, None, False, f"No data within first_k={first_k}"

        x = np.arange(1, K + 1)

        os.makedirs(out_dir, exist_ok=True)
        plt.figure(figsize=(6.8, 4.6))
        for c, lab, sty in zip(curves, labels, styles):
            plt.plot(x, c[:K], label=lab, **sty)

        plt.ylim(0.0, 1.01)
        plt.xlim(1, K)
        plt.grid(True, linestyle="--", alpha=0.5)

        # Minimal title: include base + key run params to avoid confusion across runs
        plt.title(f"{base}  (N={expect_n}, L={L}, eps={eps_tag}, first {K})")
        plt.xlabel("k")
        plt.ylabel("cumulative energy")

        if legend:
            plt.legend(loc="lower right", frameon=True)

        plt.tight_layout()

        out_png = os.path.join(out_dir, f"{key}_compare_PCA_FB_N={expect_n}_L={L}_eps={eps_tag}_first{K}.png")
        plt.savefig(out_png, dpi=int(fig_dpi))
        plt.close()

        return key, out_png, True, "ok"

    except Exception as e:
        return key, None, False, f"error: {e}"


# ============================== main ==============================

def main() -> int:
    args = parse_args()

    eps_tag = eps_to_tag(args.eps)

    pca_dir = args.pca_dir or f"mat_converted_N={args.expect_n}_matrix"
    fb_sorted_dir = args.fb_sorted_dir or f"mat_converted_N={args.expect_n}_energy_fle"
    fb_usort_dir = args.fb_usort_dir or f"mat_converted_N={args.expect_n}_FBexpansions"

    out_dir = args.out_dir
    if out_dir is None:
        # Root folder for plots (requested convention)
        plots_root = f"mat_converted_N={args.expect_n}_plots"
        # Subfolder for this run (keep run params in the name to avoid confusion)
        sub = f"compare_PCA_FB_L={args.L}_eps={eps_tag}_first{args.first_k}"
        out_dir = os.path.join(plots_root, sub)

    if not os.path.isdir(pca_dir):
        print(f"[error] pca-dir not found: {pca_dir}", file=sys.stderr)
        return 2

    # Scan inputs
    pca_map = scan_pca_npz(pca_dir, want_L=args.L, want_eps_tag=eps_tag)
    if not pca_map:
        print(f"[error] No PCA coeff NPZs found in: {pca_dir}", file=sys.stderr)
        return 2

    fb_sorted_map: Dict[str, FileChoice] = {}
    if os.path.isdir(fb_sorted_dir):
        fb_sorted_map = scan_energy_csv(fb_sorted_dir)
    else:
        print(
            f"[warn] fb-sorted-dir not found (will derive |a|-sorted from FBexpansions if available): {fb_sorted_dir}"
        )

    fb_usort_map: Dict[str, FileChoice] = {}
    if os.path.isdir(fb_usort_dir):
        fb_usort_map = scan_energy_csv(fb_usort_dir)
    else:
        print(f"[warn] fb-usort-dir not found (skipping u-sorted/lex FB curve): {fb_usort_dir}")

    # Build work list: require PCA and at least one FB curve
    bases_all = sorted(set(pca_map.keys()) & (set(fb_sorted_map.keys()) | set(fb_usort_map.keys())))

    # Apply targets filter
    bases = [k for k in bases_all if _matches_any_target(pca_map[k].base, args.targets)]

    print(f"[scan] PCA NPZs:        {len(pca_map)}  in '{pca_dir}'")
    print(f"[scan] FB sorted CSVs:  {len(fb_sorted_map)}  in '{fb_sorted_dir}'")
    print(f"[scan] FB usort CSVs:   {len(fb_usort_map)}  in '{fb_usort_dir}'")
    print(f"[match] usable volumes: {len(bases)}  (after targets filter; total matches={len(bases_all)})")

    if not bases:
        # Diagnostic examples
        missing_in_fb = sorted(set(pca_map.keys()) - (set(fb_sorted_map.keys()) | set(fb_usort_map.keys())))
        missing_in_pca = sorted((set(fb_sorted_map.keys()) | set(fb_usort_map.keys())) - set(pca_map.keys()))
        print("[stop] No common volumes found.")
        print(f"       Examples missing in FB:  {[pca_map[k].base for k in missing_in_fb[:10]]}")
        print(f"       Examples missing in PCA: {missing_in_pca[:10]}")
        return 0

    if args.dry_run:
        print("[dry-run] Example matches:")
        for k in bases[:10]:
            fb1 = fb_sorted_map.get(k)
            fb2 = fb_usort_map.get(k)
            print(f"  • {pca_map[k].base}: PCA={os.path.basename(pca_map[k].path)} | FBsorted={os.path.basename(fb1.path) if fb1 else '-'} | FBusort={os.path.basename(fb2.path) if fb2 else '-'}")
        print(f"[dry-run] Would write plots to: {out_dir}")
        return 0

    # Workers
    if args.n_jobs in (-1, None, 0):
        cpu = os.cpu_count() or 2
        workers = max(1, cpu - 1)
    else:
        workers = max(1, int(args.n_jobs))

    print(f"[mp] workers={workers}")
    print(f"[out] out-dir='{out_dir}'")

    submitted = successes = failures = 0

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = []
        for k in bases:
            pca = pca_map[k]
            fb_s = fb_sorted_map.get(k)
            fb_u = fb_usort_map.get(k)
            futures.append(
                ex.submit(
                    process_one,
                    k,
                    pca.base,
                    pca.path,
                    fb_s.path if fb_s else None,
                    fb_u.path if fb_u else None,
                    int(args.first_k),
                    out_dir,
                    int(args.dpi),
                    float(args.line_width),
                    bool(args.legend),
                    bool(args.include_fb_lex),
                    bool(args.usort_skip_zeros),
                    float(args.usort_zero_abs_tol),
                    float(args.usort_zero_rel_tol),
                    bool(args.allow_trimmed_pca),
                    int(args.expect_n),
                    int(args.L),
                    eps_tag,
                )
            )
            submitted += 1

        it = as_completed(futures)
        if args.pbar:
            it = tqdm(it, total=len(futures), desc="plot", unit="vol", dynamic_ncols=True)

        for fut in it:
            key, out_png, ok, msg = fut.result()
            if ok:
                successes += 1
                if args.pbar:
                    tqdm.write(f"[ok]   {key} -> {out_png}")
                else:
                    print(f"[ok]   {key} -> {out_png}")
            else:
                failures += 1
                if args.pbar:
                    tqdm.write(f"[skip] {key}: {msg}")
                else:
                    print(f"[skip] {key}: {msg}")

    print(f"[done] submitted={submitted} successes={successes} failures={failures}")
    print(f"[out]  figures in: {out_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
