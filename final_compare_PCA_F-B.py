#!/usr/bin/env python3
"""
final_compare_PCA_F-B_from_npz.py

Compares, per volume:
  - PCA cumulative energy ratio computed from the **NPZ coefficients** (no sum over m).
    Uses *_coeffs_centered_global_all.npz and the stored global order (order_flat).
  - FB magnitude-sorted cumulative energy ratio (from mat_converted_N={NN}_energy_fle/*.csv)
  - (optional) FB u-sorted cumulative energy ratio (from mat_converted_N={NN}_FBexpansions/*.csv)

Notes:
  • The PCA curve here treats each coefficient A_{ell, r, m} as a separate step.
    We iterate (ell, r) following the global order_flat, and within each row append all m.
  • Fixed name-normalization to strip trailing "_all" / "_topK" so intersections are correct.
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
NN = 22
FIRST_K = 100

PCA_DIR        = f"mat_converted_N={NN}_matrix"         # *_coeffs_centered_global_all.npz
FB_SORTED_DIR  = f"mat_converted_N={NN}_energy_fle"     # magnitude-sorted FB (|a| sorting)
FB_USORT_DIR   = f"mat_converted_N={NN}_FBexpansions"   # u-sorted (expects 'w_ratio_usort' or per-coef columns)

OUT_DIR        = f"plots_pcaNPZ_vs_fb_first{FIRST_K}_{NN}_with_usort"
FIG_DPI        = 150
NUM_WORKERS    = None  # None -> use os.cpu_count()-1
LINE_WIDTH     = 2.0

# Treat increments below both thresholds as "zero" and skip them for the u-sorted curve.
USORT_ZERO_ABS_TOL = 1e-5
USORT_ZERO_REL_TOL = 1e-5
# ------------------------------------------------------

# ---------- NAME NORMALIZATION (BUGFIX: add 'all' & 'topk') ----------
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
    r"centered_global", r"centered_pervol", r"uncentered",
    r"all", r"topk"  # <— important
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
        "first", "k", "top", "v", "ver", "run", "copy", "dup", "final", "clean", "rev", "draft",
        "all", "topk"
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

def map_key_to_npz_dict(folder: str) -> Dict[str, str]:
    # prefer *_coeffs_centered_global_all.npz, else *_coeffs_centered_global_topK.npz
    pats = [
        "*_coeffs_centered_global_all.npz",
        "*_coeffs_centered_global_topK.npz",
        "*_coeffs_centered_global*.npz",
    ]
    paths = []
    for pat in pats:
        paths.extend(glob.glob(os.path.join(folder, pat)))
    candidates: Dict[str, List[str]] = {}
    for path in paths:
        k = normalize_key(path)
        candidates.setdefault(k, []).append(path)
    chosen: Dict[str, str] = {}
    for k, ps in candidates.items():
        # prefer 'all' file if present
        all_first = sorted(ps, key=lambda p: (0 if "all.npz" in p.lower() else 1, _suffix_score(os.path.splitext(os.path.basename(p))[0]), len(p)))
        chosen[k] = all_first[0]
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

def _read_csv_to_cols(path: str) -> Tuple[Optional[List[str]], List[List[float]]]:
    with open(path, "r", newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return None, []
    if looks_like_header(rows[0]):
        header = [h.strip().lower() for h in rows[0]]
        data_rows = rows[1:]
        cols = [[] for _ in header]
        for r in data_rows:
            if not r or all((c or "").strip() == "" for c in r):
                continue
            for i in range(len(header)):
                val = safe_float(r[i]) if i < len(r) else np.nan
                cols[i].append(val)
        return header, cols
    # no header -> treat each row as numeric
    numeric = []
    for r in rows:
        vals = [safe_float(x) for x in r if (x or "").strip() != ""]
        if vals:
            numeric.append(vals)
    return None, numeric

def load_running_ratio(path: str, prefer_cols: Optional[List[str]] = None) -> np.ndarray:
    """Generic loader for FB(|a|-sorted) CSVs."""
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
    for cname in ["w_ratio", "ratio", "cumulative", "w", "w_cum", "cum_ratio"]:
        if cname in name_to_arr and name_to_arr[cname].size:
            return compute_running_ratio_from_vector(name_to_arr[cname])
    if "energy" in name_to_arr and name_to_arr["energy"].size:
        return compute_running_ratio_from_vector(name_to_arr["energy"])
    if "abs" in name_to_arr and name_to_arr["abs"].size:
        return compute_running_ratio_from_vector(np.square(name_to_arr["abs"]))
    if "real" in name_to_arr and "imag" in name_to_arr and name_to_arr["real"].size:
        return compute_running_ratio_from_vector(name_to_arr["real"]**2 + name_to_arr["imag"]**2)
    return compute_running_ratio_from_vector(name_to_arr[header[-1]])

# ---------- u-sorted increments loader ----------
def load_usort_increments(path: str) -> np.ndarray:
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

    energy_candidates = []
    for cname in ["energy_usort", "energy"]:
        if cname in name_to_arr and name_to_arr[cname].size:
            energy_candidates.append(name_to_arr[cname].astype(float))
            break
    if not energy_candidates and "abs_usort" in name_to_arr and name_to_arr["abs_usort"].size:
        energy_candidates.append(np.square(name_to_arr["abs_usort"].astype(float)))
    if not energy_candidates and ("real_usort" in name_to_arr and "imag_usort" in name_to_arr
                                  and name_to_arr["real_usort"].size and name_to_arr["imag_usort"].size):
        energy_candidates.append(name_to_arr["real_usort"].astype(float)**2 +
                                 name_to_arr["imag_usort"].astype(float)**2)
    if not energy_candidates and "abs" in name_to_arr and name_to_arr["abs"].size:
        energy_candidates.append(np.square(name_to_arr["abs"].astype(float)))
    if not energy_candidates and ("real" in name_to_arr and "imag" in name_to_arr
                                  and name_to_arr["real"].size and name_to_arr["imag"].size):
        energy_candidates.append(name_to_arr["real"].astype(float)**2 +
                                 name_to_arr["imag"].astype(float)**2)

    if energy_candidates:
        en = np.nan_to_num(energy_candidates[0], nan=0.0, posinf=0.0, neginf=0.0)
        en = np.maximum(en, 0.0)
        tot = en.sum()
        if tot <= 0:
            return np.zeros_like(en)
        inc = en / tot  # ratio increments that sum to 1
        return inc

    r = compute_running_ratio_from_vector(name_to_arr[header[-1]])
    inc = np.diff(np.clip(r, 0.0, 1.0), prepend=0.0)
    return np.maximum(inc, 0.0)

def filter_increments_skip_zeros(inc: np.ndarray) -> np.ndarray:
    inc = np.asarray(inc, dtype=float)
    if inc.size == 0:
        return inc
    inc = np.nan_to_num(inc, nan=0.0, posinf=0.0, neginf=0.0)
    inc = np.maximum(inc, 0.0)
    m = inc.max() if inc.size else 0.0
    rel_thresh = m * USORT_ZERO_REL_TOL
    mask = (inc > USORT_ZERO_ABS_TOL) & (inc > rel_thresh)
    return inc[mask]

# ---------- PCA from NPZ: ALL (ell,r,m) coefficients ----------
def pca_curve_from_npz_all_m(npz_path: str) -> np.ndarray:
    """
    Build cumulative energy ratio using ALL coefficients A_{ell, r, m}.
    Order: follow global 'order_flat' across (ell, r); within each row, append all m.
    """
    d = np.load(npz_path)
    order_flat = d["order_flat"].astype(int)
    # infer n_rad from any A_l? use l=0 if present, else smallest key
    l_keys = sorted([k for k in d.keys() if k.startswith("A_l")],
                    key=lambda s: int(re.findall(r"\d+", s)[0]))
    if not l_keys:
        return np.zeros(0, dtype=float)
    n_rad = d[l_keys[0]].shape[0]
    # build per-coefficient energies in the required order
    pieces = []
    for fi in order_flat:
        ell = int(fi // n_rad)
        rad = int(fi %  n_rad)
        key = f"A_l{ell}"
        if key not in d:
            # if the pack was row-trimmed for some ell, skip gracefully
            continue
        A_row = d[key]
        if rad >= A_row.shape[0]:
            continue
        # take all m (columns 0..2ell)
        coeffs = A_row[rad, :]
        e = np.abs(coeffs)**2
        pieces.append(e.astype(np.float64, copy=False))
    if not pieces:
        return np.zeros(0, dtype=float)
    energies = np.concatenate(pieces)
    tot = float(energies.sum())
    if tot <= 0:
        return np.zeros_like(energies)
    return np.cumsum(energies) / tot

# ---------- PLOTTING WORKERS ----------
def ensure_outdir():
    os.makedirs(OUT_DIR, exist_ok=True)

def build_maps() -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
    # PCA: NPZ packs with coefficients (prefer *_all.npz)
    pca_npz_map = map_key_to_npz_dict(PCA_DIR)
    # FB magnitude-sorted: all CSVs in energy_fle dir
    fbs_map = map_key_to_file_dict(FB_SORTED_DIR, pattern="*.csv")
    # FB u-sorted: all CSVs in FBexpansions dir
    fbu_map = map_key_to_file_dict(FB_USORT_DIR, pattern="*.csv")
    return pca_npz_map, fbs_map, fbu_map

def build_work_items(pca_map: Dict[str, str], fbs_map: Dict[str, str], fbu_map: Dict[str, str]) -> List[Tuple[str, str, str, Optional[str]]]:
    common = sorted(set(pca_map.keys()) & set(fbs_map.keys()))
    # use u-sorted if available for the same key; else None
    items = []
    for k in common:
        items.append((k, pca_map[k], fbs_map[k], fbu_map.get(k)))
    return items

def process_one(key: str,
                pca_npz_path: str,
                fb_sorted_path: str,
                fb_usort_path: Optional[str],
                first_k: int,
                out_dir: str,
                fig_dpi: int) -> Tuple[str, Optional[str], bool, str]:
    try:
        # PCA from NPZ (ALL m)
        r_pca = pca_curve_from_npz_all_m(pca_npz_path)
        if r_pca.size == 0:
            return (key, None, False, "PCA NPZ yielded no coefficients")

        # FB magnitude-sorted
        r_fbs = load_running_ratio(fb_sorted_path, prefer_cols=["w_ratio_sorted", "w_ratio", "ratio", "cumulative"])
        if r_fbs.size == 0:
            return (key, None, False, "FB(|a|) CSV empty/unreadable")

        # FB u-sorted (optional)
        inc_fbu = None
        r_fbu = None
        if fb_usort_path:
            inc_fbu = load_usort_increments(fb_usort_path)
            inc_fbu = filter_increments_skip_zeros(inc_fbu)
            if inc_fbu.size > 0:
                r_fbu = np.cumsum(inc_fbu)

        # K = min length among available curves (respect first_k)
        curves = [r_pca, r_fbs] + ([r_fbu] if r_fbu is not None else [])
        K = min(first_k, *[c.size for c in curves if c is not None])
        if K <= 0:
            return (key, None, False, f"no data within first {first_k} coefficients")

        x = np.arange(1, K+1)

        plt.figure(figsize=(6.8, 4.6))
        #plt.plot(x, r_pca[:K], linewidth=LINE_WIDTH, label="PCA (NPZ, all m)")
        plt.plot(x, r_pca[:K], linewidth=LINE_WIDTH)
        #plt.plot(x, r_fbs[:K], linewidth=LINE_WIDTH, linestyle="--", label="FB (|a|-sorted)")
        plt.plot(x, r_fbs[:K], linewidth=LINE_WIDTH, linestyle="--")
        if r_fbu is not None and r_fbu.size >= K:
            #plt.plot(x, r_fbu[:K], linewidth=LINE_WIDTH, linestyle=":", label="FB (u-sorted, skip ~0)")
            plt.plot(x, r_fbu[:K], linewidth=LINE_WIDTH, linestyle=":")
        plt.grid(True, linestyle="--", alpha=0.5)
        #plt.legend()
        plt.tight_layout()

        out_png = os.path.join(out_dir, f"{key}_pcaNPZ_vs_fb_N={NN}_first{first_k}.png")
        plt.savefig(out_png, dpi=fig_dpi)
        plt.close()
        return (key, out_png, True, "ok")
    except Exception as e:
        return (key, None, False, f"error: {e}")

def main():
    ensure_outdir()

    # Sanity: directories must exist
    for label, d in [("PCA_DIR", PCA_DIR), ("FB_SORTED_DIR", FB_SORTED_DIR)]:
        if not os.path.isdir(d):
            print(f"[error] {label} not found: {d}")
            return
    if not os.path.isdir(FB_USORT_DIR):
        print(f"[warn] FB_USORT_DIR not found: {FB_USORT_DIR} (u-sorted curve will be skipped)")

    pca_map, fbs_map, fbu_map = build_maps()
    print(f"[scan] PCA coeff NPZs:          {len(pca_map)} in '{PCA_DIR}'")
    print(f"[scan] FB (|a|-sorted) CSVs:    {len(fbs_map)} in '{FB_SORTED_DIR}'")
    print(f"[scan] FB (u-sorted) CSVs:      {len(fbu_map)} in '{FB_USORT_DIR}'")

    work = build_work_items(pca_map, fbs_map, fbu_map)
    if not work:
        # helpful diagnostics
        missing_in_fbs = sorted(set(pca_map.keys()) - set(fbs_map.keys()))
        missing_in_pca = sorted(set(fbs_map.keys()) - set(pca_map.keys()))
        print("[stop] No common volumes found after normalization.")
        print(f"       Examples missing in FB(|a|): {missing_in_fbs[:10]}")
        print(f"       Examples missing in PCA(NPZ): {missing_in_pca[:10]}")
        return

    print(f"[match] Common volumes (PCA NPZ ∩ FB|a|): {len(work)}")
    print(f"[usort] Zero-skip thresholds: ABS>{USORT_ZERO_ABS_TOL} and REL>{USORT_ZERO_REL_TOL}*max(increment)")

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
        for key, pca_npz_path, fbs_path, fbu_path in work:
            fut = ex.submit(process_one, key, pca_npz_path, fbs_path, fbu_path, FIRST_K, OUT_DIR, FIG_DPI)
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
