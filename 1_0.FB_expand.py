#!/usr/bin/env python3
"""
0_2.FB_expand.py

Expand volumes in the Fourier–Bessel basis (via FLEBasis3D) and write, for each
selected volume, a CSV:

    k,w_ratio_lex,w_ratio_usort

where:
  - w_ratio_lex   : cumulative energy in fixed (n, ℓ, m) lexicographic order
  - w_ratio_usort : cumulative energy when (ℓ, n) blocks are ordered by
                    increasing u_{ℓ,s} (true u-sorting; ties over m)

This is a refactor of final_expand_FB.py to match the argparse/CLI style used by
0_1.covariance_matrix.py.

Typical usage:
  python3 0_2.FB_expand.py --expect-n 20 --L 10 --eps 1e-6 --solver nvidia_torch

Restrict to a subset of volumes (basenames or glob patterns):
  python3 0_2.FB_expand.py --targets 1f* 2a* some_exact_basename

Notes:
  - The script expands each volume with FLEBasis3D.step1/step2 (FB expansion).
  - It attempts to obtain the u_{ℓ,s} table from the FLE object; if unavailable,
    it computes zeros of spherical Bessel j_ℓ via SciPy brentq (cached).
"""

import os
import sys

# CRITICAL FIX FOR MACOS SEGFAULT / OVERSUBSCRIPTION:
# Control OpenMP/MKL threading before importing numpy/torch/scipy.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import glob
import fnmatch
import argparse
import traceback
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.io import loadmat
from scipy.special import spherical_jn
from scipy.optimize import brentq

from tqdm.auto import tqdm
from fle_3d import FLEBasis3D


# =================== DEFAULTS (CLI-driven; similar conventions to 0_1) ===================
DEFAULT_EXPECT_N = 22
DEFAULT_L = 20
DEFAULT_EPS = 1e-6
DEFAULT_SOLVER = "nvidia_torch"
DEFAULT_PATTERN = "*.mat"
DEFAULT_TARGETS: List[str] = []
DEFAULT_N_JOBS = -1
DEFAULT_NORMALIZE = True
DEFAULT_PBAR = True
# ========================================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="0_2.FB_expand.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Expand volumes in the Fourier–Bessel basis (FLEBasis3D) and write per-volume "
            "cumulative energy CSVs in lexicographic order and true u-sorted order."
        ),
    )

    # High-level / paths
    p.add_argument(
        "--expect-n",
        type=int,
        default=DEFAULT_EXPECT_N,
        help="Used only to form default folder names and warn if mismatch vs data.",
    )
    p.add_argument(
        "--in-dir",
        default=None,
        help="Input directory with .mat files. Default: mat_converted_N=<expect-n>",
    )
    p.add_argument(
        "--out-dir",
        default=None,
        help="Output directory. Default: mat_converted_N=<expect-n>_FBexpansions",
    )

    # Core parameters
    p.add_argument("--L", type=int, default=DEFAULT_L, help="Bandlimit (number of ℓ blocks).")
    p.add_argument("--eps", type=float, default=DEFAULT_EPS, help="FLEBasis3D eps.")
    p.add_argument(
        "--solver",
        default=DEFAULT_SOLVER,
        help="Spherical harmonics solver for FLEBasis3D (e.g., nvidia_torch).",
    )

    # File selection
    p.add_argument(
        "--pattern",
        default=DEFAULT_PATTERN,
        help="Glob pattern for .mat files within in-dir.",
    )
    p.add_argument(
        "--targets",
        nargs="*",
        default=DEFAULT_TARGETS,
        help=(
            "Restrict which volumes to process. Each entry may be an exact basename (no .mat) "
            "or a glob pattern (e.g. 1f*, *foo*). Empty means all matching .mat files."
        ),
    )

    # Compute / performance
    p.add_argument(
        "--n-jobs",
        type=int,
        default=DEFAULT_N_JOBS,
        help="Worker threads for expansion (-1 means all cores).",
    )

    # Normalization
    p.add_argument(
        "--normalize",
        dest="normalize",
        action="store_true",
        default=DEFAULT_NORMALIZE,
        help="Normalize each volume by max(abs(vol)) before expansion.",
    )
    p.add_argument(
        "--no-normalize",
        dest="normalize",
        action="store_false",
        help="Do not normalize volumes before expansion.",
    )

    # Progress bars
    p.add_argument("--pbar", dest="pbar", action="store_true", default=DEFAULT_PBAR, help="Enable progress bars.")
    p.add_argument("--no-pbar", dest="pbar", action="store_false", help="Disable progress bars.")

    return p.parse_args()


# ----------------------- IO helpers -----------------------

def get_mat_list(directory: str, pattern: str) -> List[str]:
    return sorted(glob.glob(os.path.join(directory, pattern)))


def detect_first_data_key(mat_dict: dict) -> str:
    for k in mat_dict:
        if not k.startswith("__"):
            return k
    raise KeyError("No data key found in .mat file (only __* keys present).")


def filter_targets(mats: List[str], targets: List[str]) -> List[str]:
    if not targets:
        return mats

    exact_names = set()
    patterns: List[str] = []
    for t in targets:
        t = (t or "").strip()
        if not t:
            continue
        if any(ch in t for ch in "*?[]"):
            patterns.append(t)
        else:
            exact_names.add(os.path.splitext(os.path.basename(t))[0].lower())

    picked: List[str] = []
    for pth in mats:
        base = os.path.splitext(os.path.basename(pth))[0]
        base_l = base.lower()
        if base_l in exact_names:
            picked.append(pth)
            continue
        if any(fnmatch.fnmatch(base, pat) or fnmatch.fnmatch(base + ".mat", pat) for pat in patterns):
            picked.append(pth)

    return sorted(picked)


def load_volume(path: str, key: str, normalize: bool = True) -> np.ndarray:
    data = loadmat(path)
    if key not in data:
        # fall back to per-file detection if the first file's key isn't present
        key = detect_first_data_key(data)
    v = np.ascontiguousarray(data[key], dtype=np.float64)
    if not normalize:
        return v
    vmax = float(np.max(np.abs(v)))
    return v if vmax == 0.0 else (v / vmax)


# ----------------------- u_{ℓ,s} table -----------------------

_U_CACHE: Dict[Tuple[int, int], np.ndarray] = {}


def _try_fetch_u_table_from_fle(fle: FLEBasis3D, n_rad: int, L_eff: int) -> Optional[np.ndarray]:
    """Attempt to read a 2D (n_rad, L_eff) table from FLEBasis3D; return None if unavailable."""

    # 1) direct 2D arrays on the object
    attr_candidates = [
        "u_table",
        "u",
        "u_ls",
        "ul_s",
        "kappa",
        "k_ls",
        "roots",
        "radial_roots",
        "rho_ls",
    ]
    for name in attr_candidates:
        if hasattr(fle, name):
            obj = getattr(fle, name)
            arr = obj() if callable(obj) else obj
            try:
                U = np.asarray(arr, dtype=float)
            except Exception:
                U = None
            if U is None or U.ndim != 2:
                continue

            # accept (n_rad, L_eff) or (L_eff, n_rad)
            if U.shape == (n_rad, L_eff):
                return U.copy()
            if U.shape == (L_eff, n_rad):
                return U.T.copy()

            # permissive slice
            if U.shape[0] >= n_rad and U.shape[1] >= L_eff:
                return U[:n_rad, :L_eff].copy()
            if U.shape[0] >= L_eff and U.shape[1] >= n_rad:
                return U[:L_eff, :n_rad].T.copy()

    # 2) per-ℓ callables returning 1D arrays of length ≥ n_rad
    fn_candidates = ["radial_zeros", "radial_roots", "zeros_for_ell", "bessel_zeros"]
    for name in fn_candidates:
        if hasattr(fle, name):
            fn = getattr(fle, name)
            if callable(fn):
                cols = []
                ok = True
                for ell in range(L_eff):
                    try:
                        col = np.asarray(fn(ell), dtype=float)
                    except Exception:
                        ok = False
                        break
                    if col.ndim == 0 or col.size < n_rad:
                        ok = False
                        break
                    cols.append(col[:n_rad])
                if ok and cols:
                    return np.stack(cols, axis=1)  # (n_rad, L_eff)

    return None


def _compute_u_table_brent(n_rad: int, L_eff: int) -> np.ndarray:
    """
    Compute U[s_idx, ell] = u_{ℓ,s} with s_idx = 0..n_rad-1, ell = 0..L_eff-1
    by finding zeros of spherical_jn(ell, x) via brentq.

    Brackets: around (s + ℓ/2)*π with ±0.5π pad; widen if no sign change.
    """

    U = np.empty((n_rad, L_eff), dtype=float)
    pi = np.pi

    for ell in range(L_eff):
        # local f for this ell
        def f(x):
            return spherical_jn(ell, x)

        for s_idx in range(n_rad):
            s = s_idx + 1  # human indexing
            center = (s + 0.5 * ell) * pi
            a = max(center - 0.5 * pi, 1e-6)
            b = center + 0.5 * pi

            fa = f(a)
            fb = f(b)

            if np.sign(fa) == np.sign(fb):
                widened = False
                for k in range(1, 6):
                    aa = max(a - k * 0.25 * pi, 1e-6)
                    bb = b + k * 0.25 * pi
                    fa = f(aa)
                    fb = f(bb)
                    if np.sign(fa) != np.sign(fb):
                        a, b = aa, bb
                        widened = True
                        break
                if not widened:
                    # last resort: sample on a grid and locate first sign flip
                    xs = np.linspace(a, b + 2 * pi, 200)
                    ys = f(xs)
                    idx = np.where(np.sign(ys[:-1]) * np.sign(ys[1:]) < 0)[0]
                    if idx.size:
                        a, b = float(xs[idx[0]]), float(xs[idx[0] + 1])
                        fa = f(a)
                        fb = f(b)
                    else:
                        raise RuntimeError(f"Could not bracket root for (ell={ell}, s={s})")

            # Guard against exact zero at endpoints
            if fa == 0.0:
                a *= 0.999
            if fb == 0.0:
                b *= 1.001

            root = brentq(f, a, b, maxiter=100, xtol=1e-12, rtol=1e-12)
            U[s_idx, ell] = root

    return U


def get_u_table(fle: FLEBasis3D, n_rad: int, L_eff: int, pbar: bool = True) -> np.ndarray:
    """Get u_{ℓ,s} as (n_rad, L_eff). Try FLE first; otherwise compute with brentq (cached)."""

    key = (n_rad, L_eff)
    if key in _U_CACHE:
        return _U_CACHE[key]

    U = _try_fetch_u_table_from_fle(fle, n_rad, L_eff)
    if U is None:
        if pbar:
            print("[u-table] FLEBasis3D did not expose u_{ℓ,s}; computing zeros via brentq fallback.")
        U = _compute_u_table_brent(n_rad, L_eff)

    _U_CACHE[key] = U
    return U


# ----------------------- per-volume worker -----------------------

_tls = threading.local()


def _get_thread_fle(N_vol: int, L: int, eps: float, solver: str) -> FLEBasis3D:
    fle = getattr(_tls, "fle", None)
    if fle is None:
        fle = FLEBasis3D(
            N=N_vol,
            bandlimit=L,
            eps=eps,
            max_l=L,
            mode="complex",
            sph_harm_solver=solver,
            reduce_memory=True,
        )
        _tls.fle = fle
    return fle


def compute_energy_ratios(
    b: np.ndarray,  # (n_rad, L_eff, 2ℓ+1)
    U: np.ndarray,  # (n_rad, L_eff)
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (ratios_lex, ratios_usort), both length = total #coeffs across (n,ℓ,m)."""

    n_rad = int(b.shape[0])
    L_eff = int(b.shape[1])

    # --- lexicographic energies (n, ℓ, m)
    energy_lex_chunks: List[np.ndarray] = []
    for n_idx in range(n_rad):
        for ell in range(L_eff):
            m_len = 2 * ell + 1
            coeffs = b[n_idx, ell, :m_len]
            energy_lex_chunks.append(np.abs(coeffs) ** 2)

    energies_lex = (
        np.concatenate(energy_lex_chunks).astype(np.float64, copy=False)
        if energy_lex_chunks
        else np.empty((0,), dtype=np.float64)
    )

    total = float(energies_lex.sum())
    if total <= 0.0:
        return np.zeros_like(energies_lex), np.zeros_like(energies_lex)

    ratios_lex = np.cumsum(energies_lex) / total

    # --- true u-sorting: order (ℓ, n) by increasing u_{ℓ,s}; ties: iterate m
    order_pairs = sorted(
        ((float(U[n_idx, ell]), n_idx, ell) for ell in range(L_eff) for n_idx in range(n_rad)),
        key=lambda t: t[0],
    )

    energy_us_chunks: List[np.ndarray] = []
    for _, n_idx, ell in order_pairs:
        m_len = 2 * ell + 1
        coeffs = b[n_idx, ell, :m_len]
        energy_us_chunks.append(np.abs(coeffs) ** 2)

    energies_us = np.concatenate(energy_us_chunks).astype(np.float64, copy=False)
    ratios_us = np.cumsum(energies_us) / total

    return ratios_lex, ratios_us


def process_one(
    m_path: str,
    key: str,
    N_vol: int,
    L: int,
    eps: float,
    solver: str,
    out_dir: str,
    U: np.ndarray,
    expected_shape: Tuple[int, int],
    normalize: bool,
) -> Tuple[str, Optional[str], Optional[int], Optional[str]]:
    """Returns: (basename, out_csv, ncoef, error_msg)."""

    basename = os.path.splitext(os.path.basename(m_path))[0]

    try:
        fle = _get_thread_fle(N_vol, L, eps, solver)
        vol = load_volume(m_path, key=key, normalize=normalize)

        vmax = float(np.max(np.abs(vol)))
        if vmax == 0.0:
            return basename, None, None, f"Skip {basename}: volume is all zeros"

        z = fle.step1(vol)
        b = fle.step2(z)  # (n_rad, L_eff, 2ℓ+1)

        n_rad = int(b.shape[0])
        L_eff = int(b.shape[1])
        if (n_rad, L_eff) != expected_shape:
            raise RuntimeError(
                f"Unexpected coeff shape (n_rad={n_rad}, L_eff={L_eff}); expected {expected_shape}."
            )

        ratios_lex, ratios_us = compute_energy_ratios(b=b, U=U)

        os.makedirs(out_dir, exist_ok=True)
        out_csv = os.path.join(out_dir, f"{basename}_coeff_energy.csv")
        with open(out_csv, "w") as f:
            f.write("k,w_ratio_lex,w_ratio_usort\n")
            for k in range(1, ratios_lex.size + 1):
                f.write(f"{k},{ratios_lex[k-1]:.12g},{ratios_us[k-1]:.12g}\n")

        return basename, out_csv, int(ratios_lex.size), None

    except Exception as e:
        tb = traceback.format_exc(limit=8)
        return basename, None, None, f"{e.__class__.__name__}: {e}\n{tb}"


# ----------------------- main -----------------------

def main(
    in_dir: str,
    out_dir: str,
    L: int,
    eps: float,
    solver: str,
    pattern: str,
    targets: List[str],
    n_jobs: int,
    normalize: bool,
    pbar: bool,
    expect_N: Optional[int],
) -> None:
    mats_all = get_mat_list(in_dir, pattern=pattern)
    mats = filter_targets(mats_all, targets)

    if not mats:
        raise RuntimeError(
            f"No .mat files to process. in_dir='{in_dir}', pattern='{pattern}', targets={targets}"
        )

    # discover key & grid from first selected file
    samp = loadmat(mats[0])
    key = detect_first_data_key(samp)
    N_vol = int(np.ascontiguousarray(samp[key]).shape[0])
    if expect_N is not None and int(expect_N) != N_vol:
        print(
            f"[warn] Folder expect-n={expect_N} but first volume is {N_vol}. "
            f"Proceeding with N_vol={N_vol} for computations."
        )

    print(f"[build] Volumes={len(mats)} | N_vol={N_vol}, L={L}, eps={eps} | solver='{solver}'")
    print(f"[build] in_dir='{in_dir}'")
    print(f"[build] out_dir='{out_dir}'")

    # Probe expansion once to determine (n_rad, L_eff) and build U-table once.
    fle_probe = FLEBasis3D(
        N=N_vol,
        bandlimit=L,
        eps=eps,
        max_l=L,
        mode="complex",
        sph_harm_solver=solver,
        reduce_memory=True,
    )

    v0 = load_volume(mats[0], key=key, normalize=normalize)
    z0 = fle_probe.step1(v0)
    b0 = fle_probe.step2(z0)
    n_rad0 = int(b0.shape[0])
    L_eff0 = int(b0.shape[1])

    if L_eff0 != L:
        print(f"[warn] FLE returned L_eff={L_eff0} (requested L={L}). Using L_eff={L_eff0} for ordering.")

    U = get_u_table(fle_probe, n_rad=n_rad0, L_eff=L_eff0, pbar=pbar)

    expected_shape = (n_rad0, L_eff0)

    # workers
    workers = (os.cpu_count() or 1) if (n_jobs in (-1, None, 0)) else int(n_jobs)
    if workers < 1:
        workers = 1

    if "nvidia" in str(solver).lower() and workers > 1:
        print(
            f"[warn] solver='{solver}' looks GPU-backed. If you see instability or GPU OOM, "
            f"try --n-jobs 1. (Current workers={workers})"
        )

    if pbar:
        print("[step] Expanding volumes & writing CSVs …")

    done = 0
    errors = 0

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [
            ex.submit(
                process_one,
                m,
                key,
                N_vol,
                L,
                eps,
                solver,
                out_dir,
                U,
                expected_shape,
                normalize,
            )
            for m in mats
        ]

        it = as_completed(futures)
        if pbar:
            it = tqdm(it, total=len(futures), desc="FB expand", unit="vol", dynamic_ncols=True)

        for fut in it:
            basename, out_csv, ncoef, err = fut.result()
            if err:
                errors += 1
                if pbar:
                    tqdm.write(f"[error] {basename}: {err.strip()}")
                else:
                    print(f"[error] {basename}: {err.strip()}")
            else:
                if pbar:
                    tqdm.write(f"[ok] {basename}: coeffs={ncoef} → {out_csv}")
            done += 1

    print("\n[summary]")
    print(f"  volumes selected: {len(mats)} (from {len(mats_all)} files matching pattern)")
    print(f"  processed: {done} | errors: {errors}")
    print(f"  coeff shape: n_rad={n_rad0}, L_eff={L_eff0}")
    print(f"  wrote: {os.path.join(out_dir, '*_coeff_energy.csv')}")


if __name__ == "__main__":
    args = parse_args()

    in_dir = args.in_dir if args.in_dir is not None else f"mat_converted_N={args.expect_n}"
    out_dir = args.out_dir if args.out_dir is not None else f"mat_converted_N={args.expect_n}_FBexpansions"

    main(
        in_dir=in_dir,
        out_dir=out_dir,
        L=args.L,
        eps=args.eps,
        solver=args.solver,
        pattern=args.pattern,
        targets=args.targets,
        n_jobs=args.n_jobs,
        normalize=args.normalize,
        pbar=args.pbar,
        expect_N=args.expect_n,
    )
