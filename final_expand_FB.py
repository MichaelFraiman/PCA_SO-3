#!/usr/bin/env python3
"""
final_expand_FB.py
Expand selected volumes in the Fourier–Bessel basis and write:
    k, w_ratio_lex, w_ratio_usort
where:
  - w_ratio_lex   : cumulative energy in fixed (n, ℓ, m) lexicographic order
  - w_ratio_usort : cumulative energy when (ℓ, n) blocks are ordered by
                    increasing u_{ℓ,s} (true u-sorting; ties over m)

Features:
  • Always NVIDIA solver + max workers
  • N set at top; I/O folders derived from N
  • TARGETS to restrict which volumes to process
  • Output dir: mat_converted_N={N}_FBexpansions
  • Robust u-table: use FLEBasis3D if provided; otherwise compute zeros of
    spherical Bessel j_ℓ via brentq (SciPy), using safe brackets.

Usage:
  python3 final_expand_FB.py
"""

import os
import fnmatch
import glob
import traceback
import numpy as np
from typing import Optional, Tuple, List

from scipy.io import loadmat
from scipy.special import spherical_jn
from scipy.optimize import brentq
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm.auto import tqdm

# ----------------------- TOP-LEVEL CONFIG -----------------------
N         = 22
L         = 20
EPS       = 1e-6

IN_DIR    = f"mat_converted_N={N}"
OUT_DIR   = f"mat_converted_N={N}_FBexpansions"

SOLVER        = "nvidia_torch"               # force NVIDIA solver
NUM_WORKERS   = max(1, (os.cpu_count() or 1))

# Expand only these volumes. Basenames (w/o .mat) or glob patterns are accepted.
TARGETS = ["1f*"]
#TARGETS: List[str] = []  # empty => all .mat in IN_DIR
# ---------------------------------------------------------------

# Per-process globals
_fle: Optional["FLEBasis3D"] = None
_out_dir: Optional[str] = None
_key_cache: dict = {}              # per-process cache for MAT key
_warned_fallback: bool = False     # print fallback notice once per worker


# ----------------------- IO HELPERS -----------------------
def get_mat_list(directory, pattern="*.mat") -> List[str]:
    return sorted(glob.glob(os.path.join(directory, pattern)))

def _detect_first_data_key(mat_dict: dict) -> str:
    for k in mat_dict:
        if not k.startswith('__'):
            return k
    raise KeyError("No data key found in .mat file (only __* keys present).")

def _filter_targets(mats: List[str], targets: List[str]) -> List[str]:
    if not targets:
        return mats
    exact_names = set()
    patterns = []
    for t in targets:
        t = t.strip()
        if not t:
            continue
        if any(ch in t for ch in "*?[]"):
            patterns.append(t)
        else:
            exact_names.add(os.path.splitext(os.path.basename(t))[0].lower())

    picked = []
    for p in mats:
        base = os.path.splitext(os.path.basename(p))[0]
        base_l = base.lower()
        if base_l in exact_names:
            picked.append(p)
            continue
        if any(fnmatch.fnmatch(base, pat) or fnmatch.fnmatch(base + ".mat", pat) for pat in patterns):
            picked.append(p)
    return sorted(picked)


# ----------------------- FB INITIALIZER -----------------------
def _init_worker(N_: int, L_: int, eps_: float, solver_: str, out_dir: str):
    """Per-process initializer; creates a local FLEBasis3D and stores out_dir."""
    global _fle, _out_dir
    _out_dir = out_dir
    from fle_3d import FLEBasis3D  # local import inside worker
    _fle = FLEBasis3D(
        N=N_, bandlimit=L_, eps=eps_, max_l=L_,
        mode="complex",
        sph_harm_solver=solver_,      # "nvidia_torch"
        reduce_memory=True,
    )


# ----------------------- u_{ℓ,s} TABLE -----------------------
def _try_fetch_u_table_from_fle(n_rad: int, L_eff: int) -> Optional[np.ndarray]:
    """Attempt to read a 2D (n_rad, L_eff) table from _fle; return None if unavailable."""
    global _fle
    if _fle is None:
        return None

    # 1) direct 2D arrays on the object
    attr_candidates = [
        "u_table", "u", "u_ls", "ul_s", "kappa", "k_ls",
        "roots", "radial_roots", "rho_ls",
    ]
    for name in attr_candidates:
        if hasattr(_fle, name):
            obj = getattr(_fle, name)
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
        if hasattr(_fle, name):
            fn = getattr(_fle, name)
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
                    return np.stack(cols, axis=1)  # (n_rad, L_eff) after stack across ells
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
        for s_idx in range(n_rad):
            s = s_idx + 1  # human indexing
            # initial bracket centered at ~ (s + ell/2)*π
            center = (s + 0.5 * ell) * pi
            a = center - 0.5 * pi
            b = center + 0.5 * pi

            # Ensure positive bracket (avoid a<=0 near tiny ell/s)
            a = max(a, 1e-6)
            # If endpoint is exactly a root, nudge by tiny eps
            def f(x): return spherical_jn(ell, x)

            fa = f(a)
            fb = f(b)
            if np.sign(fa) == np.sign(fb):
                # widen bracket gradually until sign change (or give up after few tries)
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
                    # as last resort, sample on a grid and locate a sign flip
                    xs = np.linspace(a, b + 2 * pi, 200)
                    ys = f(xs)
                    idx = np.where(np.sign(ys[:-1]) * np.sign(ys[1:]) < 0)[0]
                    if idx.size:
                        a, b = xs[idx[0]], xs[idx[0] + 1]
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


def _fetch_u_table(n_rad: int, L_eff: int) -> np.ndarray:
    """Get u_{ℓ,s} as (n_rad, L_eff). Try FLE first; otherwise compute with brentq."""
    global _warned_fallback
    U = _try_fetch_u_table_from_fle(n_rad, L_eff)
    if U is not None:
        return U
    if not _warned_fallback:
        tqdm.write("[u-table] FLEBasis3D did not expose u_{ℓ,s}; computing zeros via brentq fallback.")
        _warned_fallback = True
    return _compute_u_table_brent(n_rad, L_eff)


# ----------------------- MAIN WORKER -----------------------
def _process_one(m_path: str) -> Tuple[str, Optional[str], Optional[int], Optional[str]]:
    """Returns: (basename, out_csv, ncoef, error_msg)"""
    global _fle, _out_dir, _key_cache
    basename = os.path.splitext(os.path.basename(m_path))[0]
    try:
        if _fle is None or _out_dir is None:
            raise RuntimeError("Worker not initialized: FLE or out_dir is None.")

        data = loadmat(m_path)
        key = _key_cache.get(m_path)
        if key is None:
            key = _detect_first_data_key(data)
            _key_cache[m_path] = key

        vol = np.ascontiguousarray(data[key], dtype=np.float64)
        vmax = float(np.max(np.abs(vol)))
        if vmax == 0.0:
            return basename, None, None, f"Skip {basename}: volume is all zeros"
        vol /= vmax

        # FLE expansion: b[n, ell, :2*ell+1]
        z = _fle.step1(vol)
        b = _fle.step2(z)

        n_rad = b.shape[0]
        L_eff = b.shape[1]

        # ---- energies in lexicographic (n, ℓ, m) order ----
        energy_lex_chunks = []
        for n_idx in range(n_rad):
            for ell in range(L_eff):
                m_len = 2 * ell + 1
                coeffs = b[n_idx, ell, :m_len]
                energy_lex_chunks.append(np.abs(coeffs) ** 2)

        energies_lex = (
            np.concatenate(energy_lex_chunks).astype(np.float64, copy=False)
            if energy_lex_chunks else np.empty((0,), dtype=np.float64)
        )
        total = float(energies_lex.sum())
        if total <= 0.0:
            ratios_lex = np.zeros_like(energies_lex)
            ratios_us  = np.zeros_like(energies_lex)
        else:
            ratios_lex = np.cumsum(energies_lex) / total

            # ---- true u-sorting: order (ℓ, n) by increasing u_{ℓ,s}; ties: iterate m ----
            U = _fetch_u_table(n_rad=n_rad, L_eff=L_eff)  # shape (n_rad, L_eff)
            order_pairs = sorted(
                ((U[n_idx, ell], n_idx, ell) for ell in range(L_eff) for n_idx in range(n_rad)),
                key=lambda t: t[0]
            )
            energy_us_chunks = []
            for _, n_idx, ell in order_pairs:
                m_len = 2 * ell + 1
                coeffs = b[n_idx, ell, :m_len]
                energy_us_chunks.append(np.abs(coeffs) ** 2)
            energies_us = np.concatenate(energy_us_chunks).astype(np.float64, copy=False)
            ratios_us = np.cumsum(energies_us) / total

        out_csv = os.path.join(_out_dir, f"{basename}_coeff_energy.csv")
        with open(out_csv, "w") as f:
            f.write("k,w_ratio_lex,w_ratio_usort\n")
            for k in range(1, ratios_lex.size + 1):
                f.write(f"{k},{ratios_lex[k-1]:.12g},{ratios_us[k-1]:.12g}\n")

        ncoef = int(ratios_lex.size)
        return basename, out_csv, ncoef, None

    except Exception as e:
        tb = traceback.format_exc(limit=6)
        return basename, None, None, f"{e.__class__.__name__}: {e}\n{tb}"


# ----------------------- DRIVER -----------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    mats = get_mat_list(IN_DIR)
    mats = _filter_targets(mats, TARGETS)

    if not mats:
        print(f"[init] No .mat files to process. IN_DIR='{IN_DIR}', TARGETS={TARGETS}", flush=True)
        return

    print(f"[init] Found {len(mats)} .mat files to process.", flush=True)
    print(f"[init] Grid N={N}, L={L}, eps={EPS}, solver='{SOLVER}', workers={NUM_WORKERS}", flush=True)
    print(f"[init] Output dir: {OUT_DIR}", flush=True)

    done = errors = 0
    with ProcessPoolExecutor(
        max_workers=NUM_WORKERS,
        initializer=_init_worker,
        initargs=(N, L, EPS, SOLVER, OUT_DIR),
    ) as ex:
        futures = [ex.submit(_process_one, m) for m in mats]
        with tqdm(total=len(futures), desc="expanding volumes (FB)", unit="vol",
                  dynamic_ncols=True, mininterval=0.5, smoothing=0.1) as pbar:
            for fut in as_completed(futures):
                basename, out_csv, ncoef, err = fut.result()
                if err:
                    tqdm.write(f"[error] {basename}: {err.strip()}")
                    errors += 1
                else:
                    tqdm.write(f"[ok] {basename}: coeffs={ncoef} → {out_csv}")
                done += 1
                pbar.update(1)

    print(f"[done] processed={done}, errors={errors}, out_dir='{OUT_DIR}'", flush=True)


if __name__ == '__main__':
    # Keep BLAS from oversubscribing per-process
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    main()
