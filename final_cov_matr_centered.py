#!/usr/bin/env python3
"""
final_cov_matr_centered_global_only.py

What this script does (single variant only):
  • Builds SO(3)-block second moments S_ell from volumes.
  • Computes the global mean vector μ for (ℓ=0,m=0) and applies GLOBAL CENTERING:
        S_0  ←  S_0  − μ μ*
        S_ℓ  ←  S_ℓ  for ℓ ≥ 1
  • Eigendecomposition per ℓ (no shrinkage).
  • Saves:
        cov_blocks_N=..._L=..._centered_global.npz
        top_modes_N=..._L=..._K=..._centered_global_by_raw.npz
  • Optionally exports per-volume cumulative energy CSVs (centered_global only):
        <basename>_coeff_energy_centered_global.csv

Notes:
  - Only the ℓ=0 block is changed by centering; ℓ≥1 blocks are identical to uncentered.
  - No per-volume centering, no comparisons, no shrinkage.
"""

import os
import glob
import numpy as np
from scipy.io import loadmat
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple, List, Optional

from tqdm.auto import tqdm
from fle_3d import FLEBasis3D  # your library


# =================== TOP-LEVEL CONFIG ===================
# Resolution "N" also controls default input/output folder names below.
N = 124
DEFAULT_IN_DIR  = f"mat_converted_N={N}"
DEFAULT_OUT_DIR = f"mat_converted_N={N}_matrix"
IMAG_TOL = 1e-2      # drop imag if |Im| < 0.01 when printing eigenvalues/vectors
TOP_SHOW = 30        # how many global leading eigenpairs to print

# CSV export controls
EXPORT_CSV = True           # set False to skip per-volume CSV export
EXPORT_N_JOBS = -1          # -1: all cores; if using GPU solver, consider 1
# ========================================================

APPLY_CENTERING_IN_EXPORT = True  # subtract μ from (ℓ=0,m=0) before projection


# ----------------------- IO / helpers -----------------------

def get_mat_list(d, pattern="*.mat"):
    return sorted(glob.glob(os.path.join(d, pattern)))

def load_and_normalize(path, key):
    data = loadmat(path)
    v = np.ascontiguousarray(data[key])
    vmax = np.max(np.abs(v))
    return v if vmax == 0 else (v / vmax)

# ----------------------- FB expansion & covariance -----------------------

def compute_cov_per_l(fle: FLEBasis3D, vol: np.ndarray, L: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return:
      C:   (L, n_rad, n_rad) complex128, where C[ℓ] = sum_m B_ℓm B_ℓm^H (no 1/(2ℓ+1) yet)
      b0:  (n_rad,) ℓ=0, m=0 vector for this volume (for global centering)
    """
    z = fle.step1(vol)
    b = fle.step2(z)                          # (n_rad, L, 2ℓ+1)
    n_rad = b.shape[0]
    C = np.zeros((L, n_rad, n_rad), np.complex128)
    for ell in range(L):
        Bl = b[:, ell, :2*ell+1]              # (n_rad, 2ℓ+1)
        C[ell] = Bl @ Bl.conj().T             # sum over m
    b_l0 = b[:, 0, 0].astype(np.complex128)   # (n_rad,)
    return C, b_l0

def compute_cov_for_file(m, key, N_vol, L, eps, solver):
    fle = FLEBasis3D(
        N=N_vol, bandlimit=L, eps=eps, max_l=L,
        mode="complex", sph_harm_solver=solver, reduce_memory=True
    )
    vol = load_and_normalize(m, key)
    return compute_cov_per_l(fle, vol, L)     # (C, b_l0)

# ----------------------- linear algebra -----------------------

def svd_covariance(blocks: np.ndarray, pbar: bool = True):
    """
    blocks: (L, n_rad, n_rad) Hermitian-ish
    returns:
      eigvals: (L, n_rad)          descending
      eigvecs: (L, n_rad, n_rad)   columns are eigenvectors (U from SVD)
    """
    L, n_rad, _ = blocks.shape
    eigvals = np.zeros((L, n_rad), np.float64)
    eigvecs = np.zeros((L, n_rad, n_rad), np.complex128)
    rng = range(L)
    if pbar:
        rng = tqdm(rng, desc="Eigendecomposition per ℓ", leave=False)
    for ell in rng:
        Bl = 0.5 * (blocks[ell] + blocks[ell].conj().T)  # guard symmetry
        U, S, _ = np.linalg.svd(Bl, full_matrices=False)
        eigvecs[ell] = U
        eigvals[ell] = S
    return eigvals, eigvecs

# ----------------------- printing helpers -----------------------

def fmt_number(z, imag_tol: float = IMAG_TOL, prec: int = 6) -> str:
    """
    For eigenvalues/vectors:
      - If complex and |Im| < imag_tol, drop the imag part and print real with 'prec' digits.
      - Else print as 'a+bi' with 'prec' digits.
    """
    if isinstance(z, complex) or (hasattr(z, "imag")):
        z = complex(z)
        if abs(z.imag) < imag_tol:
            return f"{z.real:.{prec}f}"
        sgn = '+' if z.imag >= 0 else '-'
        return f"{z.real:.{prec}f}{sgn}{abs(z.imag):.{prec}f}j"
    return f"{float(z):.{prec}f}"

def vec_to_string(v: np.ndarray, imag_tol: float = IMAG_TOL, prec: int = 6) -> str:
    return "[" + ", ".join(fmt_number(val, imag_tol, prec) for val in v) + "]"

def print_topk_global(raw: np.ndarray, vecs: np.ndarray, n_rad: int, top_k_show: int):
    """
    Print TOP-K eigenpairs globally (by raw eigenvalues of the centered model).
    """
    flat = raw.flatten()
    order = np.argsort(flat)[::-1]
    total = raw.size
    K = min(top_k_show, total)
    print(f"\n[top-{K}] Global top eigenpairs (centered_global; |Im|<0.01 → real)")
    for rank, fi in enumerate(order[:K], start=1):
        ell = fi // n_rad
        rad = fi %  n_rad
        lam = raw[ell, rad]
        u = vecs[ell][:, rad]
        print(f"  #{rank:02d}  (ℓ={ell}, r={rad})  λ={fmt_number(lam)}")
        print("       u*: ", vec_to_string(u))

# ----------------------- per-volume CSV export (centered_global only) -----------------------

def export_coeff_energy_centered_global(
    m_path: str,
    key: str,
    N_vol: int,
    L: int,
    eps: float,
    solver: str,
    n_rad: int,
    eigvecs: np.ndarray,
    order_flat: np.ndarray,
    out_dir: str,
    mu_l0: Optional[np.ndarray] = None,
    apply_centering_in_export: bool = True,
) -> bool:
    """
    For one volume:
      - build FLE coefficients b = step2(step1(vol))  with shape (n_rad, L, 2ℓ+1)
      - if apply_centering_in_export: center only (ℓ=0,m=0): b[:,0,0] -= μ
      - project to eigenbasis of the centered model: A_ell = U_ell^H B_ell
      - accumulate per-(ℓ,rad) energies: sum_m |A_ell|^2
      - flatten by (ℓ,rad), reorder by order_flat (global centered order), cumulative ratio
      - save CSV: <basename>_coeff_energy_centered_global.csv with columns: k,ratio
    """
    if apply_centering_in_export and mu_l0 is None:
        raise ValueError("mu_l0 must be provided when apply_centering_in_export is True.")

    fle = FLEBasis3D(
        N=N_vol, bandlimit=L, eps=eps, max_l=L,
        mode="complex", sph_harm_solver=solver, reduce_memory=True
    )

    v = load_and_normalize(m_path, key)
    z = fle.step1(v)
    b = fle.step2(z)  # shape: (n_rad, L, 2ℓ+1)

    if apply_centering_in_export:
        b[:, 0, 0] = b[:, 0, 0] - mu_l0

    energies = np.zeros((L, n_rad), dtype=np.float64)
    for ell in range(L):
        B_ell = b[:, ell, :2*ell + 1]        # (n_rad, 2ℓ+1)
        U_ell = eigvecs[ell]                 # (n_rad, n_rad), columns = eigenvectors
        A = U_ell.conj().T @ B_ell           # (n_rad, 2ℓ+1)
        energies[ell] = np.sum(np.abs(A)**2, axis=1)

    ef = energies.flatten()
    ef = ef[order_flat]
    total = float(ef.sum())
    ratios = (np.cumsum(ef) / total) if total > 0 else np.zeros_like(ef)

    k = np.arange(1, ratios.size + 1, dtype=int)
    table = np.column_stack((k, ratios))
    basename = os.path.splitext(os.path.basename(m_path))[0]
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"{basename}_coeff_energy_centered_global.csv")
    np.savetxt(csv_path, table, delimiter=",", header="k,ratio", comments="")
    return True


# ----------------------- main pipeline -----------------------

def main(
    in_dir=DEFAULT_IN_DIR,
    out_dir=DEFAULT_OUT_DIR,
    L=20,
    eps=1e-6,
    solver="nvidia_torch",
    top_k=1000,
    n_jobs=-1,
    store_as_complex64=True,
    save_full_eigendecomp=False,   # WARNING: big files if True
    pbar=True,                     # show tqdm progress bars
    expect_N=N,                    # sanity check
):
    os.makedirs(out_dir, exist_ok=True)
    mats = get_mat_list(in_dir)
    if not mats:
        raise RuntimeError(f"No .mat files in '{in_dir}'")

    # discover key & grid
    samp = loadmat(mats[0])
    key = next(k for k in samp.keys() if not k.startswith("__"))
    N_vol = int(np.ascontiguousarray(samp[key]).shape[0])
    if expect_N is not None and expect_N != N_vol:
        print(f"[warn] Folder N={expect_N} but first volume is {N_vol}. "
              f"Proceeding with N_vol={N_vol} for computations.")
    print(f"[build] Volumes={len(mats)} | N_vol={N_vol}, L={L}, eps={eps}")

    # ---- accumulate covariance pieces & ℓ=0 vectors (parallel; progress bar)
    def worker(path):
        return compute_cov_for_file(path, key, N_vol, L, eps, solver)

    workers = (os.cpu_count() or 1) if (n_jobs in (-1, None, 0)) else int(n_jobs)
    sum_over_v = None
    sum_l0 = None

    if pbar:
        print("[step] Expanding volumes & accumulating covariance …")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(worker, m) for m in mats]
        iterator = as_completed(futures)
        if pbar:
            iterator = tqdm(iterator, total=len(futures), desc="Volumes", leave=True)
        for fut in iterator:
            C, b_l0 = fut.result()
            sum_over_v = C if sum_over_v is None else (sum_over_v + C)
            sum_l0     = b_l0 if sum_l0 is None else (sum_l0 + b_l0)

    M = len(mats)

    # average over volumes
    avg_over_v = sum_over_v / M                        # (L, n_rad, n_rad)
    L_blocks, n_rad, _ = avg_over_v.shape
    assert L_blocks == L
    p = n_rad  # number of radial functions per ℓ

    # per-ℓ normalization by (2ℓ+1)
    S_raw = np.empty_like(avg_over_v)
    for ell in range(L):
        Bl = avg_over_v[ell] / float(2*ell + 1)
        S_raw[ell] = 0.5 * (Bl + Bl.conj().T)

    # global mean (DC) vector across volumes for ℓ=0
    mu_l0 = (sum_l0 / M).astype(np.complex128)        # (n_rad,)

    # ---- GLOBAL CENTERING (only ℓ=0)
    S_centered = S_raw.copy()
    S_centered[0] = S_centered[0] - np.outer(mu_l0, mu_l0.conj())
    S_centered[0] = 0.5 * (S_centered[0] + S_centered[0].conj().T)

    # ---- save centered covariance blocks
    cov_meta = dict(N=N_vol, L=L, eps=eps, solver=solver, n_rad=n_rad, M=M)
    np.savez_compressed(
        os.path.join(out_dir, f"cov_blocks_N={N_vol}_L={L}_centered_global.npz"),
        S_blocks=S_centered, mu_l0=mu_l0, **cov_meta
    )
    print("[save] wrote covariance NPZ (centered_global)")

    # ---- eigendecomposition (centered only)
    if pbar: print("[step] Eigendecomposition …")
    raw_cg, vec_cg = svd_covariance(S_centered, pbar=pbar)

    # ---- PRINT: TOP-30 global eigenpairs (centered model only)
    print_topk_global(raw_cg, vec_cg, n_rad, top_k_show=TOP_SHOW)

    # ---- top-K selection (by raw centered eigenvalues)
    total_modes = L * n_rad
    K = min(top_k, total_modes)
    if K < top_k:
        print(f"[warn] Requested top_k={top_k} but only {total_modes} modes exist; using K={K}")

    ord_cg_raw = np.argsort(raw_cg.flatten())[::-1]

    def gather_top(K, order, n_rad, eigvecs, eigvals):
        top_ell = np.empty(K, dtype=np.int32)
        top_rad = np.empty(K, dtype=np.int32)
        top_vals_raw = np.empty(K, dtype=np.float64)
        top_vecs = np.empty((K, n_rad), dtype=np.complex128)
        for i in range(K):
            fi = order[i]
            ell = fi // n_rad
            rad = fi %  n_rad
            top_ell[i] = ell
            top_rad[i] = rad
            top_vals_raw[i] = eigvals[ell, rad]
            top_vecs[i] = eigvecs[ell][:, rad]
        return top_vecs, top_ell, top_rad, top_vals_raw

    top_vecs_cg, top_ell_cg, top_rad_cg, top_raw_cg = gather_top(
        K, ord_cg_raw, n_rad, vec_cg, raw_cg
    )

    if store_as_complex64:
        top_vecs_cg = top_vecs_cg.astype(np.complex64)

    # ---- save top-K pack (centered_global only)
    meta_common = dict(
        N=N_vol, L=L, eps=eps, solver=solver,
        n_rad=n_rad, K=K, mu_l0=mu_l0,
        mats_used=np.array([os.path.basename(x) for x in mats])
    )
    np.savez_compressed(
        os.path.join(out_dir, f"top_modes_N={N_vol}_L={L}_K={K}_centered_global_by_raw.npz"),
        top_vecs=top_vecs_cg,
        top_ell=top_ell_cg, top_rad=top_rad_cg,
        top_vals_raw=top_raw_cg,
        **meta_common
    )

    # ---- per-volume CSV export (centered_global ONLY), with progress bar
    if EXPORT_CSV:
        export_workers = (os.cpu_count() or 1) if (EXPORT_N_JOBS in (-1, None, 0)) else int(EXPORT_N_JOBS)
        if "nvidia" in str(solver).lower():
            export_workers = 1 if EXPORT_N_JOBS in (-1, None, 0) else export_workers

        print(f"[export] Writing centered_global cumulative energy CSVs for {len(mats)} volumes …")
        with ThreadPoolExecutor(max_workers=export_workers) as ex:
            futures = [
                ex.submit(
                    export_coeff_energy_centered_global,
                    m, key, N_vol, L, eps, solver, n_rad,
                    eigvecs=vec_cg,
                    order_flat=ord_cg_raw,
                    out_dir=out_dir,
                    mu_l0=mu_l0,
                    apply_centering_in_export=APPLY_CENTERING_IN_EXPORT
                )
                for m in mats
            ]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Export CSVs", leave=True):
                fut.result()  # ensure errors surface

    # (optional) save full eigen-decomposition (centered only) — big!
    if save_full_eigendecomp:
        np.savez_compressed(
            os.path.join(out_dir, f"full_eigs_N={N_vol}_L={L}_centered_global.npz"),
            eigvals=raw_cg, eigvecs=vec_cg, **cov_meta, mu_l0=mu_l0
        )

    # ---------- summary ----------
    print("\n[summary]")
    print(f"  volumes processed: {M}")
    print(f"  total modes: {L*n_rad}  |  K saved: {K}")
    print("\n[save] wrote:")
    for fn in (
        f"cov_blocks_N={N_vol}_L={L}_centered_global.npz",
        f"top_modes_N={N_vol}_L={L}_K={K}_centered_global_by_raw.npz",
    ):
        print("  •", os.path.join(out_dir, fn))
    if save_full_eigendecomp:
        print("  •", os.path.join(out_dir, f"full_eigs_N={N_vol}_L={L}_centered_global.npz"))

if __name__ == "__main__":
    # ----------------------- CONFIG -----------------------
    main(
        in_dir=DEFAULT_IN_DIR,
        out_dir=DEFAULT_OUT_DIR,
        L=20,
        eps=1e-6,
        solver="nvidia_torch",
        top_k=1000,
        n_jobs=-1,               # -1 => use all cores for covariance stage
        store_as_complex64=True,
        save_full_eigendecomp=True,
        pbar=True,               # turn off if you don't want bars in nohup logs
        expect_N=N,              # sanity-check resolution vs data
    )
