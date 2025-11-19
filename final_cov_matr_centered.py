#!/usr/bin/env python3
"""
final_cov_matr_centered_global_only.py

Single-variant pipeline (GLOBAL centering only):
  • Build SO(3)-block second moments S_ell from volumes.
  • Global centering for ℓ=0 only:    S_0 ← S_0 − μ μ*
  • Eigendecomposition per ℓ (no shrinkage).
  • Save:
        cov_blocks_N=..._L=..._centered_global.npz
        top_modes_N=..._L=..._K=..._centered_global_by_raw.npz
        full_eigs_N=..._L=..._centered_global.npz    (ON by default)
  • For each volume, export:
        <basename>_coeff_energy_centered_global.csv  (k, ratio)
        <basename>_coeffs_centered_global_<all|topK>.npz
          - per-ℓ eigenbasis coefficients A_ℓ (rows are radial indices)
          - optional row downselection (keep rows used by global top-K)

Notes:
  - Only ℓ=0 changes under global centering; ℓ≥1 are same as uncentered.
  - Exports apply the same (ℓ=0,m=0) centering used to build S_0.
"""

import os
import glob
import numpy as np
from scipy.io import loadmat
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple, List, Optional, Dict

from tqdm.auto import tqdm
from fle_3d import FLEBasis3D  # your library


# =================== TOP-LEVEL CONFIG ===================
# Resolution "N" also controls default input/output folder names below.
N = 22
DEFAULT_IN_DIR  = f"mat_converted_N={N}"
DEFAULT_OUT_DIR = f"mat_converted_N={N}_matrix"

IMAG_TOL = 1e-2      # drop imag if |Im| < 0.01 when printing eigenvalues/vectors
TOP_SHOW = 30        # how many global leading eigenpairs to print

# CSV/coeff export controls
EXPORT_CSV           = True           # write <basename>_coeff_energy_centered_global.csv
EXPORT_COEFFS        = True           # write <basename>_coeffs_centered_global_*.npz
COEFFS_KEEP_TOPK     = False           # if True, keep only rows that appear in global top-K
COEFFS_COMPLEX64     = True           # store per-volume A_ℓ as complex64 to save space
EXPORT_N_JOBS        = -1             # -1: all cores; if using GPU solver, consider 1
APPLY_CENTERING_IN_EXPORT = True      # subtract μ from (ℓ=0,m=0) before projection

# Main eigens output
SAVE_FULL_EIGENDECOMP = True          # <—— now ON by default
# ========================================================


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
      eigvals: (L, n_rad)          descending (per ℓ)
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

# ----------------------- per-volume export helpers -----------------------

def _build_keep_rows_by_ell(order_flat: np.ndarray, n_rad: int, keep_topK: int) -> Dict[int, np.ndarray]:
    """
    Map ℓ -> sorted unique list of radial indices to keep,
    selecting rows that appear in the global top-K flattening.
    """
    chosen = order_flat[:keep_topK]
    keep: Dict[int, List[int]] = {}
    for fi in chosen:
        ell = fi // n_rad
        rad = fi %  n_rad
        keep.setdefault(ell, []).append(rad)
    out: Dict[int, np.ndarray] = {}
    for ell, rows in keep.items():
        out[ell] = np.array(sorted(set(rows)), dtype=np.int32)
    return out

def export_coeff_energy_and_projections_centered_global(
    m_path: str,
    key: str,
    N_vol: int,
    L: int,
    eps: float,
    solver: str,
    n_rad: int,
    eigvecs: np.ndarray,        # (L, n_rad, n_rad)
    order_flat: np.ndarray,     # global ordering (descending) of flattened modes
    out_dir: str,
    mu_l0: Optional[np.ndarray] = None,
    apply_centering_in_export: bool = True,
    save_coeffs: bool = True,
    coeffs_keep_rows_by_ell: Optional[Dict[int, np.ndarray]] = None,
    coeffs_complex64: bool = True,
) -> bool:
    """
    For one volume:
      - Build FB coefficients B (n_rad, L, 2ℓ+1)
      - If apply_centering_in_export: center (ℓ=0,m=0): B[:,0,0] -= μ
      - Project: A_ℓ = U_ℓ^H B_ℓ
      - Save cumulative energy CSV by global order (k, ratio)
      - Optionally save per-ℓ A_ℓ in NPZ (with optional row downselection)
    """
    if apply_centering_in_export and mu_l0 is None:
        raise ValueError("mu_l0 must be provided when apply_centering_in_export is True.")

    fle = FLEBasis3D(
        N=N_vol, bandlimit=L, eps=eps, max_l=L,
        mode="complex", sph_harm_solver=solver, reduce_memory=True
    )

    v = load_and_normalize(m_path, key)
    z = fle.step1(v)
    b = fle.step2(z)  # (n_rad, L, 2ℓ+1)

    if apply_centering_in_export:
        b[:, 0, 0] = b[:, 0, 0] - mu_l0

    energies = np.zeros((L, n_rad), dtype=np.float64)
    coeff_pack = {
        "meta_N": N_vol,
        "meta_L": L,
        "order_flat": order_flat.astype(np.int32),
        "apply_centering_in_export": bool(apply_centering_in_export),
    }

    for ell in range(L):
        B_ell = b[:, ell, :2*ell + 1]        # (n_rad, 2ℓ+1)
        U_ell = eigvecs[ell]                 # (n_rad, n_rad)
        A = U_ell.conj().T @ B_ell           # (n_rad, 2ℓ+1)
        energies[ell] = np.sum(np.abs(A)**2, axis=1)

        if save_coeffs:
            if coeffs_keep_rows_by_ell is not None:
                rows = coeffs_keep_rows_by_ell.get(ell, np.array([], dtype=np.int32))
                A_to_save = A[rows]
                coeff_pack[f"A_l{ell}"] = A_to_save.astype(np.complex64 if coeffs_complex64 else np.complex128)
                coeff_pack[f"rows_l{ell}"] = rows.astype(np.int16)
            else:
                coeff_pack[f"A_l{ell}"] = A.astype(np.complex64 if coeffs_complex64 else np.complex128)

    # ---- cumulative energy CSV
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

    # ---- save coefficients NPZ
    if save_coeffs:
        coeff_pack["mu_l0"] = (mu_l0.astype(np.complex64) if (mu_l0 is not None and coeffs_complex64) else mu_l0)
        tag = "_topK" if (coeffs_keep_rows_by_ell is not None) else "_all"
        npz_path = os.path.join(out_dir, f"{basename}_coeffs_centered_global{tag}.npz")
        np.savez_compressed(npz_path, **coeff_pack)

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
    save_full_eigendecomp=SAVE_FULL_EIGENDECOMP,   # now True by default
    pbar=True,
    expect_N=N,
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

    # (ON by default) save full eigen-decomposition (centered only)
    if save_full_eigendecomp:
        np.savez_compressed(
            os.path.join(out_dir, f"full_eigs_N={N_vol}_L={L}_centered_global.npz"),
            eigvals=raw_cg, eigvecs=vec_cg, **cov_meta, mu_l0=mu_l0
        )

    # ---- per-volume CSV + coefficients export
    if EXPORT_CSV or EXPORT_COEFFS:
        export_workers = (os.cpu_count() or 1) if (EXPORT_N_JOBS in (-1, None, 0)) else int(EXPORT_N_JOBS)
        if "nvidia" in str(solver).lower():
            export_workers = 1 if EXPORT_N_JOBS in (-1, None, 0) else export_workers

        # build rows-to-keep mapping (if requested)
        keep_rows_by_ell = None
        if EXPORT_COEFFS and COEFFS_KEEP_TOPK:
            keep_rows_by_ell = _build_keep_rows_by_ell(ord_cg_raw, n_rad, keep_topK=K)

        what = []
        if EXPORT_CSV:    what.append("CSV")
        if EXPORT_COEFFS: what.append("coeffs")
        print(f"[export] Writing {' + '.join(what)} for {len(mats)} volumes …")

        with ThreadPoolExecutor(max_workers=export_workers) as ex:
            futures = [
                ex.submit(
                    export_coeff_energy_and_projections_centered_global,
                    m, key, N_vol, L, eps, solver, n_rad,
                    eigvecs=vec_cg,
                    order_flat=ord_cg_raw,
                    out_dir=out_dir,
                    mu_l0=mu_l0,
                    apply_centering_in_export=APPLY_CENTERING_IN_EXPORT,
                    save_coeffs=EXPORT_COEFFS,
                    coeffs_keep_rows_by_ell=keep_rows_by_ell,
                    coeffs_complex64=True if COEFFS_COMPLEX64 else False
                )
                for m in mats
            ]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Export per-volume", leave=True):
                fut.result()  # ensure errors surface

    # ---------- summary ----------
    print("\n[summary]")
    print(f"  volumes processed: {M}")
    print(f"  total modes: {L*n_rad}  |  K (global top modes): {K}")
    print("\n[save] wrote:")
    for fn in (
        f"cov_blocks_N={N_vol}_L={L}_centered_global.npz",
        f"top_modes_N={N_vol}_L={L}_K={K}_centered_global_by_raw.npz",
    ):
        print("  •", os.path.join(out_dir, fn))
    if save_full_eigendecomp:
        print("  •", os.path.join(out_dir, f"full_eigs_N={N_vol}_L={L}_centered_global.npz"))
    if EXPORT_COEFFS:
        print("  • Per-volume: *_coeffs_centered_global_{'topK' if COEFFS_KEEP_TOPK else 'all'}.npz")
    if EXPORT_CSV:
        print("  • Per-volume: *_coeff_energy_centered_global.csv")


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
        save_full_eigendecomp=SAVE_FULL_EIGENDECOMP,
        pbar=True,               # turn off if you don't want bars in nohup logs
        expect_N=N,              # sanity-check resolution vs data
    )
