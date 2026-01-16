#!/usr/bin/env python3
"""
0_1.covariance_matrix.py

Single-variant pipeline (GLOBAL centering only):
  • Build SO(3)-block second moments S_ell from volumes.
  • Global centering for ℓ=0 only:    S_0 ← S_0 − μ μ*
  • Eigendecomposition per ℓ (no shrinkage).
  • Save:
        cov_blocks_N=..._L=..._eps=..._centered_global.npz
        top_modes_N=..._L=..._eps=..._K=..._centered_global_by_raw.npz
        full_eigs_N=..._L=..._eps=..._centered_global.npz    (ON by default)
  • For each volume, export (optional):
        <basename>_coeff_energy_centered_global.csv  (k, ratio)
        <basename>_coeffs_L=<L>_eps=<eps_tag>_centered_global_<all|topK>.npz  (suffix is "_all" or "_topK")

All parameters are optional CLI args; defaults match previous hardcoded values.
"""

import os
import sys

# CRITICAL FIX FOR MACOS SEGFAULT:
# We must strictly control OpenMP/MKL threading before importing numpy/torch.
# This prevents libraries from spawning their own threads inside the Python 
# worker threads, which causes stack corruption and segfaults.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import glob
import argparse
import numpy as np
from scipy.io import loadmat
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple, List, Optional, Dict

from tqdm.auto import tqdm
from fle_3d import FLEBasis3D  # your library


# =================== DEFAULTS (match your previous hardcoded values) ===================
DEFAULT_EXPECT_N = 22
DEFAULT_L = 20
DEFAULT_EPS = 1e-6
DEFAULT_SOLVER = "nvidia_torch"
DEFAULT_TOP_K = 1000
DEFAULT_N_JOBS = -1

DEFAULT_IMAG_TOL = 1e-2
DEFAULT_TOP_SHOW = 30

# Export defaults
DEFAULT_EXPORT_CSV = False
DEFAULT_EXPORT_COEFFS = False
DEFAULT_COEFFS_KEEP_TOPK = False
DEFAULT_COEFFS_COMPLEX64 = True
DEFAULT_EXPORT_N_JOBS = -1
DEFAULT_APPLY_CENTERING_IN_EXPORT = True

# Main eigens output
DEFAULT_SAVE_FULL_EIGENDECOMP = True

# Storage for top_vecs
DEFAULT_STORE_AS_COMPLEX64 = True
# ======================================================================================


def parse_args():
    p = argparse.ArgumentParser(
        prog="0_1.covariance_matrix.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Compute globally-centered covariance blocks (ℓ=0) and eigendecomposition, with optional per-volume exports.",
    )

    # High-level / paths
    p.add_argument("--expect-n", type=int, default=DEFAULT_EXPECT_N,
                   help="Used only to form default folder names and warn if mismatch vs data.")
    p.add_argument("--in-dir", default=None,
                   help="Input directory with .mat files. Default: mat_converted_N=<expect-n>")
    p.add_argument("--out-dir", default=None,
                   help="Output directory. Default: mat_converted_N=<expect-n>_matrix")

    # Core parameters
    p.add_argument("--L", type=int, default=DEFAULT_L, help="Bandlimit (number of ℓ blocks).")
    p.add_argument("--eps", type=float, default=DEFAULT_EPS, help="FLEBasis3D eps.")
    p.add_argument("--solver", default=DEFAULT_SOLVER, help="Spherical harmonics solver for FLEBasis3D.")
    p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Global top-K modes to save.")
    p.add_argument("--n-jobs", type=int, default=DEFAULT_N_JOBS,
                   help="Workers for covariance accumulation stage (-1 means all cores).")

    # Printing
    p.add_argument("--imag-tol", type=float, default=DEFAULT_IMAG_TOL,
                   help="When printing complex numbers: drop imag if |Im| < imag-tol.")
    p.add_argument("--top-show", type=int, default=DEFAULT_TOP_SHOW,
                   help="How many global leading eigenpairs to print.")

    # Save full eigen-decomp
    p.add_argument("--save-full-eigendecomp", dest="save_full_eigendecomp",
                   action="store_true", default=DEFAULT_SAVE_FULL_EIGENDECOMP,
                   help="Save full eigen-decomposition NPZ.")
    p.add_argument("--no-save-full-eigendecomp", dest="save_full_eigendecomp",
                   action="store_false", help="Do not save full eigen-decomposition NPZ.")

    # Progress bars
    p.add_argument("--pbar", dest="pbar", action="store_true", default=True, help="Enable progress bars.")
    p.add_argument("--no-pbar", dest="pbar", action="store_false", help="Disable progress bars.")

    # Storage types
    p.add_argument("--store-as-complex64", dest="store_as_complex64",
                   action="store_true", default=DEFAULT_STORE_AS_COMPLEX64,
                   help="Store global top_vecs as complex64.")
    p.add_argument("--store-as-complex128", dest="store_as_complex64",
                   action="store_false", help="Store global top_vecs as complex128.")

    # Export toggles
    p.add_argument("--export-csv", dest="export_csv", action="store_true", default=DEFAULT_EXPORT_CSV,
                   help="Write per-volume *_coeff_energy_centered_global.csv.")
    p.add_argument("--no-export-csv", dest="export_csv", action="store_false",
                   help="Do not write per-volume CSVs.")

    p.add_argument("--export-coeffs", dest="export_coeffs", action="store_true", default=DEFAULT_EXPORT_COEFFS,
                   help="Write per-volume *_coeffs_L=<L>_eps=<eps_tag>_centered_global_{all|topK}.npz.")
    p.add_argument("--no-export-coeffs", dest="export_coeffs", action="store_false",
                   help="Do not write per-volume coefficient NPZs.")

    p.add_argument("--coeffs-keep-topk", dest="coeffs_keep_topk", action="store_true",
                   default=DEFAULT_COEFFS_KEEP_TOPK,
                   help="If exporting coeffs: keep only rows that appear in global top-K.")
    p.add_argument("--coeffs-keep-all", dest="coeffs_keep_topk", action="store_false",
                   help="If exporting coeffs: keep all rows (default behavior).")

    p.add_argument("--coeffs-complex64", dest="coeffs_complex64", action="store_true",
                   default=DEFAULT_COEFFS_COMPLEX64,
                   help="If exporting coeffs: store as complex64.")
    p.add_argument("--coeffs-complex128", dest="coeffs_complex64", action="store_false",
                   help="If exporting coeffs: store as complex128.")

    p.add_argument("--export-n-jobs", type=int, default=DEFAULT_EXPORT_N_JOBS,
                   help="Workers for per-volume export stage (-1 means all cores; if using GPU solver, 1 is safer).")

    p.add_argument("--apply-centering-in-export", dest="apply_centering_in_export",
                   action="store_true", default=DEFAULT_APPLY_CENTERING_IN_EXPORT,
                   help="Subtract μ from (ℓ=0,m=0) before projection in per-volume export.")
    p.add_argument("--no-apply-centering-in-export", dest="apply_centering_in_export",
                   action="store_false", help="Do not center before per-volume projection.")

    return p.parse_args()


# ----------------------- IO / helpers -----------------------

def get_mat_list(d, pattern="*.mat"):
    return sorted(glob.glob(os.path.join(d, pattern)))

def load_and_normalize(path, key):
    data = loadmat(path)
    v = np.ascontiguousarray(data[key])
    vmax = np.max(np.abs(v))
    return v if vmax == 0 else (v / vmax)


def eps_to_tag(eps: float) -> str:
    """Compact scientific-notation tag for filenames (e.g., 1e-06, 1e06)."""
    return f"{eps:.0e}".replace("+", "")

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

def fmt_number(z, imag_tol: float, prec: int = 6) -> str:
    if isinstance(z, complex) or (hasattr(z, "imag")):
        z = complex(z)
        if abs(z.imag) < imag_tol:
            return f"{z.real:.{prec}f}"
        sgn = '+' if z.imag >= 0 else '-'
        return f"{z.real:.{prec}f}{sgn}{abs(z.imag):.{prec}f}j"
    return f"{float(z):.{prec}f}"

def vec_to_string(v: np.ndarray, imag_tol: float, prec: int = 6) -> str:
    return "[" + ", ".join(fmt_number(val, imag_tol, prec) for val in v) + "]"

def print_topk_global(raw: np.ndarray, vecs: np.ndarray, n_rad: int, top_k_show: int, imag_tol: float):
    flat = raw.flatten()
    order = np.argsort(flat)[::-1]
    total = raw.size
    K = min(top_k_show, total)
    print(f"\n[top-{K}] Global top eigenpairs (centered_global; |Im|<{imag_tol} → real)")
    for rank, fi in enumerate(order[:K], start=1):
        ell = fi // n_rad
        rad = fi %  n_rad
        lam = raw[ell, rad]
        u = vecs[ell][:, rad]
        print(f"  #{rank:02d}  (ℓ={ell}, r={rad})  λ={fmt_number(lam, imag_tol)}")
        print("       u*: ", vec_to_string(u, imag_tol))

# ----------------------- per-volume export helpers -----------------------

def _build_keep_rows_by_ell(order_flat: np.ndarray, n_rad: int, keep_topK: int) -> Dict[int, np.ndarray]:
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
    save_csv: bool = True,
    save_coeffs: bool = True,
    coeffs_keep_rows_by_ell: Optional[Dict[int, np.ndarray]] = None,
    coeffs_complex64: bool = True,
) -> bool:
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

    basename = os.path.splitext(os.path.basename(m_path))[0]
    os.makedirs(out_dir, exist_ok=True)

    # ---- cumulative energy CSV (optional)
    if save_csv:
        ef = energies.flatten()
        ef = ef[order_flat]
        total = float(ef.sum())
        ratios = (np.cumsum(ef) / total) if total > 0 else np.zeros_like(ef)
        k = np.arange(1, ratios.size + 1, dtype=int)
        table = np.column_stack((k, ratios))
        csv_path = os.path.join(out_dir, f"{basename}_coeff_energy_centered_global.csv")
        np.savetxt(csv_path, table, delimiter=",", header="k,ratio", comments="")

    # ---- save coefficients NPZ (optional)
    if save_coeffs:
        coeff_pack["mu_l0"] = (mu_l0.astype(np.complex64) if (mu_l0 is not None and coeffs_complex64) else mu_l0)
        tag = "_topK" if (coeffs_keep_rows_by_ell is not None) else "_all"
        eps_tag = eps_to_tag(eps)
        npz_path = os.path.join(out_dir, f"{basename}_coeffs_L={L}_eps={eps_tag}_centered_global{tag}.npz")
        np.savez_compressed(npz_path, **coeff_pack)

    return True


# ----------------------- main pipeline -----------------------

def main(
    in_dir: str,
    out_dir: str,
    L: int,
    eps: float,
    solver: str,
    top_k: int,
    n_jobs: int,
    store_as_complex64: bool,
    save_full_eigendecomp: bool,
    pbar: bool,
    expect_N: Optional[int],
    imag_tol: float,
    top_show: int,
    export_csv: bool,
    export_coeffs: bool,
    coeffs_keep_topk: bool,
    coeffs_complex64: bool,
    export_n_jobs: int,
    apply_centering_in_export: bool,
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
        print(f"[warn] Folder expect-n={expect_N} but first volume is {N_vol}. "
              f"Proceeding with N_vol={N_vol} for computations.")
    print(f"[build] Volumes={len(mats)} | N_vol={N_vol}, L={L}, eps={eps}")

    eps_tag = eps_to_tag(eps)

    # ---- accumulate covariance pieces & ℓ=0 vectors
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
    avg_over_v = sum_over_v / M
    L_blocks, n_rad, _ = avg_over_v.shape
    assert L_blocks == L

    # per-ℓ normalization by (2ℓ+1)
    S_raw = np.empty_like(avg_over_v)
    for ell in range(L):
        Bl = avg_over_v[ell] / float(2*ell + 1)
        S_raw[ell] = 0.5 * (Bl + Bl.conj().T)

    # global mean (DC) vector across volumes for ℓ=0
    mu_l0 = (sum_l0 / M).astype(np.complex128)

    # ---- GLOBAL CENTERING (only ℓ=0)
    S_centered = S_raw.copy()
    S_centered[0] = S_centered[0] - np.outer(mu_l0, mu_l0.conj())
    S_centered[0] = 0.5 * (S_centered[0] + S_centered[0].conj().T)

    # ---- save centered covariance blocks
    cov_meta = dict(N=N_vol, L=L, eps=eps, solver=solver, n_rad=n_rad, M=M)
    np.savez_compressed(
        os.path.join(out_dir, f"cov_blocks_N={N_vol}_L={L}_eps={eps_tag}_centered_global.npz"),
        S_blocks=S_centered, mu_l0=mu_l0, **cov_meta
    )
    print("[save] wrote covariance NPZ (centered_global)")

    # ---- eigendecomposition
    if pbar:
        print("[step] Eigendecomposition …")
    raw_cg, vec_cg = svd_covariance(S_centered, pbar=pbar)

    # ---- PRINT: global leading eigenpairs
    print_topk_global(raw_cg, vec_cg, n_rad, top_k_show=top_show, imag_tol=imag_tol)

    # ---- top-K selection
    total_modes = L * n_rad
    K = min(top_k, total_modes)
    if K < top_k:
        print(f"[warn] Requested top-k={top_k} but only {total_modes} modes exist; using K={K}")

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

    meta_common = dict(
        N=N_vol, L=L, eps=eps, solver=solver,
        n_rad=n_rad, K=K, mu_l0=mu_l0,
        mats_used=np.array([os.path.basename(x) for x in mats])
    )
    np.savez_compressed(
        os.path.join(out_dir, f"top_modes_N={N_vol}_L={L}_eps={eps_tag}_K={K}_centered_global_by_raw.npz"),
        top_vecs=top_vecs_cg,
        top_ell=top_ell_cg, top_rad=top_rad_cg,
        top_vals_raw=top_raw_cg,
        **meta_common
    )

    # save full eigen-decomposition (optional)
    if save_full_eigendecomp:
        np.savez_compressed(
            os.path.join(out_dir, f"full_eigs_N={N_vol}_L={L}_eps={eps_tag}_centered_global.npz"),
            eigvals=raw_cg, eigvecs=vec_cg, **cov_meta, mu_l0=mu_l0
        )

    # ---- per-volume export
    if export_csv or export_coeffs:
        export_workers = (os.cpu_count() or 1) if (export_n_jobs in (-1, None, 0)) else int(export_n_jobs)
        if "nvidia" in str(solver).lower():
            # If you're using GPU in solver path, 1 worker is typically safer
            export_workers = 1 if export_n_jobs in (-1, None, 0) else export_workers

        keep_rows_by_ell = None
        if export_coeffs and coeffs_keep_topk:
            keep_rows_by_ell = _build_keep_rows_by_ell(ord_cg_raw, n_rad, keep_topK=K)

        what = []
        if export_csv:    what.append("CSV")
        if export_coeffs: what.append("coeffs")
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
                    apply_centering_in_export=apply_centering_in_export,
                    save_csv=export_csv,
                    save_coeffs=export_coeffs,
                    coeffs_keep_rows_by_ell=keep_rows_by_ell,
                    coeffs_complex64=coeffs_complex64,
                )
                for m in mats
            ]
            it = as_completed(futures)
            if pbar:
                it = tqdm(it, total=len(futures), desc="Export per-volume", leave=True)
            for fut in it:
                fut.result()

    # ---- summary
    print("\n[summary]")
    print(f"  volumes processed: {M}")
    print(f"  total modes: {L*n_rad}  |  K (global top modes): {K}")
    print("\n[save] wrote:")
    for fn in (
        f"cov_blocks_N={N_vol}_L={L}_eps={eps_tag}_centered_global.npz",
        f"top_modes_N={N_vol}_L={L}_eps={eps_tag}_K={K}_centered_global_by_raw.npz",
    ):
        print("  •", os.path.join(out_dir, fn))
    if save_full_eigendecomp:
        print("  •", os.path.join(out_dir, f"full_eigs_N={N_vol}_L={L}_eps={eps_tag}_centered_global.npz"))
    if export_coeffs:
        if coeffs_keep_topk:
            print(f"  • Per-volume: *_coeffs_L={L}_eps={eps_tag}_centered_global_topK.npz")
        else:
            print(f"  • Per-volume: *_coeffs_L={L}_eps={eps_tag}_centered_global_all.npz")
    if export_csv:
        print("  • Per-volume: *_coeff_energy_centered_global.csv")


if __name__ == "__main__":
    args = parse_args()

    in_dir = args.in_dir if args.in_dir is not None else f"mat_converted_N={args.expect_n}"
    out_dir = args.out_dir if args.out_dir is not None else f"mat_converted_N={args.expect_n}_matrix"

    main(
        in_dir=in_dir,
        out_dir=out_dir,
        L=args.L,
        eps=args.eps,
        solver=args.solver,
        top_k=args.top_k,
        n_jobs=args.n_jobs,
        store_as_complex64=args.store_as_complex64,
        save_full_eigendecomp=args.save_full_eigendecomp,
        pbar=args.pbar,
        expect_N=args.expect_n,
        imag_tol=args.imag_tol,
        top_show=args.top_show,
        export_csv=args.export_csv,
        export_coeffs=args.export_coeffs,
        coeffs_keep_topk=args.coeffs_keep_topk,
        coeffs_complex64=args.coeffs_complex64,
        export_n_jobs=args.export_n_jobs,
        apply_centering_in_export=args.apply_centering_in_export,
    )