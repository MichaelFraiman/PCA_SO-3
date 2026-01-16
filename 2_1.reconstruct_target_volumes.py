#!/usr/bin/env python3
"""
reconstruct_targets_from_centered_pca_BY_EIGENVOLUME.py

Reconstruct target volumes using top-K **eigenvolumes** from centered-global PCA packs:
  top_modes_N=<N>_L=<L>_K=<Kpairs>_centered_global_by_raw.npz

For each target .mat and for each K in K_LIST:
  1) Forward transform target to coefficients b
  2) Center only the (ℓ=0, m=0) column by subtracting mu_l0 (from the PCA pack)
  3) Build an expanded list of actual eigenvolumes: one item per (ℓ,r,m)
     in descending eigenvalue order (pairs repeated 2ℓ+1 times).
  4) Project onto the first K expanded eigenvolumes, accumulating b̂' (centered)
  5) Uncenter: add mu_l0 back to the (ℓ=0, m=0) column
  6) Synthesize and save as .mrc

Notes:
  • Parallelization is per (NPZ × target) pair.
  • Prevents segfaults on macOS by forcing single-threaded linear algebra libraries.

Examples:
  python3 reconstruct_targets_from_centered_pca_BY_EIGENVOLUME.py --targets 1avo 1dgb --k-list 10 50 100
  python3 reconstruct_targets_from_centered_pca_BY_EIGENVOLUME.py --nn 22 --targets-dir my_mats --out-dir my_recons
"""

import os
import sys

# CRITICAL FIX FOR MACOS SEGFAULT:
# We must strictly control OpenMP/MKL threading before importing numpy/torch.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import glob
import gc
import argparse
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple, Optional

from tqdm.auto import tqdm
import mrcfile
from scipy.io import loadmat

# Your library
from fle_3d import FLEBasis3D 


# ============================== ARGUMENT PARSING ==============================

def parse_args():
    p = argparse.ArgumentParser(
        description="Reconstruct target volumes using expanded eigenvolumes from centered PCA packs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Paths / Selection
    p.add_argument("--nn", type=int, default=22,
                   help="Grid size N, used for default directory naming.")
    p.add_argument("--L", type=int, default=0,
                   help="Filter NPZ selection by bandlimit L. 0 disables filtering (L=*).")
    p.add_argument("--eps", type=float, default=0.0,
                   help="Filter NPZ selection by eps (formatted like 1e-06). 0 disables filtering (eps=*).")
    p.add_argument("--npz-dir", default=None,
                   help="Directory containing PCA NPZ files. Default: mat_converted_N=<NN>_matrix")
    p.add_argument("--npz-glob", default=None,
                   help="Glob pattern for NPZ files. Default: <npz_dir>/top_modes_N=<NN>_L=*_*centered_global_by_raw.npz")
    p.add_argument("--targets-dir", default=None,
                   help="Directory containing target .mat files. Default: mat_converted_N=<NN>")
    
    # Targets & K
    p.add_argument("--targets", nargs="+", default=["1avo", "1dgb"],
                   help="List of target basenames (without .mat) to reconstruct.")
    p.add_argument("--k-list", nargs="+", type=int, default=[5, 10, 20, 50, 100, 200],
                   help="List of K values (number of eigenvolumes) to reconstruct.")

    # Output
    p.add_argument("--out-dir", default=None,
                   help="Output root directory. Default: mat_converted_N=<NN>_reconstructed_mrc")
    p.add_argument("--voxel-size", type=float, default=1.0,
                   help="MRC voxel size.")
    
    # Performance
    p.add_argument("--workers", type=int, default=5,
                   help="Number of worker processes. -1 for auto.")

    return p.parse_args()


# ============================== WORKER LOGIC ==============================

def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p

def save_mrc(path: str, vol: np.ndarray, voxel_size: float):
    # Standardize dtype to float32 for MRC
    with mrcfile.new(path, overwrite=True) as mrcf:
        mrcf.set_data(vol.astype(np.float32, copy=False))
        try:
            mrcf.voxel_size = (voxel_size, voxel_size, voxel_size)
        except Exception:
            pass

def mat_load_and_normalize(path: str) -> Tuple[np.ndarray, str]:
    data = loadmat(path)
    # Find the variable name (skipping internal __ keys)
    key = next(k for k in data if not k.startswith("__"))
    v = np.ascontiguousarray(data[key])
    vmax = np.max(np.abs(v))
    return (v if vmax == 0 else (v / vmax)).astype(np.float32, copy=False), key

def build_fle(N: int, L: int, eps: float, solver: str) -> FLEBasis3D:
    return FLEBasis3D(
        N=N, bandlimit=L, eps=eps, max_l=L,
        mode="complex", sph_harm_solver=solver, reduce_memory=True
    )

def reconstruct_target_worker(
    npz_path: str,
    target_path: str,
    target_basename: str,
    k_list: List[int],
    out_dir_root: str,
    voxel_size: float
) -> List[str]:
    """
    Worker routine: for a single (npz, target) pair, reconstruct target for each K in k_list
    using **expanded** eigenvolumes (ℓ,r,m).
    """
    # 1. Load PCA pack
    data = np.load(npz_path, allow_pickle=False)
    top_vecs = data["top_vecs"]        # (Kpairs, n_rad) complex (radial eigenvectors)
    top_ell  = data["top_ell"]         # (Kpairs,)
    lambdas  = data["top_vals_raw"]    # (Kpairs,)
    N        = int(data["N"])
    L        = int(data["L"])
    n_rad    = int(data["n_rad"])
    eps      = float(data["eps"])
    eps_tag  = format(eps, ".0e").replace("+", "")
    solver   = str(data["solver"])
    mu_l0    = data["mu_l0"]           # (n_rad,)

    # Defense: Ensure sorted by eigenvalue descending
    order = np.argsort(lambdas)[::-1]
    top_vecs = top_vecs[order]
    top_ell  = top_ell[order]
    # top_rad not strictly needed for projection logic, but exists in pack

    # 2. Load & normalize target volume
    if not os.path.exists(target_path):
        raise RuntimeError(f"Target file not found: {target_path}")
    
    v, _ = mat_load_and_normalize(target_path)
    if v.shape[0] != N:
        raise RuntimeError(f"[recon] Target N={v.shape[0]} mismatch pack N={N}: {target_path}")

    # 3. Build FLE
    fle = build_fle(N, L, eps, solver)
    has_synthesize = hasattr(fle, "synthesize")
    B_dense = None
    if not has_synthesize:
        try:
            B_dense = fle.create_denseB(numthread=1)
        except Exception as e:
            raise RuntimeError(f"[recon] create_denseB failed: {e}")

    # 4. Forward transform and center (ℓ=0, m=0)
    z = fle.step1(v)
    b = fle.step2(z)                        # shape (n_rad, L, 2*L-1)
    if b.shape[0] != n_rad:
        raise RuntimeError(f"[recon] n_rad mismatch: pack {n_rad} vs b {b.shape[0]}")
    
    b_centered = b.copy()
    b_centered[:, 0, 0] = b_centered[:, 0, 0] - mu_l0  # subtract global mean from DC component

    # Prepare outputs
    pack_base = os.path.splitext(os.path.basename(npz_path))[0]
    # Structure: out_dir / <pack_name> / recon_target-N=<N>_<basename> / ...
    out_sub = ensure_dir(os.path.join(out_dir_root, pack_base, f"recon_target-N={N}_{target_basename}"))
    written: List[str] = []

    # Save original for reference
    orig_mrc = os.path.join(out_sub, f"{target_basename}_N={N}_L={L}_eps={eps_tag}_original.mrc")
    save_mrc(orig_mrc, v, voxel_size)
    written.append(orig_mrc)

    # 5. Build expanded eigenvolume order
    # Each pair (ℓ, r) in `top_vecs` corresponds to (2ℓ+1) actual eigenvolumes (m ∈ [-ℓ...ℓ])
    # sharing the same radial function and eigenvalue.
    # We expand them into a list of (r_index, m_idx_in_b) tuples.
    expanded: List[Tuple[int, int]] = []
    for r in range(top_vecs.shape[0]):
        ell = int(top_ell[r])
        # b array has m indices 0..(2ℓ), representing specific m values
        expanded.extend((r, mm) for mm in range(2*ell + 1))

    E_total = len(expanded)

    # 6. Precompute projection coefficients α_{r,mm} = <u_r, b_centered[:,ℓ_r,mm]>
    # Optimization: calculate only once per pair, reuse for all K
    proj_coeffs: List[np.ndarray] = []
    for r in range(top_vecs.shape[0]):
        ell = int(top_ell[r])
        u_r = top_vecs[r].astype(np.complex128, copy=False)  # (n_rad,)
        width = 2*ell + 1
        # b_centered[:, ell, :] is (n_rad, 2ℓ+1)
        # Dot u_r (conj) with each m-column.
        # np.vdot(a, b) does a.conj() @ b.
        # We need <u_r, column> = u_r^H @ column.
        alphas = np.empty(width, dtype=np.complex128)
        cols = b_centered[:, ell, :width]
        
        # Vectorized dot product: u_r^H @ Matrix
        alphas = u_r.conj() @ cols
        proj_coeffs.append(alphas)

    # 7. Reconstruct for each K
    for K in k_list:
        K_use = int(min(K, E_total))

        # Start with zero coefficients
        b_hat_centered = np.zeros_like(b_centered)

        # Accumulate top K eigenvolumes
        used = 0
        for (r, mm) in expanded:
            if used >= K_use:
                break
            
            ell = int(top_ell[r])
            u_r = top_vecs[r].astype(np.complex128, copy=False)
            alpha = proj_coeffs[r][mm]
            
            # Project back: coeff * basis_vector
            b_hat_centered[:, ell, mm] += alpha * u_r
            used += 1

        # 8. Uncenter: Add mean back to DC component
        b_hat = b_hat_centered
        b_hat[:, 0, 0] = b_hat[:, 0, 0] + mu_l0

        # 9. Synthesize volume
        if has_synthesize:
            vol = fle.synthesize(b_hat)
        else:
            a = fle.step3(b_hat)
            vol = B_dense.dot(a).reshape(N, N, N)

        vol = np.real(vol)

        out_mrc = os.path.join(out_sub, f"{target_basename}_N={N}_L={L}_eps={eps_tag}_approx-KEIG={K_use}.mrc")
        save_mrc(out_mrc, vol, voxel_size)
        written.append(out_mrc)

        del b_hat_centered, b_hat, vol
        gc.collect()

    return written


# ============================== MAIN ==============================

def main():
    args = parse_args()
    
    # Resolve Defaults based on NN if None
    nn = args.nn
    npz_dir = args.npz_dir or f"mat_converted_N={nn}_matrix"
    targets_dir = args.targets_dir or f"mat_converted_N={nn}"
    out_dir = args.out_dir or f"mat_converted_N={nn}_reconstructed_mrc"
    
    if args.npz_glob:
        npz_pattern = args.npz_glob
    else:
        L_tag = "*" if int(args.L) <= 0 else str(int(args.L))
        eps_tag = "*" if float(args.eps) <= 0.0 else format(float(args.eps), ".0e").replace("+", "")
        npz_pattern_new = os.path.join(npz_dir, f"top_modes_N={nn}_L={L_tag}_eps={eps_tag}_*centered_global_by_raw.npz")
        npz_patterns = [npz_pattern_new]
        # Only fall back to legacy (no eps tag) when eps filtering is disabled
        if eps_tag == "*":
            npz_pattern_old = os.path.join(npz_dir, f"top_modes_N={nn}_L={L_tag}_*centered_global_by_raw.npz")
            npz_patterns.append(npz_pattern_old)
        npz_pattern = npz_patterns

    ensure_dir(out_dir)

    # 1. Find NPZ files
    npz_list = []
    if isinstance(npz_pattern, (list, tuple)):
        for pat in npz_pattern:
            npz_list += glob.glob(pat)
        npz_list = sorted(set(npz_list))
    else:
        npz_list = sorted(glob.glob(npz_pattern))
    if not npz_list:
        raise RuntimeError(f"No NPZ files found matching: {npz_pattern}")

    # Strict verification: ensure NPZ metadata matches requested filters (also covers --npz-glob).
    if int(args.L) > 0 or float(args.eps) > 0.0:
        L_req = int(args.L)
        eps_req_tag = None
        if float(args.eps) > 0.0:
            eps_req_tag = format(float(args.eps), ".0e").replace("+", "")
        filtered = []
        for pth in npz_list:
            try:
                pack = np.load(pth, allow_pickle=False)
                Lp = int(pack["L"])
                eps_p = float(pack["eps"])
                eps_p_tag = format(eps_p, ".0e").replace("+", "")
            except Exception:
                continue
            if L_req > 0 and Lp != L_req:
                continue
            if eps_req_tag is not None and eps_p_tag != eps_req_tag:
                continue
            filtered.append(pth)
        npz_list = filtered
        if not npz_list:
            raise RuntimeError(
                f"No NPZ files match requested filters: L={args.L}, eps="
                f"{format(float(args.eps), '.0e').replace('+','') if float(args.eps)>0 else '*'}"
                f"\nPatterns used: {npz_pattern}\n"
                "Either recompute the covariance with that eps/L, or disable filtering with --L 0 --eps 0, or pass --npz-glob/--npz explicitly."
            )
    # Prefer newest packs first
    npz_list.sort(key=lambda p: os.path.getmtime(p), reverse=True)

    # 2. Check targets
    # args.targets is a list of basenames
    target_tasks = []
    for bn in args.targets:
        path = os.path.join(targets_dir, f"{bn}.mat")
        if not os.path.exists(path):
            print(f"[warn] Target not found, skipping: {path}")
        else:
            target_tasks.append((bn, path))

    if not target_tasks:
        print("[warn] No valid targets found. Exiting.")
        return

    # 3. Build Job List
    # We want to run every combination of (NPZ, Target)
    jobs = []
    for npz in npz_list:
        for (bn, path) in target_tasks:
            jobs.append((npz, path, bn))

    # 4. Determine Workers
    workers = args.workers
    if workers in (-1, None, 0):
        workers = os.cpu_count() or 1
    
    # Safety check for GPU solvers in the first NPZ
    try:
        tmp = np.load(npz_list[0], allow_pickle=False)
        solver_check = str(tmp["solver"])
        if "cuda" in solver_check.lower() or "nvidia" in solver_check.lower():
            if args.workers == -1: # Only override if auto
                print("[info] Detected GPU solver. Forcing workers=1 to prevent VRAM issues.")
                workers = 1
    except Exception:
        pass

    print(f"[info] Jobs: {len(jobs)} | Workers: {workers}")
    print(f"[info] NPZ Dir: {npz_dir}")
    print(f"[info] Targets Dir: {targets_dir}")
    print(f"[info] Output Dir: {out_dir}")

    # 5. Execute
    written_all = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = []
        for (npz, t_path, t_bn) in jobs:
            futures.append(
                ex.submit(
                    reconstruct_target_worker,
                    npz, t_path, t_bn, args.k_list, out_dir, args.voxel_size
                )
            )
        
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Reconstructing"):
            try:
                written_all.extend(fut.result())
            except Exception as e:
                print(f"[error] Job failed: {e}")

    print("\n[done] Generated files:")
    for p in written_all:
        print(f"  • {p}")


if __name__ == "__main__":
    main()