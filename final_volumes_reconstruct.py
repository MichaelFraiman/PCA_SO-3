#!/usr/bin/env python3
# reconstruct_targets_from_centered_pca_BY_EIGENVOLUME.py
#
# Reconstruct target volumes using top-K **eigenvolumes** from centered-global PCA packs:
#   top_modes_N=<N>_L=<L>_K=<Kpairs>_centered_global_by_raw.npz
#
# For each target .mat and for each K in K_LIST:
#   1) Forward transform to coefficients b
#   2) Center only the (ℓ=0, m=0) column by subtracting mu_l0
#   3) Build an expanded list of actual eigenvolumes: one item per (ℓ,r,m)
#      in descending eigenvalue order (pairs repeated 2ℓ+1 times).
#   4) Project onto the first K expanded eigenvolumes, accumulating b̂' (centered)
#   5) Uncenter: add mu_l0 back to the (ℓ=0, m=0) column
#   6) Synthesize and save as .mrc
#
# Notes:
#   • Parallelization is per (NPZ × target) pair. Each worker reuses its FLE and (if needed) dense B.
#   • If fle.synthesize() is missing, we fall back to step3 + create_denseB (sequential inside worker).
#
import os
import glob
import gc
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple

from tqdm.auto import tqdm
import mrcfile
from scipy.io import loadmat

# ============================== CONFIG ==============================

NN = 22  # grid size N; used to resolve default paths

# Where your NPZs (created by the global-centering covariance script) live:
NPZ_DIR = f"mat_converted_N={NN}_matrix"

# If you want to hardcode specific NPZ files, list them here; otherwise we glob:
NPZ_PATHS: List[str] = [
    # Example:
    # os.path.join(NPZ_DIR, f"top_modes_N={NN}_L=20_K=1000_centered_global_by_raw.npz"),
]

# Fallback glob if NPZ_PATHS is empty:
NPZ_GLOB = os.path.join(NPZ_DIR, f"top_modes_N={NN}_L=*_*centered_global_by_raw.npz")

# Directory with the .mat target volumes:
RESCALED_DIR = f"mat_converted_N={NN}"

# Target basenames (without .mat) to reconstruct:
TARGETS = ["1avo", "1dgb"]

# K values to reconstruct for each target
K_LIST = [5, 10, 20, 50, 100, 200]

# Output directory:
OUT_DIR = f"mat_converted_N={NN}_reconstructed_mrc"

# MRC voxel size metadata (units optional):
VOXEL_SIZE = 1.0
VOXEL_DTYPE = np.float32

# Parallelization:
#   -1 → auto (CPU count); on GPU synthesize we force 1 unless you override below.
MAX_WORKERS = 5

# ====================================================================

# Your library
from fle_3d import FLEBasis3D  # noqa: E402


# -------------------------- utilities --------------------------

def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p

def save_mrc(path: str, vol: np.ndarray):
    with mrcfile.new(path, overwrite=True) as mrcf:
        mrcf.set_data(vol.astype(VOXEL_DTYPE, copy=False))
        try:
            mrcf.voxel_size = (VOXEL_SIZE, VOXEL_SIZE, VOXEL_SIZE)
        except Exception:
            pass

def get_mat_path(basename: str) -> str:
    return os.path.join(RESCALED_DIR, f"{basename}.mat")

def mat_load_and_normalize(path: str) -> Tuple[np.ndarray, str]:
    data = loadmat(path)
    key = next(k for k in data if not k.startswith("__"))
    v = np.ascontiguousarray(data[key])
    vmax = np.max(np.abs(v))
    return (v if vmax == 0 else (v / vmax)).astype(np.float32, copy=False), key

def build_fle(N: int, L: int, eps: float, solver: str) -> FLEBasis3D:
    return FLEBasis3D(
        N=N, bandlimit=L, eps=eps, max_l=L,
        mode="complex", sph_harm_solver=solver, reduce_memory=True
    )


# -------------------------- core ops (worker) --------------------------

def reconstruct_target_worker(npz_path: str, target_basename: str, k_list: List[int]) -> List[str]:
    """
    Worker routine: for a single (npz, target) pair, reconstruct target for each K in k_list
    using **expanded** eigenvolumes (ℓ,r,m).
    Returns list of written file paths.
    """
    # Load PCA pack (centered_global_by_raw)
    data = np.load(npz_path, allow_pickle=False)
    top_vecs = data["top_vecs"]        # (Kpairs, n_rad) complex (radial eigenvectors)
    top_ell  = data["top_ell"]         # (Kpairs,)
    top_rad  = data["top_rad"]         # (Kpairs,)
    lambdas  = data["top_vals_raw"]    # (Kpairs,)
    N        = int(data["N"])
    L        = int(data["L"])
    n_rad    = int(data["n_rad"])
    eps      = float(data["eps"])
    solver   = str(data["solver"])
    mu_l0    = data["mu_l0"]           # (n_rad,)

    # Ensure sorted by eigenvalue desc (defense)
    order = np.argsort(lambdas)[::-1]
    top_vecs = top_vecs[order]
    top_ell  = top_ell[order]
    top_rad  = top_rad[order]
    lambdas  = lambdas[order]

    # Load & normalize target volume
    target_path = get_mat_path(target_basename)
    v, _ = mat_load_and_normalize(target_path)
    if v.shape[0] != N:
        raise RuntimeError(f"[recon] Target N={v.shape[0]} mismatch pack N={N}: {target_path}")

    # Build FLE and (if needed) dense B
    fle = build_fle(N, L, eps, solver)
    has_synthesize = hasattr(fle, "synthesize")
    B_dense = None
    if not has_synthesize:
        try:
            B_dense = fle.create_denseB(numthread=1)
        except Exception as e:
            raise RuntimeError(f"[recon] create_denseB failed: {e}")

    # Forward transform and center (ℓ=0, m=0) for projection
    z = fle.step1(v)
    b = fle.step2(z)                        # shape (n_rad, L, 2*L-1)
    if b.shape[0] != n_rad:
        raise RuntimeError(f"[recon] n_rad mismatch: pack {n_rad} vs b {b.shape[0]}")
    b_centered = b.copy()
    b_centered[:, 0, 0] = b_centered[:, 0, 0] - mu_l0  # only (ℓ=0,m=0)

    # Prepare outputs
    pack_base = os.path.splitext(os.path.basename(npz_path))[0]
    out_sub = ensure_dir(os.path.join(OUT_DIR, pack_base, f"recon_target-N={N}_{target_basename}"))
    written: List[str] = []

    # Save original for reference
    orig_mrc = os.path.join(out_sub, f"{target_basename}_N={N}_original.mrc")
    save_mrc(orig_mrc, v)
    written.append(orig_mrc)

    # -------- Build expanded eigenvolume order (one item per (pair r, m)) --------
    # Pair order already sorted by λ desc; we expand each pair into its (2ℓ+1) m-entries.
    # expanded: list of (r_index, m_idx) where m_idx ∈ [0 .. 2ℓ]
    expanded: List[Tuple[int, int]] = []
    for r in range(top_vecs.shape[0]):
        ell = int(top_ell[r])
        expanded.extend((r, mm) for mm in range(2*ell + 1))

    E_total = len(expanded)  # total # of actual eigenvolumes available from this pack

    # -------- Precompute all projection coefficients α_{r,mm} = <u_r, b_centered[:,ℓ_r,mm]> --------
    # Store as a list of arrays, one per pair r (length 2ℓ+1 each).
    proj_coeffs: List[np.ndarray] = []
    for r in range(top_vecs.shape[0]):
        ell = int(top_ell[r])
        u_r = top_vecs[r].astype(np.complex128, copy=False)  # (n_rad,)
        width = 2*ell + 1
        # Dot u_r with each m-column at this ℓ
        # b_centered[:, ell, mm] has shape (n_rad,)
        alphas = np.empty(width, dtype=np.complex128)
        for mm in range(width):
            alphas[mm] = np.vdot(u_r, b_centered[:, ell, mm])
        proj_coeffs.append(alphas)

    # -------- Per-K reconstructions using ONLY the first K expanded eigenvolumes --------
    for K in k_list:
        K_use = int(min(K, E_total))

        # Allocate centered reconstruction coefficients b̂' (same shape as b)
        b_hat_centered = np.zeros_like(b_centered)

        # Accumulate contributions for the first K expanded eigenvolumes
        # Each expanded item contributes only to its specific (ℓ_r, m) column.
        used = 0
        for (r, mm) in expanded:
            if used >= K_use:
                break
            ell = int(top_ell[r])
            u_r = top_vecs[r].astype(np.complex128, copy=False)
            alpha = proj_coeffs[r][mm]
            b_hat_centered[:, ell, mm] += alpha * u_r
            used += 1

        # Uncenter back (only ℓ=0, m=0)
        b_hat = b_hat_centered
        b_hat[:, 0, 0] = b_hat[:, 0, 0] + mu_l0

        # Synthesize volume
        if has_synthesize:
            vol = fle.synthesize(b_hat)
        else:
            a = fle.step3(b_hat)
            vol = B_dense.dot(a).reshape(N, N, N)

        vol = np.real(vol).astype(np.float32, copy=False)

        out_mrc = os.path.join(out_sub, f"{target_basename}_N={N}_approx-KEIG={K_use}.mrc")
        save_mrc(out_mrc, vol)
        written.append(out_mrc)

        # cleanup between K's
        del b_hat_centered, b_hat, vol
        gc.collect()

    return written


# ----------------------------- main -----------------------------

def main():
    ensure_dir(OUT_DIR)

    # Resolve NPZ list
    if NPZ_PATHS:
        npz_list: List[str] = [p for p in NPZ_PATHS if os.path.exists(p)]
    else:
        npz_list: List[str] = sorted(glob.glob(NPZ_GLOB))

    if not npz_list:
        raise RuntimeError(f"No NPZ files found. Checked list={NPZ_PATHS} and glob={NPZ_GLOB}")

    # Verify target files exist
    target_paths = [get_mat_path(bn) for bn in TARGETS]
    missing = [p for p in target_paths if not os.path.exists(p)]
    if missing:
        raise RuntimeError(f"Missing target .mat files: {missing}")

    # Build task list (one worker per (npz, target))
    tasks: List[Tuple[str, str, List[int]]] = []
    for npz in npz_list:
        for bn in TARGETS:
            tasks.append((npz, bn, K_LIST))

    # Worker budget
    if MAX_WORKERS in (-1, None, 0):
        workers = os.cpu_count() or 1
    else:
        workers = int(MAX_WORKERS)

    # If any pack uses a GPU solver, be conservative with workers (VRAM) unless user forced MAX_WORKERS.
    if MAX_WORKERS in (-1, None, 0):
        try:
            s = str(np.load(npz_list[0], allow_pickle=False)["solver"])
            if "cuda" in s.lower() or "nvidia" in s.lower():
                workers = 1
        except Exception:
            pass

    print(f"[info] Jobs: {len(tasks)} | workers: {workers}")
    print(f"[info] Output dir: {OUT_DIR}")

    written_all: List[str] = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(reconstruct_target_worker, npz, tgt, K_LIST) for (npz, tgt, _) in tasks]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Reconstructions"):
            written_all.extend(fut.result())

    print("[done] Wrote:")
    for p in written_all:
        print("  •", p)


if __name__ == "__main__":
    main()
