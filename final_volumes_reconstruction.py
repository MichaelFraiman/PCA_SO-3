#!/usr/bin/env python3
# reconstruct_targets_from_centered_pca.py
#
# Reconstruct target volumes using top-K eigenvolumes from centered-global PCA packs:
#   top_modes_N=<N>_L=<L>_K=<K>_centered_global_by_raw.npz
#
# For each target .mat and for each K in K_LIST:
#   1) Forward transform to coefficients b
#   2) Center only the (ℓ=0, m=0) column by subtracting mu_l0
#   3) Project onto the first K PCA modes (global order across ℓ,r), accumulating
#      b̂' (centered coefficients)
#   4) Uncenter: add mu_l0 back to the (ℓ=0, m=0) column
#   5) Synthesize and save as .mrc
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
from typing import List, Tuple, Optional, Dict, Any

from tqdm.auto import tqdm
import mrcfile
from scipy.io import loadmat

# ============================== CONFIG ==============================

NN = 64  # grid size N; used to resolve default paths

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
#TARGETS = ["1avo", "5cpi", "1fzf", "2qtb"]
TARGETS = ["1dgb"]

# K values to reconstruct for each target (PCA order across all ℓ,r modes):
K_LIST = [1, 5, 10, 20, 100, 200, 500]

# Output directory:
OUT_DIR = f"mat_converted_N={NN}_reconstructed_mrc"

# MRC voxel size metadata (units optional, e.g., Å if you know it):
VOXEL_SIZE = 1.0
VOXEL_DTYPE = np.float32

# Parallelization:
#   -1 → auto (CPU count); on GPU synthesize we force 1 unless you override below.
#MAX_WORKERS = -1 something is wrong here
MAX_WORKERS = 12

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
    return (v if vmax == 0 else (v / vmax)).astype(np.float32), key

def build_fle(N: int, L: int, eps: float, solver: str) -> FLEBasis3D:
    return FLEBasis3D(
        N=N, bandlimit=L, eps=eps, max_l=L,
        mode="complex", sph_harm_solver=solver, reduce_memory=True
    )

def probe_b_shape(fle: FLEBasis3D) -> Tuple[int, int, int]:
    """Get exact b tensor shape used by fle.step2 (rectangular last dim)."""
    v0 = np.zeros((fle.N, fle.N, fle.N), dtype=np.float32)
    z0 = fle.step1(v0)
    b0 = fle.step2(z0)
    return b0.shape  # (n_rad, L, 2*L-1) in this codebase


# -------------------------- core ops (worker) --------------------------

def reconstruct_target_worker(npz_path: str, target_basename: str, k_list: List[int]) -> List[str]:
    """
    Worker routine: for a single (npz, target) pair, reconstruct target for each K in k_list.
    Returns list of written file paths.
    """
    # Load PCA pack (centered_global_by_raw)
    data = np.load(npz_path, allow_pickle=False)
    top_vecs = data["top_vecs"]        # (K, n_rad) complex
    top_ell  = data["top_ell"]         # (K,)
    top_rad  = data["top_rad"]         # (K,)
    lambdas  = data["top_vals_raw"]    # (K,)
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
        # step3 + create_denseB fallback (RAM-aware: build once per worker)
        try:
            B_dense = fle.create_denseB(numthread=1)
        except Exception as e:
            raise RuntimeError(f"[recon] create_denseB failed: {e}")

    # Forward transform and center (ℓ=0, m=0) for projection
    z = fle.step1(v)
    b = fle.step2(z)                        # shape (n_rad, L, 2*L-1)
    if b.shape[0] != n_rad:
        # Defense if pack n_rad mismatches; usually shouldn't happen
        raise RuntimeError(f"[recon] n_rad mismatch: pack {n_rad} vs b {b.shape[0]}")
    b_centered = b.copy()
    b_centered[:, 0, 0] = b_centered[:, 0, 0] - mu_l0  # <<< global centering (only ℓ=0,m=0)

    # Prepare outputs
    pack_base = os.path.splitext(os.path.basename(npz_path))[0]
    out_sub = ensure_dir(os.path.join(OUT_DIR, pack_base, f"recon_target-N={N}_{target_basename}"))
    written: List[str] = []

    # Save original for reference
    orig_mrc = os.path.join(out_sub, f"{target_basename}_N={N}_original.mrc")
    save_mrc(orig_mrc, v)
    written.append(orig_mrc)

    # Per-K reconstructions (sequential inside worker to control RAM)
    K_avail = top_vecs.shape[0]
    for K in k_list:
        K_use = min(int(K), K_avail)

        # Allocate centered reconstruction coefficients b̂' (same shape as b)
        n_rad0, L0, maxw0 = b_centered.shape
        b_hat_centered = np.zeros_like(b_centered)

        # Accumulate projections for the first K modes (global order)
        for r in range(K_use):
            ell = int(top_ell[r])
            u   = top_vecs[r].astype(np.complex128, copy=False)  # (n_rad,)

            # Project each m at this ℓ onto the rank-1 subspace spanned by u
            width = 2*ell + 1
            for mm in range(width):
                alpha = np.vdot(u, b_centered[:, ell, mm])       # complex scalar
                b_hat_centered[:, ell, mm] += alpha * u

        # Uncenter back (only ℓ=0, m=0)
        b_hat = b_hat_centered
        b_hat[:, 0, 0] = b_hat[:, 0, 0] + mu_l0

        # Synthesize volume
        if has_synthesize:
            vol = fle.synthesize(b_hat)
        else:
            a = fle.step3(b_hat)                # coefficients compatible with dense B
            vol = B_dense.dot(a).reshape(N, N, N)

        vol = np.real(vol).astype(np.float32, copy=False)

        out_mrc = os.path.join(out_sub, f"{target_basename}_N={N}_approx-K={K_use}.mrc")
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
    npz_list: List[str] = []
    if NPZ_PATHS:
        npz_list = [p for p in NPZ_PATHS if os.path.exists(p)]
    else:
        npz_list = sorted(glob.glob(NPZ_GLOB))

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

    # If any pack uses a GPU solver, be conservative with workers (VRAM)
    # We’ll downscale to 1 unless user explicitly set MAX_WORKERS.
    if MAX_WORKERS in (-1, None, 0):
        try:
            # Peek at the first NPZ to read 'solver'
            s = str(np.load(npz_list[0], allow_pickle=False)["solver"])
            if "cuda" in s.lower() or "nvidia" in s.lower():
                workers = 1
        except Exception:
            pass

    print(f"[info] Jobs: {len(tasks)} | workers: {workers}")
    print(f"[info] Output dir: {OUT_DIR}")

    # Parallel map over tasks with a clean progress bar
    written_all: List[str] = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(reconstruct_target_worker, npz, tgt, K_LIST) for (npz, tgt, _) in tasks]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Reconstructions"):
            paths = fut.result()
            written_all.extend(paths)

    # Summary
    print("[done] Wrote:")
    for p in written_all:
        print("  •", p)


if __name__ == "__main__":
    main()
