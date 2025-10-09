#!/usr/bin/env python3
"""
final_top_eigenvolumes_reconstruct.py

Reconstruct the top-T eigenvolumes (largest eigenvalues, PCA order) from:
  top_modes_N=<N>_L=<L>_K=<K>_centered_global_by_raw.npz

Each eigenvolume corresponds to a single (ℓ, r) eigenvector realized at m=0.
We DO NOT add the dataset mean; these are basis elements (not reconstructions
of any dataset item).

Output directory:  mat_converted_N=<N>_eigenvolumes
Output files:      eigenvol_rank###_ellℓ_rR_lambda<λ>.mrc

Memory safety:
  • If fle.synthesize exists → parallel (1 FLE per worker).
  • Else use dense-B fallback (step3 + B.dot(a)) SEQUENTIALLY to avoid RAM spikes.

Requires: pip install mrcfile
"""

import os
import glob
import gc
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional, Dict, Any, Tuple
from tqdm.auto import tqdm
import mrcfile

# ====================== USER CONFIG ======================
# Option 1: point directly to the pack
TOP_MODES_NPZ = None  # e.g. "mat_converted_N=256_matrix/top_modes_N=256_L=20_K=1000_centered_global_by_raw.npz"

# Option 2: if TOP_MODES_NPZ is None, set NN and auto-glob within the matrix folder
NN = 256
MATRIX_DIR = f"mat_converted_N={NN}_matrix"
NPZ_GLOB = os.path.join(MATRIX_DIR, f"top_modes_N={NN}_L=*_*centered_global_by_raw.npz")

T_TOP = 50            # how many top eigenvolumes to export
M_CHOICE = 0          # which m to realize (0 ⇒ center column)
NORMALIZE = True      # scale each saved volume to max|v|=1
VOXEL_SIZE = 1.0      # MRC voxel size (units if you have them)
VOXEL_DTYPE = np.float32
MAX_WORKERS = -1      # -1 => auto (CPU count). Ignored (set to 1) for dense-B fallback.
# =========================================================

# Your library:
from fle_3d import FLEBasis3D  # noqa: E402

# -------- Worker globals (for parallel synthesize path) --------
_WORKER_FLE: Optional[FLEBasis3D] = None
_B_ZERO_SHAPE: Optional[Tuple[int, int, int]] = None  # shape of b
# ---------------------------------------------------------------

def _resolve_npz_path() -> str:
    if TOP_MODES_NPZ and os.path.exists(TOP_MODES_NPZ):
        return TOP_MODES_NPZ
    cand = sorted(glob.glob(NPZ_GLOB))
    if not cand:
        raise RuntimeError(f"No NPZ found. Set TOP_MODES_NPZ or adjust NPZ_GLOB ({NPZ_GLOB}).")
    return cand[0]

def _save_mrc(path: str, vol: np.ndarray):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with mrcfile.new(path, overwrite=True) as mrc:
        mrc.set_data(vol.astype(VOXEL_DTYPE, copy=False))
        try:
            mrc.voxel_size = (VOXEL_SIZE, VOXEL_SIZE, VOXEL_SIZE)
        except Exception:
            pass

def _build_b_zero_shape(fle: FLEBasis3D) -> Tuple[int, int, int]:
    """Probe the exact b-shape via forward on zeros."""
    v0 = np.zeros((fle.N, fle.N, fle.N), dtype=np.float32)
    z0 = fle.step1(v0)
    b0 = fle.step2(z0)
    return b0.shape  # (n_rad, L, 2*L-1) in your pipeline

def _fle_worker_init(N: int, L: int, eps: float, solver: str):
    """Initializer for synthesize-capable parallel path."""
    global _WORKER_FLE, _B_ZERO_SHAPE
    _WORKER_FLE = FLEBasis3D(
        N=N, bandlimit=L, eps=eps, max_l=L,
        mode="complex", sph_harm_solver=solver, reduce_memory=True
    )
    _B_ZERO_SHAPE = _build_b_zero_shape(_WORKER_FLE)

def _synthesize_with_worker(u_rad: np.ndarray, ell: int, rank_idx: int,
                            rad: int, lam: float, out_dir: str) -> str:
    """
    Parallel worker: build b with only (ℓ, r, m=0) active and synthesize volume.
    """
    global _WORKER_FLE, _B_ZERO_SHAPE
    n_rad, L, maxw = _B_ZERO_SHAPE
    # Build b tensor (zeros) and place u at (ell, m_idx)
    b = np.zeros((n_rad, L, maxw), dtype=np.complex128)
    m_idx = ell + M_CHOICE  # m=0 ⇒ index ell
    if not (0 <= m_idx < 2*ell + 1):
        raise ValueError(f"m={M_CHOICE} is out of range for ℓ={ell}.")
    b[:, ell, m_idx] = u_rad
    # Synthesize
    if not hasattr(_WORKER_FLE, "synthesize"):
        raise RuntimeError("synthesize() missing in worker path (should have been gated in main).")
    vol = _WORKER_FLE.synthesize(b)
    vol = np.real(vol)
    if NORMALIZE:
        vmax = float(np.max(np.abs(vol))) or 1.0
        vol = vol / vmax
    fname = f"eigenvol_rank{rank_idx+1:03d}_ell{ell}_r{rad}_lambda{lam:.6e}.mrc"
    path = os.path.join(out_dir, fname)
    _save_mrc(path, vol)
    del b, vol
    gc.collect()
    return path

def main():
    # ---------- Load pack ----------
    npz_path = _resolve_npz_path()
    pack = np.load(npz_path, allow_pickle=False)
    top_vecs = pack["top_vecs"]        # (K, n_rad) complex
    top_ell  = pack["top_ell"]         # (K,)
    top_rad  = pack["top_rad"]         # (K,)
    top_vals = pack["top_vals_raw"]    # (K,)
    N        = int(pack["N"])
    L        = int(pack["L"])
    n_rad    = int(pack["n_rad"])
    eps      = float(pack["eps"])
    solver   = str(pack["solver"])
    K_total  = int(pack["K"])

    # Ensure true PCA order by λ (defensive)
    order = np.argsort(top_vals)[::-1]
    top_vecs = top_vecs[order]
    top_ell  = top_ell[order]
    top_rad  = top_rad[order]
    top_vals = top_vals[order]

    T = min(T_TOP, K_total)
    out_dir = f"mat_converted_N={N}_eigenvolumes"
    os.makedirs(out_dir, exist_ok=True)

    print(f"[info] NPZ: {npz_path}")
    print(f"[info] N={N}, L={L}, n_rad={n_rad}, eps={eps}, solver={solver}")
    print(f"[info] Exporting top {T} eigenvolumes → {out_dir}")

    # ---------- Capability probe ----------
    fle_probe = FLEBasis3D(
        N=N, bandlimit=L, eps=eps, max_l=L,
        mode="complex", sph_harm_solver=solver, reduce_memory=True
    )
    has_synthesize = hasattr(fle_probe, "synthesize")
    has_denseB = hasattr(fle_probe, "create_denseB") and hasattr(fle_probe, "step3")

    if has_synthesize:
        # ---------- Parallel synthesize path ----------
        if MAX_WORKERS in (-1, None, 0):
            workers = os.cpu_count() or 1
            if "nvidia" in solver.lower() or "cuda" in solver.lower():
                # GPU: safer to keep 1 unless you know your VRAM
                workers = 1
        else:
            workers = int(MAX_WORKERS)

        print(f"[info] Using synthesize(); workers={workers}")
        results = []
        with ProcessPoolExecutor(max_workers=workers,
                                 initializer=_fle_worker_init,
                                 initargs=(N, L, eps, solver)) as ex:
            futs = []
            for k in range(T):
                u   = np.ascontiguousarray(top_vecs[k])  # (n_rad,)
                ell = int(top_ell[k])
                rad = int(top_rad[k])
                lam = float(top_vals[k])
                futs.append(ex.submit(_synthesize_with_worker, u, ell, k, rad, lam, out_dir))
            for fut in tqdm(as_completed(futs), total=len(futs), desc="Eigenvolumes"):
                results.append(fut.result())

        print("[done] Wrote:")
        for p in sorted(results):
            print("  •", p)

    elif has_denseB:
        # ---------- Sequential dense-B fallback (RAM-safe) ----------
        print("[info] synthesize() not found; using dense-B fallback SEQUENTIALLY.")
        # Build dense B once
        try:
            B = fle_probe.create_denseB(numthread=1)
        except Exception as e:
            raise RuntimeError(f"create_denseB failed: {e}")
        # Get b shape template
        n_rad0, L0, maxw0 = _build_b_zero_shape(fle_probe)
        if (n_rad0 != n_rad) or (L0 != L):
            print(f"[warn] b-shape probe ({n_rad0},{L0},{maxw0}) vs NPZ ({n_rad},{L},2L-1). Proceeding.")
        # Loop sequentially
        wrote = []
        for k in tqdm(range(T), desc="Eigenvolumes"):
            u   = np.ascontiguousarray(top_vecs[k])  # (n_rad,)
            ell = int(top_ell[k])
            rad = int(top_rad[k])
            lam = float(top_vals[k])
            # Build b with only (ℓ, m=0) active
            b = np.zeros((n_rad0, L0, maxw0), dtype=np.complex128)
            m_idx = ell + M_CHOICE
            if not (0 <= m_idx < 2*ell + 1):
                raise ValueError(f"m={M_CHOICE} out of range for ℓ={ell}.")
            b[:, ell, m_idx] = u
            # Convert b -> a then volume via dense B
            a = fle_probe.step3(b)
            vol = B.dot(a).reshape(N, N, N)
            vol = np.real(vol)
            if NORMALIZE:
                vmax = float(np.max(np.abs(vol))) or 1.0
                vol = vol / vmax
            fname = f"eigenvol_rank{k+1:03d}_ell{ell}_r{rad}_lambda{lam:.6e}.mrc"
            path = os.path.join(out_dir, fname)
            _save_mrc(path, vol)
            wrote.append(path)
            del b, a, vol
            gc.collect()
        print("[done] Wrote:")
        for p in wrote:
            print("  •", p)
    else:
        raise RuntimeError(
            "Your FLEBasis3D exposes neither synthesize() nor (create_denseB + step3). "
            "Please enable one of those to reconstruct volumes."
        )

if __name__ == "__main__":
    main()
