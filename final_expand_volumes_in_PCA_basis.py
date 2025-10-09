#!/usr/bin/env python3
"""
export_eigen_coeffs_centered_global.py

Compute and save per-volume eigen-coefficients using the *centered_global* top modes
written by `final_cov_matr_centered_global_only.py`.

Parallelization:
  • Unconditionally parallel over volumes (threads or processes), even if solver='nvidia_torch'.
  • Choose executor type via EXECUTOR = 'threads' | 'processes' (threads default).
  • Control worker count via N_JOBS (-1 => all cores).

Output per volume:
  <OUT_ROOT>/<TOP_BASE>/<vol_basename>_N=<N>_eigen_coeffs_top<K_use>.npz
    alpha_padded : (K_use, M_max) complex64
    ell          : (K_use,) int32
    m_counts     : (K_use,) int32
    meta         : N, L, K, n_rad, centered_variant, m_ordering, etc.
"""

import os
import re
import glob
import numpy as np
from typing import Dict, List

from scipy.io import loadmat
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **k): return x

# ----------------------- CONFIG -----------------------
NN             = 64             # grid size (must match your volumes)
L_DEFAULT      = 20
K_TOP_WANT     = 200            # use up to this many top modes from the pack

VOLS_DIR       = f"mat_converted_N={NN}"            # input .mat volumes
BASE_DIR       = f"mat_converted_N={NN}_matrix"     # where top_modes_*.npz lives

# Parallelism knobs (now honored even if solver contains 'nvidia')
N_JOBS         = -1             # -1 => use all CPU cores; else integer >=1
EXECUTOR       = "threads"      # 'threads' or 'processes'
TORCH_THREADS  = 1              # if PyTorch present, set per-worker intra-op threads to avoid oversubscription
# ------------------------------------------------------


# ---------- helpers ----------
def get_mat_list(d, pattern="*.mat"):
    return sorted(glob.glob(os.path.join(d, pattern)))

def load_and_normalize(path, key):
    data = loadmat(path)
    v = np.ascontiguousarray(data[key])
    vmax = float(np.max(np.abs(v)))
    return v if vmax == 0.0 else (v / vmax)

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def find_top_modes_file(base_dir: str, N: int, L: int) -> str:
    """
    Pick the largest-K top-modes file produced by the covariance script.
    e.g., top_modes_N=64_L=20_K=1000_centered_global_by_raw.npz
    """
    pat = os.path.join(base_dir, f"top_modes_N={N}_L={L}_K=*_*centered_global_by_raw.npz")
    cands = glob.glob(pat)
    if not cands:
        pat = os.path.join(base_dir, f"top_modes_N={N}_L={L}_K=*centered_global_by_raw.npz")
        cands = glob.glob(pat)
    if not cands:
        raise SystemExit(f"[error] No top-modes NPZ found in {base_dir} for N={N}, L={L}")
    def parse_K(p):
        m = re.search(r"_K=(\d+)", os.path.basename(p))
        return int(m.group(1)) if m else -1
    cands.sort(key=parse_K)
    return cands[-1]

def group_modes_by_ell(top_ell: np.ndarray, K_use: int) -> Dict[int, List[int]]:
    by = {}
    for r in range(K_use):
        by.setdefault(int(top_ell[r]), []).append(r)
    return by


# ---------- core worker ----------
def export_one_volume(
    mpath: str,
    key: str,
    N_vol: int,
    L: int,
    eps: float,
    solver: str,
    mu_l0: np.ndarray,
    top_vecs: np.ndarray,    # (K_use, n_rad)
    top_ell: np.ndarray,     # (K_use,)
    K_use: int,
    out_dir: str,
) -> str:
    """
    Returns output path of the saved NPZ.
    """
    # local import to keep worker self-contained for process/threads
    from fle_3d import FLEBasis3D
    # keep PyTorch from oversubscribing threads (optional)
    try:
        import torch
        torch.set_num_threads(TORCH_THREADS)
    except Exception:
        pass

    # Build FLE and expand
    fle = FLEBasis3D(
        N=N_vol, bandlimit=L, eps=eps, max_l=L,
        mode="complex", sph_harm_solver=solver, reduce_memory=True
    )
    v = load_and_normalize(mpath, key)
    z = fle.step1(v)
    b = fle.step2(z)                        # (n_rad, L, 2ℓ+1)

    # Apply the SAME global centering convention: only (ℓ=0, m=0)
    b[:, 0, 0] = b[:, 0, 0] - mu_l0

    # Layout
    sizes   = (2 * top_ell[:K_use] + 1).astype(np.int32)
    M_max   = int(sizes.max())
    K_rows  = int(K_use)

    alpha = np.zeros((K_rows, M_max), dtype=np.complex64)

    # Efficiently process by ℓ: stack the selected u_r for that ℓ and multiply once
    by_ell = group_modes_by_ell(top_ell, K_use)

    cache_B: Dict[int, np.ndarray] = {}
    for ell, rows in by_ell.items():
        # B_ell: (n_rad, 2ℓ+1)
        if ell not in cache_B:
            cache_B[ell] = b[:, ell, :2*ell+1]
        B_ell = cache_B[ell]
        # U_sel: (len(rows), n_rad)
        U_sel = top_vecs[rows]              # rows already correspond to those r
        # C_sel: (len(rows), 2ℓ+1)
        C_sel = (U_sel.conj() @ B_ell).astype(np.complex64, copy=False)
        # place into alpha
        for j, r in enumerate(rows):
            cnt = int(sizes[r])
            alpha[r, :cnt] = C_sel[j, :cnt]

    base = os.path.splitext(os.path.basename(mpath))[0]
    out_path = os.path.join(out_dir, f"{base}_N={N_vol}_eigen_coeffs_top{K_rows}.npz")
    np.savez_compressed(
        out_path,
        alpha_padded=alpha,
        ell=top_ell[:K_rows].astype(np.int32),
        m_counts=sizes,
        N=np.int32(N_vol), L=np.int32(L),
        K=np.int32(K_rows),
        m_ordering=np.array("neg_to_pos"),
        centered_variant=np.array("centered_global"),
        volume_normalization=np.array("max_abs_per_volume"),
    )
    return out_path


# ---------- main ----------
def main():
    # Load top modes
    top_path = find_top_modes_file(BASE_DIR, N=NN, L=L_DEFAULT)
    top_base = os.path.splitext(os.path.basename(top_path))[0]
    pack = np.load(top_path, allow_pickle=False)

    N0      = int(pack["N"])
    L0      = int(pack["L"])
    eps     = float(pack["eps"])
    solver  = str(pack["solver"])
    n_rad   = int(pack["n_rad"])
    K_avail = int(pack["K"])
    mu_l0   = pack["mu_l0"].astype(np.complex128)
    top_vecs_all = pack["top_vecs"]                  # (K_avail, n_rad)
    top_ell_all  = pack["top_ell"].astype(np.int32)  # (K_avail,)

    K_use = min(K_TOP_WANT, K_avail)
    top_vecs = top_vecs_all[:K_use]
    top_ell  = top_ell_all[:K_use]

    # Discover volumes and MATLAB key
    mats = get_mat_list(VOLS_DIR)
    if not mats:
        raise SystemExit(f"[error] No .mat files found in {VOLS_DIR}")
    samp = loadmat(mats[0])
    key = next(k for k in samp.keys() if not k.startswith("__"))
    N_vol = int(np.ascontiguousarray(samp[key]).shape[0])

    # Output folder: dynamic by K_use so it matches what distribution-builder expects
    OUT_ROOT = f"mat_converted_N={NN}_eigen_coeffs_top{K_use}"
    out_dir = ensure_dir(os.path.join(OUT_ROOT, top_base))

    # Worker count & executor (unconditional parallelism)
    workers = (os.cpu_count() or 1) if N_JOBS in (-1, None, 0) else int(N_JOBS)
    Executor = ThreadPoolExecutor if EXECUTOR.lower().startswith("thread") else ProcessPoolExecutor

    print(f"[setup] top pack : {os.path.basename(top_path)}")
    print(f"[setup] volumes  : {VOLS_DIR}  ({len(mats)} files)")
    print(f"[setup] using    : K_use={K_use} (of {K_avail}), n_rad={n_rad}, N={N0}, L={L0}, solver={solver}")
    print(f"[out]   folder   : {out_dir}")
    print(f"[para]  executor : {Executor.__name__}, workers={workers} (TORCH_THREADS per worker = {TORCH_THREADS})")

    outputs = []
    with Executor(max_workers=workers) as ex:
        futs = [
            ex.submit(
                export_one_volume, m,
                key, N_vol, L0, eps, solver, mu_l0,
                top_vecs, top_ell, K_use, out_dir
            )
            for m in mats
        ]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="[export] coeffs"):
            outputs.append(fut.result())

    print(f"[done] wrote {len(outputs)} NPZs to {out_dir}")
    if outputs:
        print("       example:", os.path.basename(outputs[0]))

if __name__ == "__main__":
    main()
