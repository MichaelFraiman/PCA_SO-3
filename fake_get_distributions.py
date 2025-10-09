#!/usr/bin/env python3
"""
build_coeff_distributions_from_saved_coeffs.py

Reads *per-volume* eigen-coefficient NPZs (already saved elsewhere) and fits
independent complex-Gaussian marginals per coordinate.

It auto-locates the top-modes pack produced by:
  final_cov_matr_centered_global_only.py
and then looks for coefficient NPZs under:
  mat_converted_N=<NN>_eigen_coeffs_top<K_use>/<top_modes_basename>/*_N=<N>_eigen_coeffs_top<K_use>.npz

Each per-volume NPZ is expected to contain:
  - alpha_padded : (K_use, M_max) complex
  - ell          : (K_use,) int32   (ell per mode)
  - m_counts     : (K_use,) int32   (# of m for each mode; typically 2*ell+1)

Output:
  OUT_DIR/marginals_from_saved_coeffs_<top_modes_basename>_N=<N>_L=<L>_Kused=<K_use>.npz
    mu:    (M,2) float32  -> [E[Re], E[Im]]
    Sigma: (M,2,2) float32 -> [[Var(Re)+JITTER, Cov(Re,Im)],
                                [Cov(Re,Im),   Var(Im)+JITTER]]
    offsets, sizes, ell_per_mode, and metadata.
"""

import os
import re
import glob
import numpy as np

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **k): return x

# ----------------------- CONFIG -----------------------
NN         = 64        # resolution N
L_DEFAULT  = 20
K_TOP_WANT = 200        # use up to this many top modes (must match the saved coeff NPZs)
BASE_DIR   = f"mat_converted_N={NN}_matrix"        # where top_modes_*.npz lives
OUT_DIR    = f"mat_converted_N={NN}_coeffs_distributions"
OUT_STEM   = "marginals_from_saved_coeffs"

DDOF       = 1          # sample covariance denominator (1 => unbiased)
JITTER     = 1e-9       # diagonal jitter for Σ
# ------------------------------------------------------


def find_top_modes_file(base_dir: str, N: int, L: int) -> str:
    pat = os.path.join(base_dir, f"top_modes_N={N}_L={L}_K=*_*centered_global_by_raw.npz")
    cands = glob.glob(pat)
    if not cands:
        # fallback in case of slightly different naming
        pat = os.path.join(base_dir, f"top_modes_N={N}_L={L}_K=*centered_global_by_raw.npz")
        cands = glob.glob(pat)
    if not cands:
        raise SystemExit(f"[error] No top-modes NPZ found in {base_dir} for N={N}, L={L}")
    def parse_K(p):
        m = re.search(r"_K=(\d+)", os.path.basename(p))
        return int(m.group(1)) if m else -1
    cands.sort(key=parse_K)
    return cands[-1]


def ensure_dir(p): os.makedirs(p, exist_ok=True); return p


def main():
    ensure_dir(OUT_DIR)

    # 1) Locate and load the top-modes pack for metadata & naming
    top_npz_path = find_top_modes_file(BASE_DIR, N=NN, L=L_DEFAULT)
    top_base = os.path.splitext(os.path.basename(top_npz_path))[0]
    dtop = np.load(top_npz_path, allow_pickle=False)

    N0      = int(dtop["N"])
    L0      = int(dtop["L"])
    K_avail = int(dtop["K"])
    # We’ll *use* at most K_TOP_WANT (and must match what your coeff NPZs contain)
    K_use   = min(K_TOP_WANT, K_avail)

    # 2) Per-volume coefficient files location
    #    Example: mat_converted_N=256_eigen_coeffs_top200/top_modes_N=256_L=20_K=1000_centered_global_by_raw/*.npz
    ROOT_DIR = f"mat_converted_N={NN}_eigen_coeffs_top{K_use}"
    SRC_DIR  = os.path.join(ROOT_DIR, top_base)
    pattern  = os.path.join(SRC_DIR, f"*_N={N0}_eigen_coeffs_top{K_use}.npz")
    files    = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"[error] No per-volume coeff NPZs found under {SRC_DIR} (pattern: {os.path.basename(pattern)})")

    # 3) Inspect the first file to determine layout
    first = np.load(files[0], allow_pickle=False)
    if "alpha_padded" not in first or "ell" not in first or "m_counts" not in first:
        raise SystemExit("[error] Coeff NPZ missing required arrays: alpha_padded, ell, m_counts")

    ell      = first["ell"].astype(np.int32)         # (K_file,)
    m_counts = first["m_counts"].astype(np.int32)    # (K_file,)
    K_file   = int(len(ell))
    if K_file != K_use:
        print(f"[warn] Files carry K={K_file} but K_use={K_use}; proceeding with K_use={min(K_file, K_use)}")
        K_use = min(K_file, K_use)
        ell      = ell[:K_use]
        m_counts = m_counts[:K_use]

    sizes   = m_counts.copy()                        # # of m for each mode r
    offsets = np.zeros(K_use + 1, dtype=np.int32)
    offsets[1:] = np.cumsum(sizes)
    M = int(offsets[-1])

    print(f"[scan] Found {len(files)} volumes in {SRC_DIR}")
    print(f"[layout] N={N0}, L={L0}, K_use={K_use}, total coords M={M}")

    # 4) Streaming accumulators
    sum_r  = np.zeros(M, dtype=np.float64)
    sum_i  = np.zeros(M, dtype=np.float64)
    sum_rr = np.zeros(M, dtype=np.float64)
    sum_ii = np.zeros(M, dtype=np.float64)
    sum_ri = np.zeros(M, dtype=np.float64)

    for path in tqdm(files, desc="[accum] per-volume coeffs"):
        d = np.load(path, allow_pickle=False)
        A = d["alpha_padded"]          # (K_use, M_max) complex
        if A.shape[0] < K_use:
            raise SystemExit(f"[error] {os.path.basename(path)} has fewer modes ({A.shape[0]}) than expected ({K_use})")

        # Flatten (mode r, all its m) into global coordinate vector (length M)
        for r in range(K_use):
            cnt = int(sizes[r])
            if cnt == 0: continue
            s, e = int(offsets[r]), int(offsets[r+1])
            seg  = A[r, :cnt]                   # complex values for m = -ℓ..ℓ (or whatever ordering)
            sr = seg.real.astype(np.float64, copy=False)
            si = seg.imag.astype(np.float64, copy=False)
            sum_r[s:e]  += sr
            sum_i[s:e]  += si
            sum_rr[s:e] += sr * sr
            sum_ii[s:e] += si * si
            sum_ri[s:e] += sr * si

    V = len(files)
    print(f"[stats] Aggregated V={V} volumes")

    # 5) Means and (co)variances
    mu_r = sum_r / V
    mu_i = sum_i / V

    denom = (V - DDOF) if DDOF in (0, 1) else max(1, V - DDOF)
    var_rr = (sum_rr - V * mu_r * mu_r) / denom
    var_ii = (sum_ii - V * mu_i * mu_i) / denom
    cov_ri = (sum_ri - V * mu_r * mu_i) / denom

    mu = np.stack([mu_r, mu_i], axis=1).astype(np.float32)   # (M,2)
    Sigma = np.empty((M, 2, 2), dtype=np.float32)
    Sigma[:, 0, 0] = (var_rr + JITTER).astype(np.float32)
    Sigma[:, 1, 1] = (var_ii + JITTER).astype(np.float32)
    Sigma[:, 0, 1] = Sigma[:, 1, 0] = (cov_ri).astype(np.float32)

    # 6) Save
    out_name = f"{OUT_STEM}_{top_base}_N={N0}_L={L0}_Kused={K_use}"
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"{out_name}.npz")
    np.savez_compressed(
        out_path,
        mu=mu, Sigma=Sigma,
        offsets=offsets, sizes=sizes, ell_per_mode=ell,
        N=np.int32(N0), L=np.int32(L0), K=np.int32(K_use),
        m_ordering=np.array("neg_to_pos"),   # or whatever ordering your exporter used
        source_coeff_dir=np.array(SRC_DIR),
        source_top_modes=np.array(os.path.basename(top_npz_path)),
        file_count=np.int32(V),
        ddof=np.int32(DDOF), jitter=np.float32(JITTER)
    )
    print(f"[save] wrote distributions → {out_path}")


if __name__ == "__main__":
    main()
