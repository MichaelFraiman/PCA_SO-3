#!/usr/bin/env python3
# sample_and_reconstruct_random_volume_centered_global.py
#
# Load per-coordinate marginals (μ, Σ), sample one coefficient vector (in the
# centered-global eigenbasis), ADD BACK the global mean μ_{ℓ=0} to b[:,0,0],
# and reconstruct synthetic volumes using 5, 20, 100, and all modes.

import os
import re
import time
import glob
import numpy as np
import mrcfile

# =================== CONFIG ===================
NN       = 64
L        = 20
K_TOP    = 200                   # we’ll use min(K in dists, K_TOP, K in modes)
# If you prefer to hardcode a modes filename, set MODES_NPZ below.
NPZ_BASE = f"top_modes_N={NN}_L={L}_K=1000_centered_global_by_raw"   # adjust if needed

# Distribution file (from build_coeff_distributions_from_saved_coeffs.py or equivalent)
DIST_NPZ  = f"mat_converted_N={NN}_coeffs_distributions/marginals_from_saved_coeffs_{NPZ_BASE}_N={NN}_L={L}_Kused={K_TOP}.npz"
# Modes file (from final_cov_matr_centered_global_only.py)
MODES_NPZ = f"mat_converted_N={NN}_matrix/{NPZ_BASE}.npz"  # if missing, we auto-find

# How many synthetic draws of alpha to generate (each draw yields 4 reconstructions)
NUM_SAMPLES = 30

# Global amplitude scaling (usually not present in the simple marginals file)
USE_SCALE = False  # most distribution files won’t include scale; keep False unless you added it

# Reconstruct with these cumulative K cutoffs (each <= K_use)
#K_LEVELS = [5, 20, 100, "all"]
K_LEVELS = ["all"]

# Output dir
OUT_DIR = f"mat_converted_N={NN}_synthetic_from_coeffs/{NPZ_BASE}_top{K_TOP}"
# ==============================================

# ---- FLE ----
from fle_3d import FLEBasis3D


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p


def build_fle(N, L, eps, solver="nvidia_torch"):
    # Your FLE uses CPU by default unless explicitly configured otherwise
    return FLEBasis3D(
        N=N, bandlimit=L, eps=eps, max_l=L,
        mode="complex", sph_harm_solver=solver, reduce_memory=True
    )


def make_b_template(fle):
    z = fle.step1(np.zeros((fle.N, fle.N, fle.N), dtype=np.float32))
    b = fle.step2(z)
    return b * 0  # correct shape


def synthesize_volume_from_b(fle, B, b):
    if hasattr(fle, "synthesize"):
        f = fle.synthesize(b)
        return np.real(f).astype(np.float32, copy=False)
    if B is None:
        raise RuntimeError("Need dense B if fle.synthesize() is unavailable.")
    a = fle.step3(b)
    f = B.dot(a).reshape(fle.N, fle.N, fle.N)
    return np.real(f).astype(np.float32, copy=False)


def find_top_modes_file(base_dir: str, N: int, L: int) -> str:
    """
    Pick the largest-K centered-global top-modes file, e.g.:
      top_modes_N=124_L=20_K=1000_centered_global_by_raw.npz
    """
    pat1 = os.path.join(base_dir, f"top_modes_N={N}_L={L}_K=*_*centered_global_by_raw.npz")
    cands = glob.glob(pat1)
    if not cands:
        pat2 = os.path.join(base_dir, f"top_modes_N={N}_L={L}_K=*centered_global_by_raw.npz")
        cands = glob.glob(pat2)
    if not cands:
        raise SystemExit(f"[error] No centered-global top-modes found in {base_dir} for N={N}, L={L}")
    def parse_K(p):
        m = re.search(r"_K=(\d+)", os.path.basename(p))
        return int(m.group(1)) if m else -1
    cands.sort(key=parse_K)
    return cands[-1]


def main():
    ensure_dir(OUT_DIR)

    # ---- Load distributions ----
    D = np.load(DIST_NPZ, allow_pickle=False)
    mu      = D["mu"].astype(np.float64)            # (M,2)
    Sigma   = D["Sigma"].astype(np.float64)         # (M,2,2)
    offsets = D["offsets"]; sizes = D["sizes"]; ell_modes = D["ell_per_mode"]
    Nd      = int(D["N"]); Ld = int(D["L"]); Kd = int(D["K"])
    m_ordering = str(D["m_ordering"]) if "m_ordering" in D else "neg_to_pos"
    jitter  = float(D["jitter"]) if "jitter" in D else 1e-9

    have_scale      = bool(D["have_scale"]) if "have_scale" in D else False
    mu_log_scale    = float(D["mu_log_scale"]) if "mu_log_scale" in D else 0.0
    sigma_log_scale = float(D["sigma_log_scale"]) if "sigma_log_scale" in D else 0.0

    M_total = mu.shape[0]
    assert M_total == int(offsets[-1]), "Mismatch: mu.shape[0] vs offsets[-1]"
    print(f"[load] dists: M={M_total}, K_in_dists={Kd}, N={Nd}, L={Ld}, m_ordering={m_ordering}")

    # ---- Load eigen modes (centered_global) ----
    modes_path = MODES_NPZ
    if not os.path.exists(modes_path):
        modes_path = find_top_modes_file(base_dir=f"mat_converted_N={NN}_matrix", N=NN, L=L)

    MZ = np.load(modes_path, allow_pickle=False, mmap_mode="r")
    top_vecs = MZ["top_vecs"]                      # (K_avail, n_rad)
    top_ell  = MZ["top_ell"].astype(int)
    Nm       = int(MZ["N"]); Lm = int(MZ["L"])
    eps      = float(MZ["eps"])
    solver   = str(MZ["solver"]) if "solver" in MZ else "nvidia_torch"
    if "mu_l0" not in MZ:
        raise RuntimeError("Modes NPZ lacks mu_l0; need the centered_global pack that stores it.")
    mu_l0    = MZ["mu_l0"].astype(np.complex128)   # (n_rad,)

    if (Nm != Nd) or (Lm != Ld):
        raise RuntimeError(f"Basis mismatch: dists(N={Nd},L={Ld}) vs modes(N={Nm},L={Lm})")

    # Harmonize K
    K_use = min(Kd, K_TOP, top_vecs.shape[0])
    u_modes = top_vecs[:K_use].astype(np.complex128, copy=False)   # (K_use, n_rad)
    if not np.all(top_ell[:K_use] == ell_modes[:K_use]):
        print("[warn] ell_per_mode differs from modes file for some entries in the first K; "
              "continuing under assumption of compatible ordering.")

    # Sanitize K list
    K_numeric = []
    for k in K_LEVELS:
        if k == "all":
            K_numeric.append(K_use)
        else:
            K_numeric.append(int(min(int(k), K_use)))
    # remove duplicates and sort increasing (for cumulative builds)
    K_numeric = sorted(set(K_numeric))

    # ---- Build FLE and dense B (if synthesize() not available) ----
    fle = build_fle(Nd, Ld, eps, solver=solver)
    try:
        B = fle.create_denseB(numthread=1)
    except Exception:
        B = None

    # ---- Precompute 2x2 Cholesky for each coord ----
    SigmaJ = Sigma.copy()
    SigmaJ[:, 0, 0] += jitter
    SigmaJ[:, 1, 1] += jitter

    Lchol = np.zeros_like(SigmaJ)
    a = np.sqrt(np.maximum(SigmaJ[:, 0, 0], 0.0))
    Lchol[:, 0, 0] = a
    # Guard against divide-by-zero
    safe_a = np.where(a > 0, a, 1.0)
    Lchol[:, 1, 0] = SigmaJ[:, 1, 0] / safe_a
    Lchol[:, 1, 1] = np.sqrt(np.maximum(SigmaJ[:, 1, 1] - (Lchol[:, 1, 0] ** 2), 0.0))

    rng = np.random.default_rng()

    print(f"[gen] solver={solver}, device={getattr(fle, 'device', 'cpu')}, K_use={K_use}, samples={NUM_SAMPLES}")
    for sidx in range(NUM_SAMPLES):
        # 1) Sample per-coordinate jointly (Re, Im) in the centered eigenbasis
        Z = rng.standard_normal((M_total, 2))                       # (M,2)
        Y = mu + np.einsum('mij,mj->mi', Lchol, Z, optimize=True)   # (M,2)
        alpha = (Y[:, 0] + 1j * Y[:, 1]).astype(np.complex128, copy=False)

        # Optional global scale applied to the *centered* coefficients only (not to μ_l0)
        scale = 1.0
        if USE_SCALE and have_scale and sigma_log_scale >= 0:
            scale = float(np.exp(rng.normal(mu_log_scale, sigma_log_scale)))

        # 2) Build cumulative b’s for each K cutoff from the SAME sampled alpha
        #    We accumulate contributions incrementally to avoid recomputation.
        b_cum = make_b_template(fle)
        next_r_start = 0

        # helper: add modes [next_r_start : r_end) into b_cum
        def add_modes_into_b(b_arr, r_start, r_end):
            for r in range(r_start, r_end):
                ell = int(ell_modes[r])
                s, e = int(offsets[r]), int(offsets[r+1])   # length = 2ℓ+1
                alpha_rm = alpha[s:e] * scale               # (2ℓ+1,)
                u_r = u_modes[r]                            # (n_rad,)
                # accumulate across m
                for m_idx in range(2*ell + 1):
                    b_arr[:, ell, m_idx] += alpha_rm[m_idx] * u_r

        ts = time.strftime("%Y%m%d_%H%M%S")
        for Kcut in K_numeric:
            # Add the next block of modes
            add_modes_into_b(b_cum, next_r_start, Kcut)
            next_r_start = Kcut

            # >>> ADD BACK THE GLOBAL MEAN for (ℓ=0, m=0) <<<
            b = b_cum.copy()
            b[:, 0, 0] = b[:, 0, 0] + mu_l0  # mean is *not* scaled

            # 3) Synthesize and save
            vol = synthesize_volume_from_b(fle, B, b)

            ensure_dir(OUT_DIR)
            out_mrc = os.path.join(
                OUT_DIR,
                f"synthetic_N={Nd}_L={Ld}_K={Kcut}_{ts}_{sidx+1:03d}.mrc"
            )
            with mrcfile.new(out_mrc, overwrite=True) as mrcf:
                mrcf.set_data(vol.astype(np.float32, copy=False))
            print(f"[save] {out_mrc}")

if __name__ == "__main__":
    main()
