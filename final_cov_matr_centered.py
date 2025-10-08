#!/usr/bin/env python3
"""
final_cov_matr_centered.py

Adds:
  - Top-level N controlling default I/O folders.
  - Prints the TOP-30 eigenpairs globally (by uncentered raw eigenvalues).
  - Only compares centered vs uncentered for ℓ=0 (others assumed identical).
  - While printing eigenvalues/vectors: drop imag part if |Im| < 1e-2.
  - All other printed floats use 2 decimal digits.
  - NEW: compute THREE variants:
        • uncentered
        • centered_global   (subtract μμ*, preserves between-means B)
        • centered_pervol   (per-volume centering ⇒ ℓ=0 block = 0)
  - NEW: For each volume, export cumulative energy CSVs for ALL THREE variants:
        <basename>_coeff_energy_uncentered.csv
        <basename>_coeff_energy_centered_global.csv
        <basename>_coeff_energy_centered_pervol.csv
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
N = 32
DEFAULT_IN_DIR  = f"mat_converted_N={N}"
DEFAULT_OUT_DIR = f"mat_converted_N={N}_matrix"
IMAG_TOL = 1e-2      # drop imag if |Im| < 0.01 when printing eigenvalues/vectors
TOP_SHOW = 30        # how many global leading eigenpairs to print

# CSV export controls
EXPORT_CSV = True           # set False to skip CSV export step
EXPORT_N_JOBS = -1          # -1: all cores; if using GPU solver, consider 1
# ========================================================

APPLY_CENTERING_IN_EXPORT = True


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
      b0:  (n_rad,) ℓ=0, m=0 vector for this volume (for centering)
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
    blocks: (L, n_rad, n_rad) Hermitian PSD-ish
    returns:
      eigvals: (L, n_rad)          descending
      eigvecs: (L, n_rad, n_rad)   columns are eigenvectors (U from SVD)
    """
    L, n_rad, _ = blocks.shape
    eigvals = np.zeros((L, n_rad), np.float64)
    eigvecs = np.zeros((L, n_rad, n_rad), np.complex128)
    rng = range(L)
    if pbar:
        rng = tqdm(rng, desc="SVD per ℓ", leave=False)
    for ell in rng:
        Bl = 0.5 * (blocks[ell] + blocks[ell].conj().T)  # guard symmetry
        U, S, _ = np.linalg.svd(Bl, full_matrices=False)
        eigvecs[ell] = U
        eigvals[ell] = S
    return eigvals, eigvecs

def mp_spike_shrink_block(lam, gamma):
    """Unit-noise spiked-MP inversion: return β̂ per eigenvalue (zeros in bulk)."""
    gp = (1.0 + np.sqrt(gamma))**2
    out = np.zeros_like(lam)
    mask = lam > gp
    if np.any(mask):
        x = lam[mask] - 1.0 - gamma
        disc = x*x - 4.0*gamma
        disc = np.maximum(disc, 0.0)
        out[mask] = 0.5 * (x + np.sqrt(disc))
    return out

# ----------------------- printing helpers -----------------------

def phase_align(u_ref: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Align u to u_ref by global phase so entries are directly comparable."""
    denom = np.vdot(u_ref, u)
    if np.abs(denom) < 1e-15:
        return u
    return u * np.exp(-1j * np.angle(denom))

def fmt_float(x: float) -> str:
    """Two decimals for 'everything else'."""
    return f"{x:.2f}"

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

def vector_metrics(u_unc: np.ndarray, u_ctr: np.ndarray):
    """
    Return (rho, angle_deg, l2_gap, linf_gap, u_ctr_aligned).
    Uses phase alignment and unit-norm versions of both vectors.
    """
    # normalize
    nu = np.linalg.norm(u_unc); nv = np.linalg.norm(u_ctr)
    if nu == 0 or nv == 0:
        return 0.0, 90.0, 0.0, 0.0, u_ctr
    u1 = u_unc / nu
    u2 = u_ctr / nv
    # phase-align u2 to u1
    ph = np.vdot(u1, u2)
    if np.abs(ph) > 1e-15:
        u2a = u2 * np.exp(-1j * np.angle(ph))
    else:
        u2a = u2
    # similarity & angles
    rho = float(np.clip(np.abs(np.vdot(u1, u2a)), 0.0, 1.0))
    angle_deg = float(np.degrees(np.arccos(rho)))
    # gaps
    diff = u1 - u2a
    l2_gap  = float(np.linalg.norm(diff))
    linf_gap = float(np.max(np.abs(diff)))
    return rho, angle_deg, l2_gap, linf_gap, u2a


def print_l0_summary(vec_unc_l0: np.ndarray, vec_ctr_l0: np.ndarray, label: str):
    """
    Summarize differences across *all* ℓ=0 modes (all radial columns) for a given centered variant.
    """
    n_rad = vec_unc_l0.shape[1]
    angles, l2s = [], []
    for r in range(n_rad):
        rho, ang, l2, linf, _ = vector_metrics(vec_unc_l0[:, r], vec_ctr_l0[:, r])
        angles.append(ang); l2s.append(l2)
    angles = np.array(angles); l2s = np.array(l2s)
    print(f"\n[ℓ=0 summary vs uncentered] variant={label}")
    print(f"  mean angle: {fmt_float(np.mean(angles))}°,  95%: {fmt_float(np.percentile(angles,95))}°,  max: {fmt_float(np.max(angles))}°")
    print(f"  mean L2 gap: {fmt_float(np.mean(l2s))},      max: {fmt_float(np.max(l2s))}")
    print(f"  #angles > 10°: {(angles>10).sum()} / {n_rad},  > 30°: {(angles>30).sum()} / {n_rad}")


def print_top30_global_dual(raw_unc, vec_unc, raw_ctr, vec_ctr, n_rad: int, label: str, top_k_show: int = TOP_SHOW):
    """
    Prints TOP-K eigenpairs globally, ordered by uncentered raw eigenvalues.
    For ℓ=0, also prints 'label' centered counterpart + similarity metrics.
    """
    flat = raw_unc.flatten()
    order = np.argsort(flat)[::-1]
    total = raw_unc.size
    K = min(top_k_show, total)
    print(f"\n[top-30] Top {K} eigenpairs globally (by uncentered raw λ); "
          f"for ℓ=0 also show '{label}' and differences. (|Im|<0.01 → real)")
    for rank, fi in enumerate(order[:K], start=1):
        ell = fi // n_rad
        rad = fi %  n_rad
        lam_unc = raw_unc[ell, rad]
        u_unc = vec_unc[ell][:, rad]

        if ell == 0:
            lam_ctr = raw_ctr[ell, rad]
            u_ctr = vec_ctr[ell][:, rad]
            # metrics + aligned vector
            rho, ang_deg, l2_gap, linf_gap, u_ctr_al = vector_metrics(u_unc, u_ctr)
            print(f"  #{rank:02d}  (ℓ={ell}, r={rad})  λ_unc={fmt_number(lam_unc)}  λ_{label}={fmt_number(lam_ctr)}  "
                  f"|<u,u’>|={fmt_float(rho)}  θ={fmt_float(ang_deg)}°  L2={fmt_float(l2_gap)}  L∞={fmt_float(linf_gap)}")
            print("       u_unc*: ", vec_to_string(u_unc))
            print("       u_{:s}*: ".format(label), vec_to_string(u_ctr_al))
        else:
            print(f"  #{rank:02d}  (ℓ={ell}, r={rad})  λ_unc={fmt_number(lam_unc)}")
            print("       u_unc*: ", vec_to_string(u_unc))

# ----------------------- per-volume CSV export -----------------------

def export_coeff_energy_for_volume_variants(
    m_path: str,
    key: str,
    N_vol: int,
    L: int,
    eps: float,
    solver: str,
    n_rad: int,
    variants: List[Tuple[str, np.ndarray, np.ndarray]],
    out_dir: str,
    mu_l0: Optional[np.ndarray] = None,
    apply_centering_in_export: bool = True,
):
    """
    For one volume:
      - build FLE coefficients b = step2(step1(vol))  with shape (n_rad, L, 2ℓ+1)
      - for EACH variant (tag, eigvecs, order):
          * optionally center ℓ=0 coefficients in b according to the variant
          * project onto the eigenbasis: A_ell = U_ell^H @ B_ell
          * accumulate per-(ℓ,rad) energies: sum_m |A_ell|^2
          * flatten, reorder by the variant's global order, take cumulative ratio
          * save CSV: <basename>_coeff_energy_<tag>.csv with columns: k,ratio

    Args:
        variants: list of tuples (tag, eigvecs, order), where
                  tag ∈ {"uncentered", "centered_global", "centered_pervol"},
                  eigvecs has shape (L, n_rad, n_rad) with columns = eigenvectors,
                  order is a flattened index order (np.ndarray) for that variant.
        mu_l0:    (n_rad,) global rotational mean for ℓ=0 (needed for centered_global).
        apply_centering_in_export:
                  if True, adjust only ℓ=0,m=0 coefficients of this volume before projection:
                      - centered_pervol: b[:,0,0] = 0
                      - centered_global: b[:,0,0] -= mu_l0
                  if False, project raw coefficients (useful for diagnostics).

    Returns:
        True on success (paths are not returned since callers usually ignore them).
    """
    # Build FLE for this worker
    fle = FLEBasis3D(
        N=N_vol, bandlimit=L, eps=eps, max_l=L,
        mode="complex", sph_harm_solver=solver, reduce_memory=True
    )

    # Load and normalize the volume, then get coefficients
    v = load_and_normalize(m_path, key)
    z = fle.step1(v)
    b = fle.step2(z)  # shape: (n_rad, L, 2ℓ+1)

    basename = os.path.splitext(os.path.basename(m_path))[0]
    os.makedirs(out_dir, exist_ok=True)

    for tag, eigvecs, order in variants:
        # Optionally adjust ℓ=0 according to the centering model we want to reflect
        if apply_centering_in_export and tag != "uncentered":
            b_use = b.copy()
            if tag == "centered_pervol":
                b_use[:, 0, 0] = 0.0
            elif tag == "centered_global":
                if mu_l0 is None:
                    raise ValueError("mu_l0 must be provided for centered_global export.")
                b_use[:, 0, 0] = b_use[:, 0, 0] - mu_l0
        else:
            b_use = b  # use raw coefficients

        # Per-(ℓ,rad) energies
        energies = np.zeros((L, n_rad), dtype=np.float64)
        for ell in range(L):
            B_ell = b_use[:, ell, :2*ell + 1]        # (n_rad, 2ℓ+1)
            U_ell = eigvecs[ell]                     # (n_rad, n_rad), columns = eigenvectors
            A = U_ell.conj().T @ B_ell               # (n_rad, 2ℓ+1): eigenbasis coords
            energies[ell] = np.sum(np.abs(A)**2, axis=1)

        # Flatten, reorder by this variant's global order, cumulative ratio
        ef = energies.flatten()
        ef = ef[order]
        total = float(ef.sum())
        ratios = (np.cumsum(ef) / total) if total > 0 else np.zeros_like(ef)

        # Save CSV
        k = np.arange(1, ratios.size + 1, dtype=int)
        table = np.column_stack((k, ratios))
        csv_path = os.path.join(out_dir, f"{basename}_coeff_energy_{tag}.csv")
        np.savetxt(csv_path, table, delimiter=",", header="k,ratio", comments="")

    return True



# additional stuff

def compare_centered_bases(vec_cg: np.ndarray,
                           vec_cw: np.ndarray,
                           raw_cg: np.ndarray,
                           raw_cw: np.ndarray,
                           angle_hi: float = 30.0,
                           angle_med: float = 10.0):
    """
    Compare eigenvectors between centered_global and centered_pervol.
    Skips ℓ=0 (per-volume centering makes S_0 = 0 → eigenvectors not meaningful).
    Prints summary stats of alignment per ℓ.

    vec_cg, vec_cw: shapes (L, n_rad, n_rad), columns are eigenvectors.
    raw_cg, raw_cw: shapes (L, n_rad), raw eigenvalues (for sanity checks).
    """
    L, n_rad, _ = vec_cg.shape
    print("\n[compare] Eigenvectors: centered_global vs centered_pervol")
    print("  Note: ℓ=0 skipped (per-volume centering makes S_0 ≡ 0, eigenvectors undefined).")

    total_pairs = 0
    angles_all = []
    l2_all = []

    for ell in range(1, L):
        ell_angles = []
        ell_l2 = []
        for r in range(n_rad):
            u = vec_cg[ell][:, r]
            v = vec_cw[ell][:, r]
            rho, ang_deg, l2_gap, linf_gap, _ = vector_metrics(u, v)
            ell_angles.append(ang_deg)
            ell_l2.append(l2_gap)
        ell_angles = np.array(ell_angles)
        ell_l2 = np.array(ell_l2)
        total_pairs += n_rad
        angles_all.append(ell_angles)
        l2_all.append(ell_l2)

        print(f"  [ℓ={ell:2d}]  mean θ={fmt_float(ell_angles.mean())}°  "
              f"95%={fmt_float(np.percentile(ell_angles,95))}°  "
              f"max={fmt_float(ell_angles.max())}°  "
              f"> {angle_med:.0f}°: {(ell_angles>angle_med).sum():2d}/{n_rad}  "
              f"> {angle_hi:.0f}°: {(ell_angles>angle_hi).sum():2d}/{n_rad}")

    if angles_all:
        angles_all = np.concatenate(angles_all)
        l2_all = np.concatenate(l2_all)
        print("\n  [overall ℓ≥1]")
        print(f"    mean θ={fmt_float(angles_all.mean())}°  "
              f"95%={fmt_float(np.percentile(angles_all,95))}°  "
              f"max={fmt_float(angles_all.max())}°")
        print(f"    mean L2={fmt_float(l2_all.mean())}  max L2={fmt_float(l2_all.max())}")

def print_top30_tables(raw_unc: np.ndarray,
                       ord_unc_raw: np.ndarray,
                       raw_cg: np.ndarray,
                       ord_cg_raw: np.ndarray,
                       raw_cw: np.ndarray,
                       ord_cw_raw: np.ndarray,
                       n_rad: int,
                       top_k_show: int = 30):
    """
    Print three separate tables (uncentered, centered_global, centered_pervol),
    each listing the top-30 eigenpairs by that variant's own raw eigenvalues.
    For each entry: rank, ℓ, radial index r, eigenvalue.
    """
    def table_for(label, raw, order, n_rad, k):
        flatN = raw.size
        K = min(k, flatN)
        print(f"\n[table:{label}] Top {K} eigenvalues (by {label} raw λ)")
        print("  rank  ℓ   r        λ")
        for rank, fi in enumerate(order[:K], start=1):
            ell = fi // n_rad
            rad = fi %  n_rad
            lam = raw[ell, rad]
            print(f"  {rank:>4d}  {ell:>2d}  {rad:>2d}  {fmt_number(lam)}")

    table_for("uncentered",      raw_unc, ord_unc_raw, n_rad, top_k_show)
    table_for("centered_global", raw_cg,  ord_cg_raw,  n_rad, top_k_show)
    table_for("centered_pervol", raw_cw,  ord_cw_raw,  n_rad, top_k_show)


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
    expect_N=N,                    # for sanity check against data
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
    p = n_rad

    # per-ℓ normalization by (2ℓ+1)
    S_unc = np.empty_like(avg_over_v)
    for ell in range(L):
        Bl = avg_over_v[ell] / float(2*ell + 1)
        S_unc[ell] = 0.5 * (Bl + Bl.conj().T)

    # mean (DC) vector across volumes for ℓ=0
    mu_l0 = (sum_l0 / M).astype(np.complex128)        # (n_rad,)

    # ---- build BOTH centered variants
    # 1) Global centering: subtract μμ* → keeps B
    S_ctr_global = S_unc.copy()
    S_ctr_global[0] = S_ctr_global[0] - np.outer(mu_l0, mu_l0.conj())
    S_ctr_global[0] = 0.5 * (S_ctr_global[0] + S_ctr_global[0].conj().T)

    # 2) Per-volume centering (within): zero ℓ=0 block
    S_ctr_pervol = S_unc.copy()
    S_ctr_pervol[0] = np.zeros_like(S_ctr_pervol[0])  # Σ_within has no ℓ=0 power

    # ---- save the covariances
    cov_meta = dict(N=N_vol, L=L, eps=eps, solver=solver, n_rad=n_rad, M=M)
    np.savez_compressed(
        os.path.join(out_dir, f"cov_blocks_N={N_vol}_L={L}_uncentered.npz"),
        S_blocks=S_unc, mu_l0=mu_l0, **cov_meta
    )
    np.savez_compressed(
        os.path.join(out_dir, f"cov_blocks_N={N_vol}_L={L}_centered_global.npz"),
        S_blocks=S_ctr_global, mu_l0=mu_l0, **cov_meta
    )
    np.savez_compressed(
        os.path.join(out_dir, f"cov_blocks_N={N_vol}_L={L}_centered_pervol.npz"),
        S_blocks=S_ctr_pervol, mu_l0=mu_l0, **cov_meta
    )
    print("[save] wrote covariance NPZs (uncentered, centered_global, centered_pervol)")

    # ---- eigendecompositions (all three) with progress bars
    if pbar: print("[step] Eigendecomposition …")
    raw_unc,  vec_unc  = svd_covariance(S_unc, pbar=pbar)
    raw_cg,   vec_cg   = svd_covariance(S_ctr_global, pbar=pbar)
    raw_cw,   vec_cw   = svd_covariance(S_ctr_pervol, pbar=pbar)

    # ---- PRINT: TOP-30 global eigenpairs; compare each centered variant vs uncentered (ℓ=0)
    print_top30_global_dual(raw_unc, vec_unc, raw_cg, vec_cg, n_rad, label="centered_global", top_k_show=TOP_SHOW)
    print_top30_global_dual(raw_unc, vec_unc, raw_cw, vec_cw, n_rad, label="centered_pervol", top_k_show=TOP_SHOW)

    # Compact ℓ=0 diffs for each centered variant
    print_l0_summary(vec_unc[0], vec_cg[0], label="centered_global")
    print_l0_summary(vec_unc[0], vec_cw[0], label="centered_pervol")

    compare_centered_bases(vec_cg, vec_cw, raw_cg, raw_cw)

    # MP shrinkage per ℓ (with bar)
    gamma_by_ell = np.zeros(L, dtype=np.float64)
    shr_unc = np.zeros_like(raw_unc)
    shr_cg  = np.zeros_like(raw_cg)
    shr_cw  = np.zeros_like(raw_cw)
    spikes_unc = np.zeros(L, dtype=np.int32)
    spikes_cg  = np.zeros(L, dtype=np.int32)
    spikes_cw  = np.zeros(L, dtype=np.int32)

    rng = range(L)
    if pbar:
        rng = tqdm(rng, desc="Shrinkage per ℓ", leave=False)
    for ell in rng:
        n_samples = M * (2*ell + 1)
        gamma = p / float(n_samples)
        gamma_by_ell[ell] = gamma
        beta_unc = mp_spike_shrink_block(raw_unc[ell], gamma)
        beta_cg  = mp_spike_shrink_block(raw_cg[ell],  gamma)
        beta_cw  = mp_spike_shrink_block(raw_cw[ell],  gamma)
        shr_unc[ell] = beta_unc
        shr_cg[ell]  = beta_cg
        shr_cw[ell]  = beta_cw
        spikes_unc[ell] = int(np.count_nonzero(beta_unc > 0))
        spikes_cg[ell]  = int(np.count_nonzero(beta_cg  > 0))
        spikes_cw[ell]  = int(np.count_nonzero(beta_cw  > 0))

    # ---- top-K selections (for saving packs)
    total_modes = L * n_rad
    K = min(top_k, total_modes)
    if K < top_k:
        print(f"[warn] Requested top_k={top_k} but only {total_modes} modes exist; using K={K}")

    # orders for saving (and for CSV export)
    ord_unc_raw = np.argsort(raw_unc.flatten())[::-1]
    ord_cg_raw  = np.argsort(raw_cg.flatten())[::-1]
    ord_cw_raw  = np.argsort(raw_cw.flatten())[::-1]

    variants = [
        ("uncentered",      vec_unc, ord_unc_raw),
        ("centered_global", vec_cg,  ord_cg_raw),
        ("centered_pervol", vec_cw,  ord_cw_raw),
    ]

    # gather helper
    def gather_top(K, order, n_rad, eigvecs, raw_eigvals, shr_eigvals):
        top_ell = np.empty(K, dtype=np.int32)
        top_rad = np.empty(K, dtype=np.int32)
        top_vals_raw = np.empty(K, dtype=np.float64)
        top_vals_shr = np.empty(K, dtype=np.float64)
        top_vecs = np.empty((K, n_rad), dtype=np.complex128)
        for i in range(K):
            fi = order[i]
            ell = fi // n_rad
            rad = fi %  n_rad
            top_ell[i] = ell
            top_rad[i] = rad
            top_vals_raw[i] = raw_eigvals[ell, rad]
            top_vals_shr[i] = shr_eigvals[ell, rad]
            top_vecs[i] = eigvecs[ell][:, rad]
        return top_vecs, top_ell, top_rad, top_vals_raw, top_vals_shr

    # compute shrunk orders (not used for CSV)
    ord_unc_shr = np.argsort(shr_unc.flatten())[::-1]
    ord_cg_shr  = np.argsort(shr_cg.flatten())[::-1]
    ord_cw_shr  = np.argsort(shr_cw.flatten())[::-1]

    # gather (uncentered)
    top_vecs_u_r, top_ell_u_r, top_rad_u_r, top_raw_u_r, top_shr_u_r = gather_top(
        K, ord_unc_raw, n_rad, vec_unc, raw_unc, shr_unc
    )
    top_vecs_u_s, top_ell_u_s, top_rad_u_s, top_raw_u_s, top_shr_u_s = gather_top(
        K, ord_unc_shr, n_rad, vec_unc, raw_unc, shr_unc
    )

    # gather (centered_global)
    top_vecs_cg_r, top_ell_cg_r, top_rad_cg_r, top_raw_cg_r, top_shr_cg_r = gather_top(
        K, ord_cg_raw, n_rad, vec_cg, raw_cg, shr_cg
    )
    top_vecs_cg_s, top_ell_cg_s, top_rad_cg_s, top_raw_cg_s, top_shr_cg_s = gather_top(
        K, ord_cg_shr, n_rad, vec_cg, raw_cg, shr_cg
    )

    # gather (centered_pervol)
    top_vecs_cw_r, top_ell_cw_r, top_rad_cw_r, top_raw_cw_r, top_shr_cw_r = gather_top(
        K, ord_cw_raw, n_rad, vec_cw, raw_cw, shr_cw
    )
    top_vecs_cw_s, top_ell_cw_s, top_rad_cw_s, top_raw_cw_s, top_shr_cw_s = gather_top(
        K, ord_cw_shr, n_rad, vec_cw, raw_cw, shr_cw
    )

    if store_as_complex64:
        top_vecs_u_r = top_vecs_u_r.astype(np.complex64)
        top_vecs_u_s = top_vecs_u_s.astype(np.complex64)
        top_vecs_cg_r = top_vecs_cg_r.astype(np.complex64)
        top_vecs_cg_s = top_vecs_cg_s.astype(np.complex64)
        top_vecs_cw_r = top_vecs_cw_r.astype(np.complex64)
        top_vecs_cw_s = top_vecs_cw_s.astype(np.complex64)

    # ---- save top-K packs
    meta_common = dict(
        N=N_vol, L=L, eps=eps, solver=solver,
        n_rad=n_rad, K=K, gamma_by_ell=gamma_by_ell, mu_l0=mu_l0,
        mats_used=np.array([os.path.basename(x) for x in mats])
    )

    def save_top(tag, top_vecs, top_ell, top_rad, top_vals_raw, top_vals_shr, spikes):
        np.savez_compressed(
            os.path.join(out_dir, f"top_modes_N={N_vol}_L={L}_K={K}_{tag}.npz"),
            top_vecs=top_vecs,
            top_ell=top_ell, top_rad=top_rad,
            top_vals_raw=top_vals_raw, top_vals_shr=top_vals_shr,
            spikes_by_ell=spikes,
            **meta_common
        )

    # uncentered
    save_top("uncentered_by_raw",    top_vecs_u_r,  top_ell_u_r,  top_rad_u_r,  top_raw_u_r,  top_shr_u_r,  spikes_unc)
    save_top("uncentered_by_shrunk", top_vecs_u_s,  top_ell_u_s,  top_rad_u_s,  top_raw_u_s,  top_shr_u_s,  spikes_unc)
    # centered_global
    save_top("centered_global_by_raw",    top_vecs_cg_r, top_ell_cg_r, top_rad_cg_r, top_raw_cg_r, top_shr_cg_r, spikes_cg)
    save_top("centered_global_by_shrunk", top_vecs_cg_s, top_ell_cg_s, top_rad_cg_s, top_raw_cg_s, top_shr_cg_s, spikes_cg)
    # centered_pervol
    save_top("centered_pervol_by_raw",    top_vecs_cw_r, top_ell_cw_r, top_rad_cw_r, top_raw_cw_r, top_shr_cw_r, spikes_cw)
    save_top("centered_pervol_by_shrunk", top_vecs_cw_s, top_ell_cw_s, top_rad_cw_s, top_raw_cw_s, top_shr_cw_s, spikes_cw)

    # ---- per-volume CSV export (ALL components), with progress bar
    if EXPORT_CSV:
        export_workers = (os.cpu_count() or 1) if (EXPORT_N_JOBS in (-1, None, 0)) else int(EXPORT_N_JOBS)
        if "nvidia" in str(solver).lower():
            export_workers = 1 if EXPORT_N_JOBS in (-1, None, 0) else export_workers

        print(f"[export] Writing cumulative energy CSVs for {len(mats)} volumes …")
        with ThreadPoolExecutor(max_workers=export_workers) as ex:
            futures = [
                ex.submit(
                    export_coeff_energy_for_volume_variants,
                    m, key, N_vol, L, eps, solver, n_rad,
                    variants=variants,
                    out_dir=out_dir,
                    mu_l0=mu_l0,
                    apply_centering_in_export=APPLY_CENTERING_IN_EXPORT
                )
                for m in mats
            ]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Export CSVs", leave=True):
                fut.result()  # just ensure errors surface


    # (optional) save full eigendecompositions — big!
    if save_full_eigendecomp:
        np.savez_compressed(
            os.path.join(out_dir, f"full_eigs_N={N_vol}_L={L}_uncentered.npz"),
            eigvals=raw_unc, eigvecs=vec_unc, shrunk=shr_unc, **cov_meta, mu_l0=mu_l0
        )
        np.savez_compressed(
            os.path.join(out_dir, f"full_eigs_N={N_vol}_L={L}_centered_global.npz"),
            eigvals=raw_cg, eigvecs=vec_cg, shrunk=shr_cg, **cov_meta, mu_l0=mu_l0
        )
        np.savez_compressed(
            os.path.join(out_dir, f"full_eigs_N={N_vol}_L={L}_centered_pervol.npz"),
            eigvals=raw_cw, eigvecs=vec_cw, shrunk=shr_cw, **cov_meta, mu_l0=mu_l0
        )

    # ---------- quick summary (two decimals for floats) ----------
    print("\n[summary]")
    print(f"  volumes processed: {M}")
    print(f"  total modes per variant: {L*n_rad}  |  K saved: {K}")
    print(f"  spikes per ℓ (uncentered):      {spikes_unc.tolist()}")
    print(f"  spikes per ℓ (centered_global): {spikes_cg.tolist()}")
    print(f"  spikes per ℓ (centered_pervol): {spikes_cw.tolist()}")

    # NEW: end-of-run tables of top-30 eigenvalues for each variant
    print_top30_tables(raw_unc, ord_unc_raw, raw_cg, ord_cg_raw, raw_cw, ord_cw_raw, n_rad, top_k_show=TOP_SHOW)


    print("\n[save] wrote:")
    for fn in (
        f"cov_blocks_N={N_vol}_L={L}_uncentered.npz",
        f"cov_blocks_N={N_vol}_L={L}_centered_global.npz",
        f"cov_blocks_N={N_vol}_L={L}_centered_pervol.npz",
        f"top_modes_N={N_vol}_L={L}_K={K}_uncentered_by_raw.npz",
        f"top_modes_N={N_vol}_L={L}_K={K}_uncentered_by_shrunk.npz",
        f"top_modes_N={N_vol}_L={L}_K={K}_centered_global_by_raw.npz",
        f"top_modes_N={N_vol}_L={L}_K={K}_centered_global_by_shrunk.npz",
        f"top_modes_N={N_vol}_L={L}_K={K}_centered_pervol_by_raw.npz",
        f"top_modes_N={N_vol}_L={L}_K={K}_centered_pervol_by_shrunk.npz",
    ):
        print("  •", os.path.join(out_dir, fn))


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
        save_full_eigendecomp=False,
        pbar=True,               # turn off if you don't want bars in nohup logs
        expect_N=N,              # sanity-check resolution vs data
    )
