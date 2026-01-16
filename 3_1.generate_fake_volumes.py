#!/usr/bin/env python3
"""fake_generate.py

Generate synthetic ("fake") volumes by sampling eigen-coefficients from
per-coordinate complex-Gaussian marginals and reconstructing into voxel space.

Inputs
------
1) Distribution NPZ (output of fake_get_distributions.py):
   - mu      : (M,2) float    per-coordinate mean of (Re, Im)
   - Sigma   : (M,2,2) float  per-coordinate covariance of (Re, Im)
   - offsets : (K+1,) int     start/end offsets into flattened coefficient vector
   - sizes   : (K,) int       number of m-coordinates for each radial mode r
   - ell_per_mode : (K,) int  ell for each radial mode r
   - N, L, K  : int metadata
   - m_ordering  : string describing how m is ordered within each mode segment
                  ("md" or "neg_to_pos")

2) Top-modes NPZ (output of 0_1.covariance_matrix.py):
   - top_vecs : (K_avail, n_rad) complex  radial eigenvectors u_r
   - top_ell  : (K_avail,) int            ell per mode r
   - mu_l0    : (n_rad,) complex          dataset mean in the (ell=0,m=0) radial slice
   - N, L, eps, solver : metadata

Algorithm
---------
For each sample:
  - independently sample each coefficient coordinate (Re,Im) ~ N(mu, Sigma)
  - optionally apply an overall log-normal scale factor (if present in dist NPZ)
  - build b(q, ell, md) via b[:, ell, md] += alpha_{r,m} * u_r(q)
  - add mean: b[:,0,0] += mu_l0
  - convert to Laplacian-eigen coefficients a = step3(b)
  - reconstruct volume f = evaluate(a)
  - save as MRC

Notes
-----
* This script avoids building a dense B matrix. It uses FLEBasis3D.evaluate()
  for reconstruction, which is significantly more memory-friendly.
"""

import os
import sys

# CRITICAL FIX FOR MACOS SEGFAULT:
# Control OpenMP/MKL threading before importing numpy / torch.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import re
import time
import glob
import argparse
from typing import List, Optional, Sequence, Tuple

import numpy as np
import mrcfile

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **k):
        return x

from fle_3d import FLEBasis3D  # noqa: E402


# ----------------------------- m-index helpers -----------------------------
def m_to_md(m: int) -> int:
    """Map m in [-ell,ell] to md index used internally by FLEBasis3D.

    md ordering is: 0, -1, +1, -2, +2, -3, +3, ...
    """
    if m == 0:
        return 0
    if m > 0:
        return 2 * m
    return 2 * (-m) - 1


def _as_py_str(x) -> str:
    """Robust conversion of np scalar / bytes / 0-d array to python str."""
    if isinstance(x, np.ndarray) and x.shape == ():
        x = x.item()
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="replace")
    return str(x)


def _reorder_segment_to_md(seg: np.ndarray, ell: int, m_ordering: str) -> np.ndarray:
    """Return a length-(2*ell+1) array in md-order.

    Parameters
    ----------
    seg:
        Complex coefficient segment for a fixed (ell, r). Length must be 2*ell+1.
    m_ordering:
        One of:
          - "md"           : already md-order
          - "neg_to_pos"   : m = -ell, ..., +ell
    """
    m_ordering = (m_ordering or "").strip().lower()
    m_range = 2 * ell + 1
    if seg.shape[0] != m_range:
        raise ValueError(f"Segment length {seg.shape[0]} does not match 2*ell+1={m_range} (ell={ell}).")

    if m_ordering in ("md", "md_order", "0,-1,+1", "torch_md"):
        return seg
    if m_ordering in ("neg_to_pos", "neg2pos", "-l..l", "m=-l..l", "m=-ell..ell"):
        out = np.empty_like(seg)
        for i, m in enumerate(range(-ell, ell + 1)):
            out[m_to_md(m)] = seg[i]
        return out

    # If unknown, default to md (most consistent with FLE internals) but warn once.
    # We avoid printing inside tight loops; caller prints warning.
    return seg


# ----------------------------- path resolution -----------------------------
def _resolve_newest(globs: Sequence[str]) -> str:
    cands: List[str] = []
    for g in globs:
        cands.extend(glob.glob(g))
    cands = sorted(set(cands))
    if not cands:
        raise RuntimeError("No files matched globs:\n  " + "\n  ".join(globs))
    cands.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return cands[0]


def _resolve_dist_npz_path(args: argparse.Namespace) -> str:
    if args.dist_npz:
        if os.path.exists(args.dist_npz):
            return args.dist_npz
        raise RuntimeError(f"--dist-npz does not exist: {args.dist_npz}")

    dist_dir = args.dist_dir or f"mat_converted_N={args.nn}_coeffs_distributions"
    L_tag = "*" if int(args.L_sel) <= 0 else str(int(args.L_sel))

    globs = []
    if args.dist_glob:
        globs.append(args.dist_glob)
    else:
        globs.append(os.path.join(dist_dir, f"*N={args.nn}_L={L_tag}_Kused=*.npz"))
        if L_tag == "*":
            globs.append(os.path.join(dist_dir, f"*N={args.nn}_Kused=*.npz"))
        globs.append(os.path.join(dist_dir, "*.npz"))

    return _resolve_newest(globs)


def _resolve_modes_npz_path(args: argparse.Namespace, dist_meta: dict) -> str:
    if args.modes_npz:
        if os.path.exists(args.modes_npz):
            return args.modes_npz
        raise RuntimeError(f"--modes-npz does not exist: {args.modes_npz}")

    matrix_dir = args.matrix_dir or f"mat_converted_N={args.nn}_matrix"

    # Try dist metadata first.
    src = dist_meta.get("source_top_modes")
    if isinstance(src, str) and src:
        cand = os.path.join(matrix_dir, src)
        if os.path.exists(cand):
            return cand

    # Fallback: newest matching centered_global pack.
    L_tag = "*" if int(args.L_sel) <= 0 else str(int(args.L_sel))
    eps_tag = "*" if float(args.eps_sel) <= 0.0 else format(float(args.eps_sel), ".0e").replace("+", "")

    globs = []
    if args.modes_glob:
        globs.append(args.modes_glob)
    else:
        globs.append(os.path.join(matrix_dir, f"top_modes_N={args.nn}_L={L_tag}_eps={eps_tag}_*centered_global_by_raw.npz"))
        if eps_tag == "*":
            globs.append(os.path.join(matrix_dir, f"top_modes_N={args.nn}_L={L_tag}_*centered_global_by_raw.npz"))

    return _resolve_newest(globs)


# ----------------------------- I/O helpers ---------------------------------
def _save_mrc(path: str, vol: np.ndarray, voxel_size: float, voxel_dtype: np.dtype, overwrite: bool) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if (not overwrite) and os.path.exists(path):
        raise FileExistsError(path)
    with mrcfile.new(path, overwrite=True) as mrc:
        mrc.set_data(vol.astype(voxel_dtype, copy=False))
        try:
            mrc.voxel_size = (voxel_size, voxel_size, voxel_size)
        except Exception:
            pass


def _normalize_real_volume(vol: np.ndarray) -> np.ndarray:
    mx = float(np.max(np.abs(vol)))
    if mx > 0:
        return (vol / mx).astype(vol.dtype, copy=False)
    return vol


# ----------------------------- main logic ----------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sample eigen-coefficients from fitted marginals and reconstruct synthetic volumes."
    )

    # Inputs
    p.add_argument("--dist-npz", dest="dist_npz", default=None,
                   help="Path to distributions NPZ (mu/Sigma/offsets/...). If omitted, auto-picks newest in --dist-dir.")
    p.add_argument("--dist-dir", dest="dist_dir", default=None,
                   help="Directory to search for distributions if --dist-npz not set. Default: mat_converted_N=<nn>_coeffs_distributions")
    p.add_argument("--dist-glob", dest="dist_glob", default=None,
                   help="Glob pattern to find distributions if --dist-npz not set.")

    p.add_argument("--modes-npz", dest="modes_npz", default=None,
                   help="Path to top_modes_*.npz. If omitted, auto-picks newest in --matrix-dir (or uses dist metadata).")
    p.add_argument("--matrix-dir", dest="matrix_dir", default=None,
                   help="Directory to search for top_modes if --modes-npz not set. Default: mat_converted_N=<nn>_matrix")
    p.add_argument("--modes-glob", dest="modes_glob", default=None,
                   help="Glob pattern to find top_modes if --modes-npz not set.")

    # Selection-only filters for auto-pick
    p.add_argument("--nn", dest="nn", type=int, default=22,
                   help="N used for default dirs and auto-search patterns (default: 22).")
    p.add_argument("--L", dest="L_sel", type=int, default=20,
                   help="Filter auto-picked NPZs by bandlimit L. Use --L 0 to disable filtering (default: 20).")
    p.add_argument("--eps", dest="eps_sel", type=float, default=0.0,
                   help="Filter auto-picked top_modes NPZ by eps. Use --eps 0 to disable filtering (default: 0).")

    # Sampling / reconstruction
    p.add_argument("--k-use", dest="k_use", type=int, default=0,
                   help="Use at most this many radial modes K. 0 => use K from dist file (default: 0).")
    p.add_argument("--k-levels", dest="k_levels", default="all",
                   help="Comma-separated cumulative K cutoffs to reconstruct, e.g. '5,20,100,all' (default: all).")
    p.add_argument("--num-samples", dest="num_samples", type=int, default=30,
                   help="How many independent coefficient draws to generate (default: 30).")
    p.add_argument("--seed", dest="seed", type=int, default=0,
                   help="RNG seed (default: 0).")

    p.add_argument("--use-scale", dest="use_scale", action="store_true", default=False,
                   help="Apply an overall log-normal scale if dist NPZ contains scale metadata (default: off).")
    p.add_argument("--add-mean", dest="add_mean", action="store_true", default=True,
                   help="Add back mu_l0 into b[:,0,0] (default: on).")
    p.add_argument("--no-add-mean", dest="add_mean", action="store_false",
                   help="Do NOT add mu_l0 (output will be centered).")

    # Output
    p.add_argument("--out-dir", dest="out_dir", default=None,
                   help="Output directory. Default: mat_converted_N=<N>_synthetic_from_dists/<dist_basename>")
    p.add_argument("--prefix", dest="prefix", default="synthetic",
                   help="Output filename prefix (default: synthetic).")
    p.add_argument("--voxel-size", dest="voxel_size", type=float, default=1.0,
                   help="MRC voxel size (default: 1.0).")
    p.add_argument("--dtype", dest="voxel_dtype", default="float32",
                   help="Voxel dtype for MRC data (numpy dtype string; default: float32).")
    p.add_argument("--normalize", dest="normalize", action="store_true", default=True,
                   help="Normalize each saved volume to max|v|=1 (default: on).")
    p.add_argument("--no-normalize", dest="normalize", action="store_false",
                   help="Disable normalization.")
    p.add_argument("--overwrite", dest="overwrite", action="store_true", default=False,
                   help="Overwrite existing MRC files (default: off).")

    # Advanced overrides
    p.add_argument("--solver", dest="solver_override", default=None,
                   help="Override solver used to build FLE basis (e.g. nvidia_torch, FastTransforms.jl).")
    p.add_argument("--eps-override", dest="eps_override", type=float, default=None,
                   help="Override eps passed to FLEBasis3D (advanced).")
    p.add_argument("--reduce-memory", dest="reduce_memory", action="store_true", default=True,
                   help="Pass reduce_memory=True to FLEBasis3D (default: on).")
    p.add_argument("--no-reduce-memory", dest="reduce_memory", action="store_false",
                   help="Pass reduce_memory=False to FLEBasis3D.")

    return p.parse_args()


def _parse_k_levels(spec: str, k_use: int) -> List[int]:
    parts = [s.strip().lower() for s in (spec or "").split(",") if s.strip()]
    if not parts:
        parts = ["all"]
    out: List[int] = []
    for s in parts:
        if s == "all":
            out.append(int(k_use))
        else:
            try:
                out.append(int(min(int(s), k_use)))
            except Exception as e:
                raise ValueError(f"Bad --k-levels token '{s}': {e}")
    out = sorted(set([k for k in out if k > 0]))
    if not out:
        out = [int(k_use)]
    return out


def main() -> None:
    args = _parse_args()

    dist_path = _resolve_dist_npz_path(args)
    D = np.load(dist_path, allow_pickle=False)

    # --- distributions ---
    required = ("mu", "Sigma", "offsets", "sizes", "ell_per_mode", "N", "L", "K")
    for k in required:
        if k not in D:
            raise RuntimeError(f"Distributions NPZ missing key '{k}': {dist_path}")

    mu = np.asarray(D["mu"], dtype=np.float64)            # (M,2)
    Sigma = np.asarray(D["Sigma"], dtype=np.float64)      # (M,2,2)
    offsets = np.asarray(D["offsets"], dtype=np.int64)
    sizes = np.asarray(D["sizes"], dtype=np.int64)
    ell_per_mode = np.asarray(D["ell_per_mode"], dtype=np.int64)

    Nd = int(np.asarray(D["N"]).item())
    Ld = int(np.asarray(D["L"]).item())
    Kd = int(np.asarray(D["K"]).item())

    m_ordering = _as_py_str(D["m_ordering"]) if "m_ordering" in D else "md"
    jitter = float(np.asarray(D["jitter"]).item()) if "jitter" in D else 1e-9

    # optional scale metadata
    have_scale = bool(np.asarray(D["have_scale"]).item()) if "have_scale" in D else False
    mu_log_scale = float(np.asarray(D["mu_log_scale"]).item()) if "mu_log_scale" in D else 0.0
    sigma_log_scale = float(np.asarray(D["sigma_log_scale"]).item()) if "sigma_log_scale" in D else 0.0

    # dist metadata used for resolving top_modes
    dist_meta: dict = {}
    if "source_top_modes" in D:
        dist_meta["source_top_modes"] = _as_py_str(D["source_top_modes"])
    if "source_coeff_dir" in D:
        dist_meta["source_coeff_dir"] = _as_py_str(D["source_coeff_dir"])

    M_total = int(mu.shape[0])
    if mu.shape != (M_total, 2):
        raise RuntimeError(f"mu has unexpected shape {mu.shape}; expected (M,2).")
    if Sigma.shape != (M_total, 2, 2):
        raise RuntimeError(f"Sigma has unexpected shape {Sigma.shape}; expected (M,2,2).")
    if offsets.ndim != 1 or offsets.shape[0] != Kd + 1:
        raise RuntimeError(f"offsets has shape {offsets.shape}; expected (K+1,) with K={Kd}.")
    if int(offsets[-1]) != M_total:
        raise RuntimeError(f"Mismatch: offsets[-1]={int(offsets[-1])} but mu.shape[0]={M_total}.")
    if sizes.ndim != 1 or sizes.shape[0] != Kd:
        raise RuntimeError(f"sizes has shape {sizes.shape}; expected (K,) with K={Kd}.")
    if ell_per_mode.ndim != 1 or ell_per_mode.shape[0] != Kd:
        raise RuntimeError(f"ell_per_mode has shape {ell_per_mode.shape}; expected (K,) with K={Kd}.")

    print(f"[info] dist: {os.path.basename(dist_path)}")
    print(f"[info] dist meta: N={Nd}, L={Ld}, K={Kd}, M={M_total}, m_ordering={m_ordering}, jitter={jitter:g}")

    # --- modes pack ---
    modes_path = _resolve_modes_npz_path(args, dist_meta)
    MZ = np.load(modes_path, allow_pickle=False, mmap_mode="r")

    for k in ("top_vecs", "top_ell", "N", "L", "eps"):
        if k not in MZ:
            raise RuntimeError(f"Modes NPZ missing key '{k}': {modes_path}")
    if "mu_l0" not in MZ:
        raise RuntimeError(
            "Modes NPZ lacks 'mu_l0'. You must use a centered_global pack that stores the dataset mean.\n"
            f"  modes: {modes_path}"
        )

    top_vecs = MZ["top_vecs"]
    top_ell = np.asarray(MZ["top_ell"], dtype=np.int64)
    Nm = int(np.asarray(MZ["N"]).item())
    Lm = int(np.asarray(MZ["L"]).item())
    eps = float(np.asarray(MZ["eps"]).item())
    solver = _as_py_str(MZ["solver"]) if "solver" in MZ else "nvidia_torch"
    mu_l0 = np.asarray(MZ["mu_l0"], dtype=np.complex128)

    if Nm != Nd or Lm != Ld:
        raise RuntimeError(f"Basis mismatch: dist(N={Nd},L={Ld}) vs modes(N={Nm},L={Lm}).")

    if args.solver_override:
        solver = str(args.solver_override)
    if args.eps_override is not None:
        eps = float(args.eps_override)

    # Harmonize K
    K_use = int(Kd)
    if int(args.k_use) > 0:
        K_use = min(K_use, int(args.k_use))
    K_use = min(K_use, int(top_vecs.shape[0]), int(top_ell.shape[0]))
    if K_use <= 0:
        raise RuntimeError(f"K_use resolved to {K_use}; check inputs.")

    # Consistency check on ell ordering
    if not np.all(top_ell[:K_use] == ell_per_mode[:K_use]):
        n_bad = int(np.sum(top_ell[:K_use] != ell_per_mode[:K_use]))
        print(f"[warn] ell_per_mode differs from modes top_ell for {n_bad}/{K_use} entries; continuing.")

    u_modes = np.asarray(top_vecs[:K_use], dtype=np.complex128)  # (K_use, n_rad)
    if u_modes.ndim != 2:
        raise RuntimeError(f"top_vecs has unexpected ndim={u_modes.ndim}; expected 2.")

    # Build FLE basis
    print(f"[info] modes: {os.path.basename(modes_path)}")
    print(f"[info] build FLE: N={Nd}, L={Ld}, eps={eps:g}, solver={solver}, reduce_memory={bool(args.reduce_memory)}")
    fle = FLEBasis3D(
        N=Nd,
        bandlimit=Ld,
        eps=eps,
        max_l=Ld,
        mode="complex",
        sph_harm_solver=solver,
        reduce_memory=bool(args.reduce_memory),
    )

    # Validate radial dimension
    n_rad = int(getattr(fle, "n_radial", u_modes.shape[1]))
    if u_modes.shape[1] != n_rad:
        raise RuntimeError(
            f"Radial dimension mismatch: top_vecs has n_rad={u_modes.shape[1]} but FLE has n_radial={n_rad}.\n"
            "If you used --eps-override, this mismatch is expected; regenerate modes/dists with the same eps."
        )
    if mu_l0.shape[0] != n_rad:
        raise RuntimeError(f"mu_l0 length {mu_l0.shape[0]} != n_rad {n_rad}.")

    # Parse K-levels
    K_levels = _parse_k_levels(args.k_levels, K_use)
    print(f"[plan] K_use={K_use}; recon levels={K_levels}; samples={int(args.num_samples)}; seed={int(args.seed)}")

    # Prepare per-coordinate Cholesky for (Re,Im)
    SigmaJ = Sigma.copy()
    SigmaJ[:, 0, 0] += jitter
    SigmaJ[:, 1, 1] += jitter

    # 2x2 Cholesky per coordinate, vectorized.
    Lchol = np.zeros_like(SigmaJ)
    a = np.sqrt(np.maximum(SigmaJ[:, 0, 0], 0.0))
    Lchol[:, 0, 0] = a
    safe_a = np.where(a > 0, a, 1.0)
    Lchol[:, 1, 0] = SigmaJ[:, 1, 0] / safe_a
    Lchol[:, 1, 1] = np.sqrt(np.maximum(SigmaJ[:, 1, 1] - (Lchol[:, 1, 0] ** 2), 0.0))

    # If m_ordering is unknown, warn once.
    known_orderings = {"md", "md_order", "0,-1,+1", "torch_md", "neg_to_pos", "neg2pos", "-l..l", "m=-l..l", "m=-ell..ell"}
    warn_unknown_ordering = (m_ordering.strip().lower() not in known_orderings)
    if warn_unknown_ordering:
        print(f"[warn] Unknown m_ordering='{m_ordering}'. Assuming coefficients are already in md-order.")

    # Output config
    voxel_dtype = np.dtype(args.voxel_dtype)
    dist_base = os.path.splitext(os.path.basename(dist_path))[0]
    default_out = os.path.join(f"mat_converted_N={Nd}_synthetic_from_dists", dist_base)
    out_dir = args.out_dir or default_out
    os.makedirs(out_dir, exist_ok=True)

    run_id = time.strftime("%Y%m%d_%H%M%S")
    rng = np.random.default_rng(int(args.seed))

    # cumulative b builder
    b_cum = np.zeros((n_rad, Ld + 1, 2 * Ld + 1), dtype=np.complex128)
    next_r_start = 0

    def add_modes_into_b(b_arr: np.ndarray, r_start: int, r_end: int, alpha_vec: np.ndarray, scale: float) -> None:
        for r in range(r_start, r_end):
            ell = int(ell_per_mode[r])
            if ell < 0 or ell > Ld:
                raise RuntimeError(f"Invalid ell_per_mode[{r}]={ell} (Ld={Ld}).")
            s, e = int(offsets[r]), int(offsets[r + 1])
            seg = alpha_vec[s:e] * scale
            seg_md = _reorder_segment_to_md(seg, ell=ell, m_ordering=m_ordering)
            u_r = u_modes[r]  # (n_rad,)
            m_range = 2 * ell + 1
            # broadcast: (n_rad,1) * (1,m_range)
            b_arr[:, ell, :m_range] += u_r[:, None] * seg_md[None, :]

    # Loop over samples, building each reconstruction from the same alpha draw.
    # We rebuild b_cum per sample (not per K-level) to keep memory low.
    for sidx in tqdm(range(int(args.num_samples)), desc="[gen] samples"):
        # sample alpha in *flattened coordinate order* defined by (offsets/sizes)
        Z = rng.standard_normal((M_total, 2))
        Y = mu + np.einsum("mij,mj->mi", Lchol, Z, optimize=True)
        alpha = (Y[:, 0] + 1j * Y[:, 1]).astype(np.complex128, copy=False)

        # optional global scale (applied only to centered coefficients)
        scale = 1.0
        if args.use_scale:
            if have_scale and sigma_log_scale >= 0:
                scale = float(np.exp(rng.normal(mu_log_scale, sigma_log_scale)))
            else:
                if sidx == 0:
                    print("[warn] --use-scale was set but dist NPZ lacks scale metadata; ignoring.")

        # reset cumulative b for this sample
        b_cum.fill(0)
        next_r_start = 0

        # Make filenames stable per-sample (tie to run_id, seed, and sample index).
        for Kcut in K_levels:
            add_modes_into_b(b_cum, next_r_start, Kcut, alpha_vec=alpha, scale=scale)
            next_r_start = Kcut

            b = b_cum.copy()
            if args.add_mean:
                b[:, 0, 0] = b[:, 0, 0] + mu_l0  # mean is not scaled

            # b -> a -> volume
            a_vec = fle.step3(b)
            f = fle.evaluate(a_vec)
            vol = np.real(f).astype(np.float32, copy=False)
            if args.normalize:
                vol = _normalize_real_volume(vol)

            out_name = (
                f"{args.prefix}_N={Nd}_L={Ld}_K={Kcut}_seed={int(args.seed)}_"
                f"{run_id}_{sidx + 1:04d}.mrc"
            )
            out_path = os.path.join(out_dir, out_name)
            _save_mrc(out_path, vol, voxel_size=float(args.voxel_size), voxel_dtype=voxel_dtype, overwrite=bool(args.overwrite))

    print(f"[done] wrote synthetic volumes → {out_dir}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[abort] KeyboardInterrupt", file=sys.stderr)
        raise
