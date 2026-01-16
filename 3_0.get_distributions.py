#!/usr/bin/env python3
"""fake_get_distributions.py

Build empirical *per-coordinate* complex-Gaussian marginals for saved eigen-coefficients.

This script aggregates coefficients across a set of per-volume NPZs and fits an independent
2D Gaussian to each complex coordinate (Re, Im):

  mu[j]    = [E[Re(alpha_j)], E[Im(alpha_j)]]
  Sigma[j] = Cov([Re(alpha_j), Im(alpha_j)])

It is intended to be used after computing a PCA pack:
  top_modes_N=<N>_L=<L>_eps=<eps>_K=<K>_centered_global_by_raw.npz

and after exporting per-volume coefficients.

Supported per-volume coefficient formats
--------------------------------------

The script supports *either* of the following NPZ layouts (auto-detected):

1) "alpha_padded" layout (legacy)
   - alpha_padded : (K, M_max) complex
     Row i corresponds to i-th top eigenpair (in the same order as the PCA pack).
     The first (2*ell_i+1) entries are the m-coefficients for that eigenpair.

2) "A_l<ell>" block layout (current export in 0_1.covariance_matrix.py)
   - A_l<ell> : (n_rows, 2*ell+1) complex
     Optionally also stores:
       rows_l<ell> : (n_rows,) int, the radial indices included in A_l<ell>

In both cases, the m-axis ordering is assumed to be the FLE "md" ordering:
  md = 0,1,2,3,... corresponds to m = 0,-1,+1,-2,+2,...

Output
------

Writes a single compressed NPZ containing mu/Sigma and the mode layout:

  <out_dir>/<out_stem>_<top_base>_N=<N>_L=<L>_Kused=<K_use>.npz
    mu:      (M, 2)    float32
    Sigma:   (M, 2,2)  float32
    offsets: (K_use+1,) int32
    sizes:   (K_use,)   int32  where sizes[i] = 2*top_ell[i] + 1
    ell_per_mode: (K_use,) int32
    rad_per_mode: (K_use,) int32
    plus metadata fields.

Examples
--------

Auto-pick the newest PCA pack and coefficient files:

  python3 fake_get_distributions.py --nn 22 --L 20 --k-use 200

Specify the PCA pack explicitly:

  python3 fake_get_distributions.py --npz mat_converted_N=22_matrix/top_modes_N=22_L=20_eps=1e-06_K=580_centered_global_by_raw.npz \
      --k-use 200

Point at a custom coefficient directory/glob:

  python3 fake_get_distributions.py --coeff-dir mat_converted_N=22_matrix \
      --coeff-glob 'mat_converted_N=22_matrix/*_coeffs_L=20_eps=1e-06_centered_global*.npz'
"""

import os
import sys

# CRITICAL FIX FOR MACOS SEGFAULT (keep consistent with other ExpMax scripts):
# Strictly control OpenMP/MKL threading before importing numpy.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import glob
import argparse
import numpy as np
from typing import Dict, List, Optional, Tuple, Literal

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **k):
        return x


CoeffFormat = Literal["alpha_padded", "blocks"]


# ============================== Defaults ==============================

DEFAULT_NN = 22
DEFAULT_L_FILTER = 20
DEFAULT_EPS_FILTER = 0.0
DEFAULT_K_USE = 200

DEFAULT_DDOF = 1
DEFAULT_JITTER = 1e-9

DEFAULT_OUT_STEM = "marginals_from_saved_coeffs"
# =====================================================================


def _eps_tag(eps: float) -> str:
    return format(float(eps), ".0e").replace("+", "")


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fit per-coordinate complex-Gaussian marginals from saved eigen-coefficients.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # PCA pack selection
    p.add_argument("--npz", dest="top_modes_npz", default=None,
                   help="Path to top_modes_*.npz. If omitted, auto-select newest match.")
    p.add_argument("--nn", type=int, default=DEFAULT_NN,
                   help="Grid size N used for default directory naming when --npz is not set.")
    p.add_argument("--L", dest="L_filter", type=int, default=DEFAULT_L_FILTER,
                   help="Filter auto-selection of PCA packs by L. Use 0 to disable filtering.")
    p.add_argument("--eps", dest="eps_filter", type=float, default=DEFAULT_EPS_FILTER,
                   help="Filter auto-selection of PCA packs by eps. Use 0 to disable filtering.")
    p.add_argument("--matrix-dir", default=None,
                   help="Directory to search for PCA packs if --npz not set. Default: mat_converted_N=<nn>_matrix")
    p.add_argument("--npz-glob", default=None,
                   help="Optional glob pattern for PCA pack search if --npz not set.")

    # Mode selection
    p.add_argument("--k-use", type=int, default=DEFAULT_K_USE,
                   help="Use at most this many top eigenpairs from the PCA pack (0 means use all available).")

    # Coefficient files
    p.add_argument("--coeff-dir", default=None,
                   help="Root directory to search for per-volume coeff NPZs. If omitted, tries common defaults.")
    p.add_argument("--coeff-glob", default=None,
                   help="Glob pattern for coefficient NPZs. Overrides --coeff-dir auto-search.")
    p.add_argument("--coeff-format", choices=["auto", "alpha_padded", "blocks"], default="auto",
                   help="Coefficient NPZ format. 'auto' inspects the first matching file.")

    # Statistics
    p.add_argument("--ddof", type=int, default=DEFAULT_DDOF,
                   help="Degrees of freedom correction for sample covariance (1 => unbiased).")
    p.add_argument("--jitter", type=float, default=DEFAULT_JITTER,
                   help="Diagonal jitter added to Sigma to avoid singular matrices.")

    # Output
    p.add_argument("--out-dir", default=None,
                   help="Output directory. Default: mat_converted_N=<N_from_npz>_coeffs_distributions")
    p.add_argument("--out-stem", default=DEFAULT_OUT_STEM,
                   help="Output filename stem.")
    p.add_argument("--overwrite", action="store_true", default=False,
                   help="Overwrite output NPZ if it already exists.")

    # Progress bars
    p.add_argument("--pbar", dest="pbar", action="store_true", default=True,
                   help="Enable progress bars.")
    p.add_argument("--no-pbar", dest="pbar", action="store_false",
                   help="Disable progress bars.")

    return p.parse_args()


def _resolve_top_modes_npz(args: argparse.Namespace) -> str:
    """Resolve a PCA pack path.

    Matches the selection logic used in other scripts: when not provided,
    choose the newest matching pack in the matrix directory.
    """
    if args.top_modes_npz:
        if os.path.exists(args.top_modes_npz):
            return args.top_modes_npz
        raise RuntimeError(f"--npz does not exist: {args.top_modes_npz}")

    matrix_dir = args.matrix_dir or f"mat_converted_N={int(args.nn)}_matrix"
    L_tag = "*" if int(getattr(args, "L_filter", 0)) <= 0 else str(int(args.L_filter))
    eps_tag = "*" if float(getattr(args, "eps_filter", 0.0)) <= 0.0 else _eps_tag(float(args.eps_filter))

    globs: List[str]
    if args.npz_glob:
        globs = [args.npz_glob]
    else:
        globs = [os.path.join(matrix_dir, f"top_modes_N={args.nn}_L={L_tag}_eps={eps_tag}_*centered_global_by_raw.npz")]
        if eps_tag == "*":
            # legacy packs without eps tag
            globs.append(os.path.join(matrix_dir, f"top_modes_N={args.nn}_L={L_tag}_*centered_global_by_raw.npz"))

    cand: List[str] = []
    for g in globs:
        cand += glob.glob(g)
    cand = sorted(set(cand))

    if not cand:
        raise RuntimeError(
            "No PCA pack found.\n"
            "  Tried globs:\n    " + "\n    ".join(globs) + "\n"
            "  Fix by passing --npz or adjusting --matrix-dir/--npz-glob or disabling filters with --L 0 --eps 0."
        )

    cand.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return cand[0]


def _detect_coeff_format(npz_path: str) -> CoeffFormat:
    with np.load(npz_path, allow_pickle=False) as d:
        keys = set(d.files)
        if "alpha_padded" in keys:
            return "alpha_padded"
        if any(k.startswith("A_l") for k in keys):
            return "blocks"
    raise RuntimeError(
        f"Could not detect coefficient format for {os.path.basename(npz_path)}. "
        "Expected 'alpha_padded' or keys like 'A_l0', 'A_l1', ..."
    )


def _resolve_coeff_files(
    args: argparse.Namespace,
    top_base: str,
    N: int,
    L: int,
    eps: float,
    K_use: int,
) -> Tuple[List[str], List[str]]:
    """Return (files, tried_globs)."""
    if args.coeff_glob:
        files = sorted(glob.glob(args.coeff_glob))
        return files, [args.coeff_glob]

    eps_tag = _eps_tag(eps)

    # Candidate roots to try (order matters).
    roots: List[str] = []
    if args.coeff_dir:
        roots.append(args.coeff_dir)
    else:
        # Legacy layout described in the original script.
        roots.append(f"mat_converted_N={N}_eigen_coeffs_top{K_use}")
        # Current default: per-volume coeffs often live alongside the PCA pack.
        roots.append(f"mat_converted_N={N}_matrix")

    tried: List[str] = []
    files: List[str] = []

    # A small collection of likely filename conventions.
    patterns_per_root: List[str] = [
        # Legacy alpha_padded naming under a subfolder named by top_base.
        os.path.join("{root}", top_base, f"*_N={N}_eigen_coeffs_top{K_use}.npz"),
        os.path.join("{root}", top_base, f"*eigen_coeffs_top{K_use}.npz"),
        # 0_1 export naming (blocks format) usually in matrix dir.
        os.path.join("{root}", f"*_coeffs_L={L}_eps={eps_tag}_centered_global*.npz"),
        # Fallback if eps tag differs / absent.
        os.path.join("{root}", f"*_coeffs_L={L}_*centered_global*.npz"),
    ]

    for root in roots:
        for pat in patterns_per_root:
            g = pat.format(root=root)
            tried.append(g)
            files = sorted(glob.glob(g))
            if files:
                return files, tried

    return [], tried


def _build_layout_from_top_modes(top_ell: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
    """Given (K_use,) top_ell, build (sizes, offsets, M_total)."""
    sizes = (2 * top_ell.astype(np.int32) + 1).astype(np.int32)
    offsets = np.zeros((sizes.size + 1,), dtype=np.int32)
    offsets[1:] = np.cumsum(sizes, dtype=np.int64).astype(np.int32)
    M_total = int(offsets[-1])
    return sizes, offsets, M_total


def _accumulate_alpha_padded(
    d: np.lib.npyio.NpzFile,
    K_use: int,
    sizes: np.ndarray,
    offsets: np.ndarray,
    sum_r: np.ndarray,
    sum_i: np.ndarray,
    sum_rr: np.ndarray,
    sum_ii: np.ndarray,
    sum_ri: np.ndarray,
):
    A = d["alpha_padded"]
    if A.ndim != 2:
        raise RuntimeError(f"alpha_padded must be 2D, got shape {A.shape}")
    if A.shape[0] < K_use:
        raise RuntimeError(f"alpha_padded has K={A.shape[0]} but K_use={K_use}")

    for r in range(K_use):
        cnt = int(sizes[r])
        s, e = int(offsets[r]), int(offsets[r + 1])
        seg = A[r, :cnt]

        sr = np.asarray(seg.real, dtype=np.float64)
        si = np.asarray(seg.imag, dtype=np.float64)

        sum_r[s:e] += sr
        sum_i[s:e] += si
        sum_rr[s:e] += sr * sr
        sum_ii[s:e] += si * si
        sum_ri[s:e] += sr * si


def _accumulate_blocks(
    d: np.lib.npyio.NpzFile,
    K_use: int,
    top_ell: np.ndarray,
    top_rad: np.ndarray,
    sizes: np.ndarray,
    offsets: np.ndarray,
    sum_r: np.ndarray,
    sum_i: np.ndarray,
    sum_rr: np.ndarray,
    sum_ii: np.ndarray,
    sum_ri: np.ndarray,
):
    # Group modes by ell to avoid repeated key lookups.
    idx_by_ell: Dict[int, List[int]] = {}
    for i in range(K_use):
        ell = int(top_ell[i])
        idx_by_ell.setdefault(ell, []).append(i)

    for ell, idx_list in idx_by_ell.items():
        A_key = f"A_l{ell}"
        if A_key not in d:
            raise RuntimeError(f"Missing key {A_key} in coeff NPZ")
        A = d[A_key]

        rows_key = f"rows_l{ell}"
        row_map: Optional[Dict[int, int]] = None
        if rows_key in d:
            rows = d[rows_key].astype(np.int64)
            row_map = {int(r): j for j, r in enumerate(rows.tolist())}

        for i in idx_list:
            rad = int(top_rad[i])
            if row_map is None:
                row_idx = rad
            else:
                if rad not in row_map:
                    raise RuntimeError(
                        f"Coeff NPZ is missing ell={ell} rad={rad}. "
                        f"(rows_l{ell} does not contain {rad})"
                    )
                row_idx = int(row_map[rad])

            cnt = int(sizes[i])  # = 2*ell+1
            if A.ndim != 2 or A.shape[1] < cnt:
                raise RuntimeError(f"{A_key} has shape {A.shape}, expected (*, {cnt})")
            if row_idx < 0 or row_idx >= A.shape[0]:
                raise RuntimeError(f"{A_key}: row index {row_idx} out of range (shape {A.shape})")

            seg = A[row_idx, :cnt]
            s, e = int(offsets[i]), int(offsets[i + 1])

            sr = np.asarray(seg.real, dtype=np.float64)
            si = np.asarray(seg.imag, dtype=np.float64)

            sum_r[s:e] += sr
            sum_i[s:e] += si
            sum_rr[s:e] += sr * sr
            sum_ii[s:e] += si * si
            sum_ri[s:e] += sr * si


def main() -> None:
    args = _parse_args()

    # 1) Load PCA pack
    top_npz_path = _resolve_top_modes_npz(args)
    top_base = os.path.splitext(os.path.basename(top_npz_path))[0]

    with np.load(top_npz_path, allow_pickle=False) as dtop:
        N = int(dtop["N"])
        L = int(dtop["L"])
        eps = float(dtop["eps"]) if "eps" in dtop else 0.0
        solver = str(dtop["solver"]) if "solver" in dtop else "unknown"
        K_avail = int(dtop["K"]) if "K" in dtop else int(dtop["top_ell"].shape[0])

        top_ell_all = dtop["top_ell"].astype(np.int32)
        top_rad_all = dtop["top_rad"].astype(np.int32)
        top_vals_raw_all = dtop["top_vals_raw"] if "top_vals_raw" in dtop else None

    K_use = int(K_avail) if int(args.k_use) <= 0 else min(int(args.k_use), int(K_avail))
    if K_use <= 0:
        raise RuntimeError(f"K_use must be >= 1, got {K_use}")

    top_ell = top_ell_all[:K_use]
    top_rad = top_rad_all[:K_use]
    top_vals_raw = top_vals_raw_all[:K_use] if top_vals_raw_all is not None else None

    sizes, offsets, M_total = _build_layout_from_top_modes(top_ell)

    out_dir = args.out_dir or f"mat_converted_N={N}_coeffs_distributions"
    _ensure_dir(out_dir)

    # 2) Find coefficient NPZs
    coeff_files, tried_globs = _resolve_coeff_files(
        args=args,
        top_base=top_base,
        N=N,
        L=L,
        eps=eps,
        K_use=K_use,
    )

    if not coeff_files:
        msg = (
            f"[error] No coefficient NPZs found.\n"
            f"  top_modes: {top_npz_path}\n"
            f"  tried globs:\n    " + "\n    ".join(tried_globs)
        )
        raise RuntimeError(msg)

    # 3) Determine coefficient format
    if args.coeff_format == "auto":
        coeff_format: CoeffFormat = _detect_coeff_format(coeff_files[0])
    else:
        coeff_format = "alpha_padded" if args.coeff_format == "alpha_padded" else "blocks"

    # 4) Prepare accumulators
    sum_r = np.zeros((M_total,), dtype=np.float64)
    sum_i = np.zeros((M_total,), dtype=np.float64)
    sum_rr = np.zeros((M_total,), dtype=np.float64)
    sum_ii = np.zeros((M_total,), dtype=np.float64)
    sum_ri = np.zeros((M_total,), dtype=np.float64)

    print(f"[info] PCA pack: {top_npz_path}")
    print(f"[info] N={N}, L={L}, eps={eps}, solver={solver}")
    print(f"[info] Using K_use={K_use} eigenpairs → total coords M={M_total}")
    print(f"[info] Coeff format: {coeff_format}")
    print(f"[scan] Found {len(coeff_files)} coefficient NPZs")

    it = coeff_files
    if args.pbar:
        it = tqdm(coeff_files, desc="[accum] per-volume coeffs", leave=True)

    for path in it:
        with np.load(path, allow_pickle=False) as d:
            if coeff_format == "alpha_padded":
                _accumulate_alpha_padded(
                    d=d,
                    K_use=K_use,
                    sizes=sizes,
                    offsets=offsets,
                    sum_r=sum_r,
                    sum_i=sum_i,
                    sum_rr=sum_rr,
                    sum_ii=sum_ii,
                    sum_ri=sum_ri,
                )
            else:
                _accumulate_blocks(
                    d=d,
                    K_use=K_use,
                    top_ell=top_ell,
                    top_rad=top_rad,
                    sizes=sizes,
                    offsets=offsets,
                    sum_r=sum_r,
                    sum_i=sum_i,
                    sum_rr=sum_rr,
                    sum_ii=sum_ii,
                    sum_ri=sum_ri,
                )

    V = int(len(coeff_files))
    print(f"[stats] Aggregated V={V} volumes")

    ddof = int(args.ddof)
    if V <= ddof:
        raise RuntimeError(f"Need V > ddof for covariance; got V={V}, ddof={ddof}")

    # 5) Means and covariances
    mu_r = sum_r / V
    mu_i = sum_i / V
    denom = float(V - ddof)

    var_rr = (sum_rr - V * mu_r * mu_r) / denom
    var_ii = (sum_ii - V * mu_i * mu_i) / denom
    cov_ri = (sum_ri - V * mu_r * mu_i) / denom

    # Numerical safety: variances should be nonnegative up to precision.
    var_rr = np.maximum(var_rr, 0.0)
    var_ii = np.maximum(var_ii, 0.0)

    mu = np.stack([mu_r, mu_i], axis=1).astype(np.float32)  # (M,2)
    Sigma = np.empty((M_total, 2, 2), dtype=np.float32)
    jitter = float(args.jitter)
    Sigma[:, 0, 0] = (var_rr + jitter).astype(np.float32)
    Sigma[:, 1, 1] = (var_ii + jitter).astype(np.float32)
    Sigma[:, 0, 1] = Sigma[:, 1, 0] = cov_ri.astype(np.float32)

    # 6) Save
    out_name = f"{str(args.out_stem)}_{top_base}_N={N}_L={L}_Kused={K_use}"
    out_path = os.path.join(out_dir, f"{out_name}.npz")
    if (not args.overwrite) and os.path.exists(out_path):
        raise RuntimeError(f"Output exists (use --overwrite): {out_path}")

    save_dict = dict(
        mu=mu,
        Sigma=Sigma,
        offsets=offsets,
        sizes=sizes,
        ell_per_mode=top_ell.astype(np.int32),
        rad_per_mode=top_rad.astype(np.int32),
        N=np.int32(N),
        L=np.int32(L),
        K=np.int32(K_use),
        eps=np.float64(eps),
        solver=np.array(solver),
        m_ordering=np.array("md"),
        coeff_format=np.array(coeff_format),
        source_top_modes=np.array(os.path.basename(top_npz_path)),
        source_top_modes_path=np.array(top_npz_path),
        coeff_glob=np.array(args.coeff_glob if args.coeff_glob else ""),
        coeff_dir=np.array(args.coeff_dir if args.coeff_dir else ""),
        file_count=np.int32(V),
        ddof=np.int32(ddof),
        jitter=np.float32(jitter),
    )
    if top_vals_raw is not None:
        save_dict["top_vals_raw"] = np.asarray(top_vals_raw)

    np.savez_compressed(out_path, **save_dict)
    print(f"[save] wrote distributions -> {out_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[interrupt]", file=sys.stderr)
        raise
