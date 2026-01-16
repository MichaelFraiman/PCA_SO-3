#!/usr/bin/env python3
"""
final_top_eigenvolumes_reconstruct_all_m.py

Reconstruct the top-T eigenvolumes (largest eigenvalues, PCA order) from:
  top_modes_N=<N>_L=<L>_K=<K>_centered_global_by_raw.npz

For each eigenpair (ℓ, r), this script realizes **all** m ∈ [-ℓ, ℓ].
We DO NOT add the dataset mean; these are basis elements (not reconstructions
of any dataset item).

Output directory (default):  mat_converted_N=<N>_eigenvolumes
Output files:               eigenvol_rank###_ellℓ_rR_mM_lambda<λ>.mrc

Requires: pip install mrcfile

Examples:
  python3 final_top_eigenvolumes_reconstruct_all_m.py --npz path/to/top_modes_...npz --top 50
  python3 final_top_eigenvolumes_reconstruct_all_m.py --nn 256 --top 20 --workers 8
  python3 final_top_eigenvolumes_reconstruct_all_m.py --npz ...npz --solver nvidia_torch --workers 1
"""

import os
import sys

# CRITICAL FIX FOR MACOS SEGFAULT:
# We must strictly control OpenMP/MKL threading before importing numpy/torch.
# This prevents libraries from spawning their own threads inside the Python 
# worker threads/processes, which causes stack corruption and segfaults.
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
from typing import Optional, Tuple, List
from tqdm.auto import tqdm
import mrcfile

# Optional: export reconstructed volumes as MATLAB .mat
try:
    from scipy.io import savemat  # type: ignore
except Exception:  # pragma: no cover
    savemat = None  # type: ignore

# Your library:
from fle_3d import FLEBasis3D  # noqa: E402

# -------- Worker globals (for parallel synthesize path) --------
_WORKER_FLE: Optional[FLEBasis3D] = None
_WORKER_B_ZERO_SHAPE: Optional[Tuple[int, int, int]] = None  # (n_rad, L, 2*L-1)
_WORKER_NORMALIZE: bool = True
_WORKER_VOXEL_SIZE: float = 1.0
_WORKER_VOXEL_DTYPE: np.dtype = np.dtype(np.float32)
_WORKER_EXPORT_MAT: bool = False
_WORKER_MAT_DIR: Optional[str] = None
_WORKER_MAT_COMPRESS: bool = True
_WORKER_MAT_VAR: str = "vol"
# ---------------------------------------------------------------



# --- m-index mapping used by fle_3d_gpt (md-order: 0, -1, +1, -2, +2, ...) ---
def m_to_md(m: int) -> int:
    if m == 0:
        return 0
    if m > 0:
        return 2 * m
    return 2 * (-m) - 1

def md_to_m(md: int) -> int:
    if md == 0:
        return 0
    if md % 2 == 1:
        return -(md + 1) // 2
    return md // 2

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Reconstruct top eigenvolumes (all m) from top_modes_*.npz and write MRC files."
    )

    # NPZ selection
    p.add_argument("--npz", dest="top_modes_npz", default=None,
                   help="Path to top_modes_*.npz. If omitted, uses --npz-glob.")
    p.add_argument("--nn", dest="nn", type=int, default=22,
                   help="N used only for default --matrix-dir/--npz-glob when --npz is not set.")

    # NPZ selection filters (selection-only)
    p.add_argument("--L", dest="L_sel", type=int, default=20,
                   help="Filter NPZ selection by bandlimit L (from filename). Set --L 0 to disable L filtering. Default: 20.")
    p.add_argument("--eps", dest="eps_sel", type=float, default=0.0,
                   help="Filter NPZ selection by eps (from filename, formatted like 1e-06). Set --eps 0 to disable eps filtering. Default: 0.")
    p.add_argument("--matrix-dir", dest="matrix_dir", default=None,
                   help="Matrix directory to search in if --npz not set. Default: mat_converted_N=<nn>_matrix")
    p.add_argument("--npz-glob", dest="npz_glob", default=None,
                   help="Glob pattern for NPZ search if --npz not set. Default: <matrix_dir>/top_modes_N=<nn>_L=*_*centered_global_by_raw.npz")

    # What to export
    p.add_argument("--top", dest="t_top", type=int, default=20,
                   help="How many top (ℓ,r) eigenpairs to export (default: 20).")

    # Output
    p.add_argument("--out-dir", dest="out_dir", default=None,
                   help="Output directory for MRC files. Default: mat_converted_N=<N_from_npz>_eigenvolumes")
    p.add_argument("--normalize", dest="normalize", action="store_true", default=True,
                   help="Normalize each saved volume to max|v|=1 (default: on).")
    p.add_argument("--no-normalize", dest="normalize", action="store_false",
                   help="Disable normalization.")
    p.add_argument("--voxel-size", dest="voxel_size", type=float, default=1.0,
                   help="MRC voxel size (default: 1.0).")
    p.add_argument("--dtype", dest="voxel_dtype", default="float32",
                   help="Voxel dtype for MRC data (numpy dtype string; default: float32).")

    # Optional: also export each reconstructed volume as MATLAB .mat
    p.add_argument("--export-mat", dest="export_mat", action="store_true", default=False,
                   help="Also save each reconstructed volume as .mat (one file per volume). Default: off.")
    p.add_argument("--mat-dir", dest="mat_dir", default=None,
                   help="Output directory for .mat files. Default: same as --out-dir.")
    p.add_argument("--no-mat-compress", dest="mat_compress", action="store_false", default=True,
                   help="Disable compression in savemat (default: compression enabled).")
    p.add_argument("--mat-var", dest="mat_var", default="vol",
                   help="Variable name for the volume array inside the .mat (default: vol).")

    # Backend / performance
    p.add_argument("--workers", dest="max_workers", type=int, default=-1,
                   help="Max worker processes for synthesize-path. -1 => auto. (default: -1)")
    p.add_argument("--force-sequential", dest="force_sequential", action="store_true", default=False,
                   help="Force sequential run (equivalent to --workers 1).")
    p.add_argument("--prefer-denseB", dest="prefer_denseB", action="store_true", default=False,
                   help="If synthesize() exists, still force dense-B fallback (RAM-heavy but simple).")
    p.add_argument("--reduce-memory", dest="reduce_memory", action="store_true", default=True,
                   help="Pass reduce_memory=True to FLEBasis3D (default: on).")
    p.add_argument("--no-reduce-memory", dest="reduce_memory", action="store_false",
                   help="Pass reduce_memory=False to FLEBasis3D.")

    # Overrides (optional)
    p.add_argument("--solver", dest="solver_override", default=None,
                   help="Override solver used to build the FLE basis (e.g. nvidia_torch or FastTransforms.jl).")
    p.add_argument("--eps-override", dest="eps_override", type=float, default=None,
            help="(Advanced) Override eps passed to FLEBasis3D for reconstruction. WARNING: this usually changes n_rad; only use if you know what you are doing.")

    return p.parse_args()


def _resolve_npz_path(args: argparse.Namespace) -> str:
    if args.top_modes_npz:
        if os.path.exists(args.top_modes_npz):
            return args.top_modes_npz
        raise RuntimeError(f"--npz does not exist: {args.top_modes_npz}")

    matrix_dir = args.matrix_dir or f"mat_converted_N={args.nn}_matrix"

    L_tag = "*" if int(getattr(args, "L_sel", 0)) <= 0 else str(int(args.L_sel))
    eps_tag = "*" if float(getattr(args, "eps_sel", 0.0)) <= 0.0 else format(float(args.eps_sel), ".0e").replace("+", "")

    if args.npz_glob:
        globs = [args.npz_glob]
    else:
        globs = [os.path.join(matrix_dir, f"top_modes_N={args.nn}_L={L_tag}_eps={eps_tag}_*centered_global_by_raw.npz")]
        if eps_tag == "*":
            # no eps filtering: accept both new (eps=*) and legacy (no eps)
            globs.append(os.path.join(matrix_dir, f"top_modes_N={args.nn}_L={L_tag}_*centered_global_by_raw.npz"))

    cand = []
    for g in globs:
        cand += glob.glob(g)
    cand = sorted(set(cand))

    if not cand:
        raise RuntimeError(
            "No NPZ found.\n"
            "  Tried globs:\n    " + "\n    ".join(globs) + "\n"
            "  Fix by passing --npz, or adjusting --matrix-dir/--npz-glob, or disable filters with --L 0 --eps 0."
        )

    cand.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return cand[0]


def _save_mrc(path: str, vol: np.ndarray, voxel_size: float, voxel_dtype: np.dtype):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with mrcfile.new(path, overwrite=True) as mrc:
        mrc.set_data(vol.astype(voxel_dtype, copy=False))
        try:
            mrc.voxel_size = (voxel_size, voxel_size, voxel_size)
        except Exception:
            pass


def _save_mat(path: str, vol: np.ndarray, meta: dict, do_compression: bool, var_name: str):
    if savemat is None:
        raise RuntimeError(
            "scipy is required for .mat export, but could not be imported. "
            "Install with: pip install scipy"
        )
    os.makedirs(os.path.dirname(path), exist_ok=True)

    mdict = {var_name: vol}
    # include small metadata fields for convenience
    for k, v in meta.items():
        mdict[k] = v
    savemat(path, mdict, do_compression=bool(do_compression))


def _build_b_zero_shape(fle: FLEBasis3D) -> Tuple[int, int, int]:
    """Probe the exact b-shape via forward on zeros."""
    v0 = np.zeros((fle.N, fle.N, fle.N), dtype=np.float32)
    z0 = fle.step1(v0)
    b0 = fle.step2(z0)
    return b0.shape  # (n_rad, L, 2*L-1)


def _fle_worker_init(N: int, L: int, eps: float, solver: str, reduce_memory: bool,
                     normalize: bool, voxel_size: float, voxel_dtype_str: str,
                     export_mat: bool, mat_dir: Optional[str], mat_compress: bool, mat_var: str):
    """Initializer for synthesize-capable parallel path."""
    global _WORKER_FLE, _WORKER_B_ZERO_SHAPE, _WORKER_NORMALIZE, _WORKER_VOXEL_SIZE, _WORKER_VOXEL_DTYPE
    global _WORKER_EXPORT_MAT, _WORKER_MAT_DIR, _WORKER_MAT_COMPRESS, _WORKER_MAT_VAR
    _WORKER_NORMALIZE = bool(normalize)
    _WORKER_VOXEL_SIZE = float(voxel_size)
    _WORKER_VOXEL_DTYPE = np.dtype(voxel_dtype_str)
    _WORKER_EXPORT_MAT = bool(export_mat)
    _WORKER_MAT_DIR = mat_dir
    _WORKER_MAT_COMPRESS = bool(mat_compress)
    _WORKER_MAT_VAR = str(mat_var)

    _WORKER_FLE = FLEBasis3D(
        N=N, bandlimit=L, eps=eps, max_l=L,
        mode="complex", sph_harm_solver=solver, reduce_memory=reduce_memory
    )
    _WORKER_B_ZERO_SHAPE = _build_b_zero_shape(_WORKER_FLE)


def _synthesize_with_worker(u_rad: np.ndarray, ell: int, rank_idx: int,
                            rad: int, lam: float, m_val: int, out_dir: str) -> str:
    """
    Parallel worker: build b with only (ℓ, r, m=m_val) active and synthesize volume.
    """
    global _WORKER_FLE, _WORKER_B_ZERO_SHAPE, _WORKER_NORMALIZE, _WORKER_VOXEL_SIZE, _WORKER_VOXEL_DTYPE
    global _WORKER_EXPORT_MAT, _WORKER_MAT_DIR, _WORKER_MAT_COMPRESS, _WORKER_MAT_VAR
    if _WORKER_FLE is None or _WORKER_B_ZERO_SHAPE is None:
        raise RuntimeError("Worker not initialized.")
    n_rad0, L0, maxw0 = _WORKER_B_ZERO_SHAPE

    b = np.zeros((n_rad0, L0, maxw0), dtype=np.complex128)
    m_idx = m_to_md(m_val)  # md-order: 0,-1,+1,-2,+2,...
    if not (0 <= m_idx < 2 * ell + 1):
        raise ValueError(f"m={m_val} is out of range for ℓ={ell}.")
    b[:, ell, m_idx] = u_rad

    if not hasattr(_WORKER_FLE, "synthesize"):
        raise RuntimeError("synthesize() missing in worker path (should have been gated in main).")
    vol = _WORKER_FLE.synthesize(b)
    vol = np.real(vol)

    if _WORKER_NORMALIZE:
        vmax = float(np.max(np.abs(vol))) or 1.0
        vol = vol / vmax

    fname = f"eigenvol_rank{rank_idx+1:03d}_ell{ell}_r{rad}_m{m_val:+d}_lambda{lam:.6e}.mrc"
    path = os.path.join(out_dir, fname)
    _save_mrc(path, vol, _WORKER_VOXEL_SIZE, _WORKER_VOXEL_DTYPE)

    if _WORKER_EXPORT_MAT:
        mat_dir = _WORKER_MAT_DIR or out_dir
        mat_path = os.path.join(mat_dir, os.path.splitext(fname)[0] + ".mat")
        meta = {
            "rank": np.int32(rank_idx + 1),
            "ell": np.int32(ell),
            "rad": np.int32(rad),
            "m": np.int32(m_val),
            "lambda": np.float64(lam),
            "voxel_size": np.float64(_WORKER_VOXEL_SIZE),
            "normalized": np.bool_(_WORKER_NORMALIZE),
        }
        _save_mat(mat_path, vol.astype(_WORKER_VOXEL_DTYPE, copy=False), meta,
                  do_compression=_WORKER_MAT_COMPRESS, var_name=_WORKER_MAT_VAR)

    del b, vol
    gc.collect()
    return path


def main():
    args = _parse_args()
    if args.force_sequential:
        args.max_workers = 1

    voxel_dtype = np.dtype(args.voxel_dtype)  # validate early

    # ---------- Load pack ----------
    npz_path = _resolve_npz_path(args)
    pack = np.load(npz_path, allow_pickle=False)

    top_vecs = pack["top_vecs"]        # (K, n_rad) complex
    top_ell  = pack["top_ell"]         # (K,)
    top_rad  = pack["top_rad"]         # (K,)
    top_vals = pack["top_vals_raw"]    # (K,)
    N        = int(pack["N"])
    L        = int(pack["L"])
    n_rad    = int(pack["n_rad"])
    eps_npz  = float(pack["eps"])
    eps_tag_npz = format(eps_npz, ".0e").replace("+", "")
    if float(getattr(args, "eps_sel", 0.0)) > 0.0:
        req_tag = format(float(args.eps_sel), ".0e").replace("+", "")
        if req_tag != eps_tag_npz:
            raise RuntimeError(
                f"Selected NPZ has eps={eps_tag_npz} (eps={eps_npz}), but you requested --eps {req_tag}. "
                f"Pick the matching NPZ or pass --eps 0 to disable eps filtering."
            )
    solver_npz = str(pack["solver"])
    K_total  = int(pack["K"])

    eps = float(args.eps_override) if args.eps_override is not None else eps_npz
    solver = str(args.solver_override) if args.solver_override is not None else solver_npz

    # Ensure true PCA order by λ (defensive)
    order = np.argsort(top_vals)[::-1]
    top_vecs = top_vecs[order]
    top_ell  = top_ell[order]
    top_rad  = top_rad[order]
    top_vals = top_vals[order]

    T = min(int(args.t_top), K_total)
    out_dir = args.out_dir or f"mat_converted_N={N}_eigenvolumes"

    mat_dir = args.mat_dir or out_dir
    if args.export_mat and savemat is None:
        raise RuntimeError(
            "Requested --export-mat, but scipy could not be imported. "
            "Install with: pip install scipy"
        )
    os.makedirs(out_dir, exist_ok=True)

    print(f"[info] NPZ: {npz_path}")
    print(f"[info] N={N}, L={L}, n_rad={n_rad}, eps={eps} (npz={eps_npz}), solver={solver} (npz={solver_npz})")
    print(f"[info] Exporting top {T} eigenpairs; realizing all m; out_dir={out_dir}")
    print(f"[info] normalize={args.normalize}, voxel_size={args.voxel_size}, dtype={voxel_dtype}")

    # Precompute task list (one task per (k, m))
    tasks: List[Tuple[int, int]] = []
    for k in range(T):
        ell = int(top_ell[k])
        for m in range(-ell, ell + 1):
            tasks.append((k, m))
    print(f"[plan] Total volumes to synthesize: {len(tasks)}")

    # ---------- Capability probe ----------
    fle_probe = FLEBasis3D(
        N=N, bandlimit=L, eps=eps, max_l=L,
        mode="complex", sph_harm_solver=solver, reduce_memory=args.reduce_memory
    )
    has_synthesize = hasattr(fle_probe, "synthesize")
    has_denseB = hasattr(fle_probe, "create_denseB") and hasattr(fle_probe, "step3")

    if has_synthesize and not args.prefer_denseB:
        # ---------- Parallel synthesize path ----------
        if args.max_workers in (-1, None, 0):
            workers = os.cpu_count() or 1
            # GPU backends typically need workers=1 to keep VRAM stable
            if "nvidia" in solver.lower() or "cuda" in solver.lower():
                workers = 1
        else:
            workers = max(1, int(args.max_workers))

        print(f"[info] Using synthesize(); workers={workers}")

        results: List[str] = []
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_fle_worker_init,
            initargs=(N, L, eps, solver, args.reduce_memory, args.normalize, args.voxel_size, args.voxel_dtype,
                     args.export_mat, mat_dir, args.mat_compress, args.mat_var),
        ) as ex:
            futs = []
            for (k, m_val) in tasks:
                u   = np.ascontiguousarray(top_vecs[k])  # (n_rad,)
                ell = int(top_ell[k])
                rad = int(top_rad[k])
                lam = float(top_vals[k])
                futs.append(ex.submit(_synthesize_with_worker, u, ell, k, rad, lam, m_val, out_dir))

            for fut in tqdm(as_completed(futs), total=len(futs), desc="Eigenvolumes (all m)"):
                results.append(fut.result())

        print("[done] Wrote:")
        for pth in sorted(results):
            print("  •", pth)

    elif has_denseB:
        # ---------- Sequential dense-B fallback (RAM-safe-ish, but B is huge for big N) ----------
        print("[info] Using dense-B fallback SEQUENTIALLY.")
        try:
            B = fle_probe.create_denseB(numthread=1)
        except Exception as e:
            raise RuntimeError(f"create_denseB failed: {e}")

        n_rad0, L0, maxw0 = _build_b_zero_shape(fle_probe)
        if (n_rad0 != n_rad) or (L0 != L):
            raise RuntimeError(f"b-shape probe ({n_rad0},{L0},{maxw0}) vs NPZ ({n_rad},{L},2L-1).")

        wrote: List[str] = []
        for (k, m_val) in tqdm(tasks, desc="Eigenvolumes (all m)"):
            u   = np.ascontiguousarray(top_vecs[k])  # (n_rad,)
            ell = int(top_ell[k])
            rad = int(top_rad[k])
            lam = float(top_vals[k])

            b = np.zeros((n_rad0, L0, maxw0), dtype=np.complex128)
            m_idx = m_to_md(m_val)
            if not (0 <= m_idx < 2 * ell + 1):
                raise ValueError(f"m={m_val} out of range for ℓ={ell}.")
            b[:, ell, m_idx] = u

            a = fle_probe.step3(b)
            vol = B.dot(a).reshape(N, N, N)
            vol = np.real(vol)

            if args.normalize:
                vmax = float(np.max(np.abs(vol))) or 1.0
                vol = vol / vmax

            fname = f"eigenvol_rank{k+1:03d}_ell{ell}_r{rad}_m{m_val:+d}_lambda{lam:.6e}.mrc"
            path = os.path.join(out_dir, fname)
            _save_mrc(path, vol, args.voxel_size, voxel_dtype)

            if args.export_mat:
                mat_fname = os.path.splitext(fname)[0] + ".mat"
                mat_path = os.path.join(mat_dir, mat_fname)
                meta = {
                    "rank": np.int32(k + 1),
                    "ell": np.int32(ell),
                    "rad": np.int32(rad),
                    "m": np.int32(m_val),
                    "lambda": np.float64(lam),
                    "voxel_size": np.float64(args.voxel_size),
                    "normalized": np.bool_(args.normalize),
                }
                _save_mat(mat_path, vol.astype(voxel_dtype, copy=False), meta,
                          do_compression=args.mat_compress, var_name=args.mat_var)

            wrote.append(path)

            del b, a, vol
            gc.collect()

        print("[done] Wrote:")
        for pth in wrote:
            print("  •", pth)

    else:
        raise RuntimeError(
            "Your FLEBasis3D exposes neither synthesize() nor (create_denseB + step3). "
            "Enable one of those to reconstruct volumes."
        )


if __name__ == "__main__":
    main()