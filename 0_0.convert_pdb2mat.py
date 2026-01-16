#!/usr/bin/env python3
"""
convert_pdb2mat.py

Convert every .pdb.gz in pdb_cache/ → N³ voxel grid → .mat files in mat_converted_N=*/
- Adaptive voxel size per molecule so the whole protein fits into the frame.
- GRID_N controls resolution: larger GRID_N → smaller Å/voxel → more detail.
- Gaussian density with GAUSS_SIGMA (in Å).
- Parallelized with ProcessPoolExecutor.
- Progress bar via tqdm.
- Robust per-file error handling + summary.

CLI:
  All parameters are optional. If not provided, defaults match the values previously hardcoded.
"""

import os
import glob
import argparse
import numpy as np
import prody as pd
from scipy.io import savemat
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm.auto import tqdm

# Reduce BLAS oversubscription in each worker
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def parse_args():
    default_grid_n = 22
    default_pdb_dir = "pdb_cache"
    default_gauss_sigma = 1.0
    default_box_margin_sigmas = 1.0
    default_out_dir = f"mat_converted_N={default_grid_n}"
    default_n_jobs = max(1, (os.cpu_count() or 2) - 1)

    p = argparse.ArgumentParser(
        prog="convert_pdb2mat.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Convert .pdb.gz files to voxelized .mat volumes with adaptive voxel size.",
    )

    p.add_argument("--pdb-dir", default=default_pdb_dir,
                   help="Directory containing *.pdb.gz files")
    p.add_argument("--grid-n", type=int, default=default_grid_n,
                   help="Output grid size (N); volume is N×N×N")
    p.add_argument("--gauss-sigma", type=float, default=default_gauss_sigma,
                   help="Gaussian blur sigma per atom (Å)")
    p.add_argument("--box-margin-sigmas", type=float, default=default_box_margin_sigmas,
                   help="Padding around furthest atom in units of gauss-sigma")
    p.add_argument("--out-dir", default=None,
                   help="Output directory. If omitted, uses mat_converted_N=<grid-n>")
    p.add_argument("--n-jobs", type=int, default=default_n_jobs,
                   help="Number of parallel workers")
    p.add_argument("--skip-existing", dest="skip_existing", action="store_true", default=True,
                   help="Skip files that already exist")
    p.add_argument("--no-skip-existing", dest="skip_existing", action="store_false",
                   help="Recompute and overwrite outputs")

    return p.parse_args()


def atoms_to_grid(coords: np.ndarray, n: int, h: float, sigma: float) -> np.ndarray:
    """
    Return (n,n,n) float32 Gaussian density from atomic coords.

    coords are in Å in a centered coordinate system.
    h is voxel size in Å (same for x, y, z).
    We place voxel centers at:
      x_i = (i - (n - 1)/2) * h,  i = 0,...,n-1
    so the grid spans approximately [-half, +half] with half ≈ (n-1)/2 * h.
    """
    ax = (np.arange(n, dtype=np.float32) - (n - 1) / 2.0) * h
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")

    grid = np.zeros((n, n, n), dtype=np.float32)
    two_s2 = 2.0 * sigma * sigma

    for x, y, z in coords:
        dx = X - x
        dy = Y - y
        dz = Z - z
        grid += np.exp(-(dx * dx + dy * dy + dz * dz) / two_s2).astype(np.float32)

    return grid


def process_one(pdb_path: str, cfg: dict) -> str:
    """
    Parse a single .pdb.gz, voxelize, and save .mat.
    Returns the output path on success. Raises on failure.
    """
    out_dir = cfg["out_dir"]
    grid_n = cfg["grid_n"]
    gauss_sigma = cfg["gauss_sigma"]
    box_margin_sigmas = cfg["box_margin_sigmas"]
    skip_existing = cfg["skip_existing"]

    os.makedirs(out_dir, exist_ok=True)

    base = os.path.basename(pdb_path).split('.')[0]  # '1abc' from '1abc.pdb.gz'
    out_path = os.path.join(out_dir, f"{base}.mat")

    if skip_existing and os.path.exists(out_path):
        return out_path

    pd.confProDy(verbosity='none')
    ag = pd.parsePDB(pdb_path)
    if ag is None:
        raise ValueError("parsePDB returned None")

    sel = ag.select("protein and not hydrogen")
    if sel is None or sel.numAtoms() == 0:
        raise ValueError("no protein heavy atoms found")

    coords = sel.getCoords().astype(np.float32)
    coords_mean = coords.mean(axis=0, keepdims=True)
    coords -= coords_mean

    r_max = np.linalg.norm(coords, axis=1).max()
    margin = box_margin_sigmas * gauss_sigma
    half_box = r_max + margin

    voxel_size = 2.0 * half_box / (grid_n - 1)
    vol = atoms_to_grid(coords, grid_n, voxel_size, gauss_sigma)

    savemat(out_path, {
        "vol": vol,
        "voxel_size": np.float32(voxel_size),
        "gauss_sigma": np.float32(gauss_sigma),
        # "center": coords_mean.astype(np.float32),
    })
    return out_path


def safe_process_one(pdb_path: str, cfg: dict):
    """Wrapper that never raises: returns (pdb_path, ok, message)."""
    try:
        out_path = process_one(pdb_path, cfg)
        return (pdb_path, True, out_path)
    except Exception as e:
        return (pdb_path, False, str(e))


def main():
    args = parse_args()

    out_dir = args.out_dir if args.out_dir is not None else f"mat_converted_N={args.grid_n}"

    cfg = dict(
        pdb_dir=args.pdb_dir,
        grid_n=int(args.grid_n),
        gauss_sigma=float(args.gauss_sigma),
        box_margin_sigmas=float(args.box_margin_sigmas),
        out_dir=out_dir,
        n_jobs=int(args.n_jobs),
        skip_existing=bool(args.skip_existing),
    )

    os.makedirs(cfg["out_dir"], exist_ok=True)

    pdb_files = sorted(glob.glob(os.path.join(cfg["pdb_dir"], "*.pdb.gz")))
    if not pdb_files:
        print(f"No .pdb.gz files found in {cfg['pdb_dir']}/")
        return

    print(f"Converting {len(pdb_files)} files → {cfg['out_dir']}/ with {cfg['n_jobs']} workers")
    successes, failures = [], []

    with ProcessPoolExecutor(max_workers=cfg["n_jobs"]) as ex:
        futures = [ex.submit(safe_process_one, p, cfg) for p in pdb_files]
        for f in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"Converting → {cfg['out_dir']}",
            unit="file",
        ):
            pdb_path, ok, msg = f.result()
            if ok:
                successes.append((pdb_path, msg))
            else:
                failures.append((pdb_path, msg))

    print("\nSummary:")
    print(f"  OK:   {len(successes)}")
    print(f"  FAIL: {len(failures)}")
    if failures:
        print("\nFailures (first 10):")
        for pdb_path, err in failures[:10]:
            print(f"  - {os.path.basename(pdb_path)}: {err}")


if __name__ == "__main__":
    main()
