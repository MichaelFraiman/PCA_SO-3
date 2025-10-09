#!/usr/bin/env python3
"""
convert_pdb2mat.py
Convert every .pdb.gz in pdb_cache/ → N³ voxel grid → .mat files in mat_vols_N/
- Parallelized with ProcessPoolExecutor
- Progress bar via tqdm
- Robust per-file error handling + summary
"""

import os
import glob
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

# ── user settings ────────────────────────────────────────────────────────────
PDB_DIR      = "pdb_cache"             # where your .pdb.gz files live
GRID_N       = 128                     # output grid will be GRID_N³
VOXEL_SIZE   = 2.0                     # Å per voxel
GAUSS_SIGMA  = 1.0                     # Å Gaussian blur per atom
OUT_DIR      = f"mat_converted_N={GRID_N}"
N_JOBS       = max(1, (os.cpu_count() or 2) - 1)  # parallel workers
SKIP_EXISTING = True                   # skip files that already exist
# ─────────────────────────────────────────────────────────────────────────────

def atoms_to_grid(coords: np.ndarray, n: int, h: float, sigma: float) -> np.ndarray:
    """Return (n,n,n) float32 Gaussian density from atomic coords."""
    half = (n // 2) * h
    ax   = np.linspace(-half, half - h, n, dtype=np.float32)
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")

    grid   = np.zeros((n, n, n), dtype=np.float32)
    two_s2 = 2.0 * sigma * sigma
    for x, y, z in coords:
        dx, dy, dz = X - x, Y - y, Z - z
        grid += np.exp(-(dx*dx + dy*dy + dz*dz) / two_s2, dtype=np.float32)
    return grid

def process_one(pdb_path: str) -> str:
    """
    Parse a single .pdb.gz, voxelize, and save .mat.
    Returns the output path on success. Raises on failure.
    """
    # build output name
    base = os.path.basename(pdb_path).split('.')[0]  # '1abc' from '1abc.pdb.gz'
    out_path = os.path.join(OUT_DIR, f"{base}.mat")

    if SKIP_EXISTING and os.path.exists(out_path):
        return out_path  # treat as success

    # parse, keep protein heavy atoms only
    # suppress prody chattiness
    pd.confProDy(verbosity='none')
    ag = pd.parsePDB(pdb_path)
    if ag is None:
        raise ValueError("parsePDB returned None")
    sel = ag.select("protein and not hydrogen")
    if sel is None or sel.numAtoms() == 0:
        raise ValueError("no protein heavy atoms found")

    # center coordinates
    coords = sel.getCoords().astype(np.float32)
    coords -= coords.mean(axis=0, keepdims=True)

    # voxelize
    vol = atoms_to_grid(coords, GRID_N, VOXEL_SIZE, GAUSS_SIGMA)

    # save
    savemat(out_path, {"vol": vol})
    return out_path

def safe_process_one(pdb_path: str):
    """Wrapper that never raises: returns (pdb_path, ok, message)."""
    try:
        out_path = process_one(pdb_path)
        return (pdb_path, True, out_path)
    except Exception as e:
        return (pdb_path, False, str(e))

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    pdb_files = sorted(glob.glob(os.path.join(PDB_DIR, "*.pdb.gz")))
    if not pdb_files:
        print(f"No .pdb.gz files found in {PDB_DIR}/")
        return

    print(f"Converting {len(pdb_files)} files → {OUT_DIR}/ with {N_JOBS} workers")
    successes, failures = [], []

    with ProcessPoolExecutor(max_workers=N_JOBS) as ex:
        futures = [ex.submit(safe_process_one, p) for p in pdb_files]
        for f in tqdm(as_completed(futures),
                      total=len(futures),
                      desc=f"Converting → {OUT_DIR}",
                      unit="file"):
            pdb_path, ok, msg = f.result()
            if ok:
                successes.append((pdb_path, msg))  # msg is out_path
            else:
                failures.append((pdb_path, msg))   # msg is error string

    # Summary
    print("\nSummary:")
    print(f"  OK:   {len(successes)}")
    print(f"  FAIL: {len(failures)}")
    if failures:
        print("\nFailures (first 10):")
        for pdb_path, err in failures[:10]:
            print(f"  - {os.path.basename(pdb_path)}: {err}")

if __name__ == "__main__":
    main()
