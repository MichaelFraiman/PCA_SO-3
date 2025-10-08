#!/usr/bin/env python3
"""
Batch-parallel-convert all PDB files in pdb_cache/ to MATLAB `.mat` volumes (appending `_converted` to filenames), writing outputs to a configurable directory.

Dependencies:
    pip install prody scipy numpy
"""
import os
import glob
import numpy as np
import prody as pd
from scipy.io import savemat
from scipy.ndimage import gaussian_filter
from concurrent.futures import ProcessPoolExecutor, as_completed

# User-configurable parameters
GRID_N            = 256                 # number of voxels per axis
INPUT_DIR         = 'pdb_cache'         # folder containing .pdb files
OUTPUT_DIR        = f'mat_converted_N={GRID_N}'  # folder to save converted .mat files
VOXEL_SIZE        = 1                   # Å per voxel
GAUSSIAN_SIGMA    = 1.0                 # smoothing sigma in voxels (0 for none)
MAT_SUFFIX        = '_converted.mat'    # suffix for output filenames
NUM_WORKERS       = None                # None == os.cpu_count()


def pdb_to_volume(pdb_path):
    # Parse coordinates
    structure = pd.parsePDB(pdb_path)
    coords    = structure.getCoords()

    # Shift so minimum coordinate is at origin
    min_coord = coords.min(axis=0)
    shifted   = coords - min_coord

    # Compute integer voxel indices
    indices = np.floor(shifted / VOXEL_SIZE).astype(int)

    # Build empty volume
    volume = np.zeros((GRID_N, GRID_N, GRID_N), dtype=np.float32)
    for x, y, z in indices:
        if 0 <= x < GRID_N and 0 <= y < GRID_N and 0 <= z < GRID_N:
            volume[x, y, z] += 1.0

    # Optional Gaussian smoothing
    if GAUSSIAN_SIGMA > 0:
        volume = gaussian_filter(volume, sigma=GAUSSIAN_SIGMA)

    return volume


def process_pdb(pdb_path):
    basename = os.path.splitext(os.path.basename(pdb_path))[0]
    output_name = basename + MAT_SUFFIX
    mat_path = os.path.join(OUTPUT_DIR, output_name)

    volume = pdb_to_volume(pdb_path)
    savemat(mat_path, {'volume': volume})
    return pdb_path, mat_path


def main():
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pattern = os.path.join(INPUT_DIR, '*.pdb')
    pdb_files = sorted(glob.glob(pattern))
    if not pdb_files:
        print(f"No PDB files found in {INPUT_DIR}")
        return

    # Parallel processing
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        for pdb_path, mat_path in executor.map(process_pdb, pdb_files):
            print(f"Converted '{pdb_path}' -> '{mat_path}'")

if __name__ == '__main__':
    main()
