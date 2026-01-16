# Readme

This repository contains reference code accompanying:

- Michael Fraiman, Paulina Hoyos, Tamir Bendory, Joe Kileel, Oscar Mickelin, Nir Sharon, Amit Singer,
  **“SO(3)-invariant PCA with application to molecular data”**, arXiv:2510.18827 (2025).

The pipeline voxelizes molecules from PDB files, expands the resulting 3D volumes in a spherical Fourier–Bessel basis, and computes an SO(3)-invariant covariance and its eigendecomposition.

## Citation

If you use this code, please cite the paper above.

## Update

Please note that on 16.01.2026 the repository was updated.
The update includes major code refactoring and stability improvements.

## Repository Layout

The repository includes the following scripts:

### Main Scripts

- `0_0.convert_pdb2mat.py`: voxelize `*.pdb.gz` files into `N×N×N` volumes and write MATLAB `.mat` files.

- `0_1.covariance_matrix.py`: compute SO(3)-block covariance, apply global centering for the `ℓ=0` block, and export PCA modes.

### Plotting

- `1_0.FB_expand.py`: expand volumes in the Fourier–Bessel basis and write cumulative energy CSVs. (Run `0_1.covariance_matrix.py` with the `--export-csv` flag to execute this script).

- `1_1.compare_PCA_FB.py`: compare per-volume PCA energy concentration vs Fourier–Bessel energy curves. (Must be run after `1_0.FB_expand.py`)

### Volume Visualization

- `2_0.top_eigenvolumes_reconstruct.py`: reconstruct the leading eigenvolumes (basis elements) and write `.mrc` files. (Requires running `0_1.covariance_matrix.py` with the `--export-coeffs` flag).

- `2_1.reconstruct_target_volumes.py`: approximate selected target volumes using the leading eigenvolumes.

### Synthesizing Fake Structures


- `3_0.get_distributions.py`: fit per-coordinate complex-Gaussian marginals to saved PCA coefficients. (Requires running `0_1.covariance_matrix.py` with the `--export-coeffs` flag).

- `3_1.generate_fake_volumes.py`: sample coefficients from those marginals and reconstruct synthetic (“fake”) volumes.

### Library Code

- `fle_3d.py`: fast Laplacian eigenfunction expansion on the ball (FLEBasis3D), adapted from upstream code *(see Acknowledgements)*.
This `FLEBasis3D` implementation is based on code from the repository: `oscarmickelin/fle_3d`,
which accompanies the paper _Joe Kileel, Nicholas F. Marshall, Oscar Mickelin, Amit Singer,
  “Fast expansion into harmonics on the ball,” SIAM Journal on Scientific Computing 47(2), A1117–A1144 (2025), DOI: 10.1137/24M1668159._

- Slight modifications have been implemented due to the incompatibility with the latest version of the `scipy` library.

## Dependencies and Tested Environment

This repository was developed/tested in the following environment on MacOS 26.2 (with M4 Pro CPU):

- Python 3.14.2
- pip 25.3

Key Python packages:

- `numpy==2.4.1`
- `scipy==1.17.0`
- `torch==2.9.1` and `torch_harmonics==0.8.0` (default spherical-harmonics backend)
- `finufft==2.4.1`
- `mrcfile==1.5.4` (writing `.mrc` volumes)
- `tqdm==4.67.1` (progress bars)
- `matplotlib==3.10.8` (plots in `1_1.compare_PCA_FB.py`)
- `ProDy==2.6.1` (PDB parsing in `0_0.convert_pdb2mat.py`)

Optional / backend-specific dependencies:

- Julia backend (only if you set `--solver FastTransforms.jl`): `juliacall==0.9.31`, `juliapkg==0.1.22` + a working Julia installation + the required Julia packages.
- Dense-matrix fallback for large grids (only used for `N>32` inside `create_denseB`): `pyshtools==4.13.1`.



## Required data files for FLEBasis3D

`fle_3d_gpt.py` expects the following precomputed tables to live next to the Python file:

- `jl_zeros_l=3000_k=2500.mat`
- `cs_l=3000_k=2500.mat`

These are taken from the upstream FLE repository:

```text
https://github.com/oscarmickelin/fle_3d
```

## How to run

All scripts provide CLI help:

```bash
python3 <script>.py -h
```

### 0) Put PDB files in the cache

Place your dataset as compressed PDBs:

- `pdb_cache/*.pdb.gz`

(You can start with a small subset to test the pipeline.)

### 1) Voxelize PDB → `.mat`

```bash
python3 0_0.convert_pdb2mat.py --pdb-dir pdb_cache --grid-n 22
```

This writes volumes to:

- `mat_converted_N=22/*.mat`

### 2) Compute covariance blocks and PCA modes

Basic run:

```bash
python3 0_1.covariance_matrix.py --expect-n 22 --L 20 --eps 1e-6 --solver nvidia_torch
```

If you want later steps that require per-volume PCA coefficients (plots, distributions, synthesis), add:

```bash
python3 0_1.covariance_matrix.py --expect-n 22 --L 20 --eps 1e-6 --solver nvidia_torch --export-coeffs
```

Outputs are written under:

- `mat_converted_N=22_matrix/`

### 3) (Optional) Fourier–Bessel expansions + energy comparison plots

Compute per-volume Fourier–Bessel cumulative energy curves:

```bash
python3 1_0.FB_expand.py --expect-n 22 --L 20 --eps 1e-6 --solver nvidia_torch
```

Generate per-volume comparison plots (PCA vs FB):

```bash
python3 1_1.compare_PCA_FB.py --expect-n 22 --L 20 --eps 1e-6 --first-k 200
```

### 4) Reconstruct the leading eigenvolumes (basis elements)

```bash
python3 2_0.top_eigenvolumes_reconstruct.py --nn 22 --top 20
```

This writes `.mrc` files (one per eigenpair, for all `m`):

- `mat_converted_N=22_eigenvolumes/*.mrc`

### 5) Reconstruct / approximate selected target volumes

```bash
python3 2_1.reconstruct_target_volumes.py --nn 22 --L 20 --eps 1e-6 --solver nvidia_torch \
  --targets 1abc 2xyz --k-list 5 10 20 50 100 200
```

(Replace `1abc 2xyz` with basenames of `.mat` files in `mat_converted_N=22/`.)

### 6) Fit coefficient distributions

This step requires the per-volume coefficient exports from step (2) (`--export-coeffs`).

```bash
python3 3_0.get_distributions.py --nn 22 --L 20 --k-use 200
```

### 7) Generate synthetic (“fake”) volumes

```bash
python3 3_1.generate_fake_volumes.py --nn 22 --L 20 --k-levels 5,20,100,all --num-samples 30
```

## Output folders (defaults)

For grid size `N=22`, the scripts use these default folders:

- `mat_converted_N=22/` — voxelized `.mat` volumes.
- `mat_converted_N=22_matrix/` — covariance blocks, PCA packs, and (optional) per-volume coefficient NPZs.
- `mat_converted_N=22_FBexpansions/` — Fourier–Bessel energy CSVs.
- `mat_converted_N=22_plots/` — PCA-vs-FB comparison plots.
- `mat_converted_N=22_eigenvolumes/` — reconstructed eigenvolumes (`.mrc`).
- `mat_converted_N=22_reconstructed_mrc/` — target reconstructions (`.mrc`).
- `mat_converted_N=22_coeffs_distributions/` — fitted distribution parameters (`.npz`).
- `mat_converted_N=22_synthetic_from_dists/` — synthetic volumes (`.mrc`).

## Notes / troubleshooting

- **SciPy 1.17+ compatibility**: SciPy removed `scipy.special.sph_harm`. This repo’s `fle_3d.py` is intended to use `scipy.special.sph_harm_y` (modern API). If you see an error about missing `sph_harm`, ensure you are using the patched FLE file.
- **macOS threading**: the scripts set `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, etc. before importing numerical libraries to reduce oversubscription/segfault risk.

## Acknowledgements

The `FLEBasis3D` implementation is based on code from the repository:

```text
oscarmickelin/fle_3d
```

which accompanies:

- Joe Kileel, Nicholas F. Marshall, Oscar Mickelin, Amit Singer,
  “Fast expansion into harmonics on the ball,” SIAM Journal on Scientific Computing 47(2), A1117–A1144 (2025), DOI: 10.1137/24M1668159.
