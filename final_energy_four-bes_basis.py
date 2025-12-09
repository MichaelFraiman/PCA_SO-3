#!/usr/bin/env python3
"""
final_energy_four-bes_basis.py
Parallel cumulative-energy computation with a progress bar.
"""

import os
import glob
import traceback
import numpy as np
from scipy.io import loadmat
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional, Tuple
from tqdm.auto import tqdm  # <-- progress bar (TTY-friendly; degrades under nohup)

try:
    from fle_3d import FLEBasis3D
except Exception as e:
    raise RuntimeError("Could not import fle_3d.FLEBasis3D. Make sure fle_3d is installed.") from e

# ----------------------- CONFIG -----------------------
# If using GPU ("nvidia_torch"), too many processes contend for the GPU and can appear stuck.
SOLVER = "nvidia_torch"  # change to "fasttransforms" for CPU-only
RESERVE_CORES = 1
DEFAULT_WORKERS = 12 if "nvidia" in SOLVER else max(1, (os.cpu_count() or 2) - RESERVE_CORES)
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", DEFAULT_WORKERS))
REDUCE_MEMORY = True
NN = 128
# ------------------------------------------------------


# Per-process globals
_fle = None  # type: Optional[FLEBasis3D]
_out_dir = None  # type: Optional[str]
_key_cache = {}  # per-process small cache


def get_mat_list(directory, pattern="*.mat"):
    return sorted(glob.glob(os.path.join(directory, pattern)))


def _detect_first_data_key(mat_dict: dict) -> str:
    for k in mat_dict:
        if not k.startswith('__'):
            return k
    raise KeyError("No data key found in .mat file (only __* keys present).")


def _init_worker(N: int, L: int, eps: float, solver: str, reduce_memory: bool, out_dir: str):
    """Per-process initializer; creates a local FLEBasis3D and stores out_dir."""
    global _fle, _out_dir
    _out_dir = out_dir
    _fle = FLEBasis3D(
        N=N, bandlimit=L, eps=eps, max_l=L,
        mode="complex",
        sph_harm_solver=solver,    # "nvidia_torch" or "fasttransforms"
        reduce_memory=reduce_memory,
    )


def _process_one(m_path: str) -> Tuple[str, Optional[str], Optional[int], Optional[str]]:
    """Returns: (basename, out_csv, ncoef, error_msg)"""
    global _fle, _out_dir, _key_cache
    basename = os.path.splitext(os.path.basename(m_path))[0]
    try:
        if _fle is None or _out_dir is None:
            raise RuntimeError("Worker not initialized: FLE or out_dir is None.")

        data = loadmat(m_path)
        key = _key_cache.get(m_path)
        if key is None:
            key = _detect_first_data_key(data)
            _key_cache[m_path] = key

        vol = np.ascontiguousarray(data[key], dtype=np.float64)
        vmax = np.max(np.abs(vol))
        if vmax == 0:
            return basename, None, None, f"Skip {basename}: volume is all zeros"
        vol /= vmax

        # FLE expansion
        z = _fle.step1(vol)
        b = _fle.step2(z)  # expected shape: (n_rad, L, 2ℓ+1)

        # Flatten, sort, cumulative energy
        a = b.flatten()
        idx = np.argsort(np.abs(a))[::-1]
        a_sorted = a[idx]
        energies = np.abs(a_sorted) ** 2
        total = energies.sum()
        ratios = np.zeros_like(energies, dtype=np.float64) if total == 0 else np.cumsum(energies) / total

        ncoef = int(energies.size)
        out_csv = os.path.join(_out_dir, f"{basename}_coeff_energy.csv")
        with open(out_csv, "w") as f:
            f.write("k,w_ratio\n")
            for k in range(1, ncoef + 1):
                f.write(f"{k},{ratios[k-1]:.12g}\n")

        return basename, out_csv, ncoef, None

    except Exception as e:
        tb = traceback.format_exc(limit=3)
        return basename, None, None, f"{e.__class__.__name__}: {e}\n{tb}"


def main(rescaled_dir=f'mat_converted_N={NN}',
         out_dir=f'mat_converted_N={NN}_energy_fle',
         L=20, eps=1e-6):
    """
    For each .mat volume in `rescaled_dir`:
    - Expand in the FLE basis
    - Sort coefficients by magnitude
    - Compute w(k) = sum_{n=1}^k |a_n|^2 / sum_{n=1}^N |a_n|^2
    - Save ALL k=1..N to CSV in `out_dir` as: k,w_ratio
    """
    os.makedirs(out_dir, exist_ok=True)

    mats = get_mat_list(rescaled_dir)
    if not mats:
        print(f"No .mat files found in '{rescaled_dir}'", flush=True)
        return

    # Infer N from first volume
    samp = loadmat(mats[0])
    key0 = _detect_first_data_key(samp)
    vol0 = np.ascontiguousarray(samp[key0], dtype=np.float64)
    vmax0 = np.max(np.abs(vol0))
    if vmax0 == 0:
        raise ValueError("First volume is all zeros; cannot infer N.")
    vol0 /= vmax0
    N = vol0.shape[0]

    print(f"[init] Found {len(mats)} .mat files in '{rescaled_dir}'.", flush=True)
    print(f"[init] Grid N={N}, L={L}, eps={eps}, solver='{SOLVER}', workers={NUM_WORKERS}", flush=True)
    print(f"[init] Output dir: {out_dir}", flush=True)

    done = 0
    errors = 0
    with ProcessPoolExecutor(
        max_workers=NUM_WORKERS,
        initializer=_init_worker,
        initargs=(N, L, eps, SOLVER, REDUCE_MEMORY, out_dir),
    ) as ex:
        futures = [ex.submit(_process_one, m) for m in mats]
        # Progress bar
        with tqdm(total=len(futures), desc="processing volumes", unit="vol",
                  dynamic_ncols=True, mininterval=0.5, smoothing=0.1) as pbar:
            for fut in as_completed(futures):
                basename, out_csv, ncoef, err = fut.result()
                if err:
                    tqdm.write(f"[error] {basename}: {err.strip()}")
                    errors += 1
                else:
                    tqdm.write(f"[ok] {basename}: coefficients={ncoef} → {out_csv}")
                done += 1
                pbar.update(1)

    print(f"[done] processed={done}, errors={errors}, out_dir='{out_dir}'", flush=True)


if __name__ == '__main__':
    # Avoid BLAS oversubscription (helps when using many workers)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    # Make stdout unbuffered for live logs under nohup
    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    main()