"""Micro-benchmark for the ALOGPS prefetch.

Times prefetch_alogps over a list of fragment SMILES, forcing HTTP traffic by
pointing the local logP database at /dev/null so nothing short-circuits via
the file cache. Compares:
  - serial baseline: smi2alogps one-by-one (old behaviour)
  - prefetch_alogps with threads (and MPI distribution when launched with mpirun)
"""
import os
import time

from auto_martiniM3 import topology
from auto_martiniM3.mpi_utils import get_mpi


# Fragments unlikely to be in the local database (mix of small heteroatom
# combos -- the actual values don't matter, only that ALOGPS responds).
FRAGMENTS = [
    "CCO", "CCN", "CCS", "CCCl", "CCBr",
    "c1ccccc1O", "c1ccccc1N", "c1ccccc1F", "c1ccccc1Cl",
    "CC(=O)N", "CC(=O)O", "CC(N)=O", "C(=O)N",
    "CCOC", "CCSC", "CCNC", "CCOCC",
]


def reset_caches():
    topology._ALOGPS_CACHE.clear()
    topology._LOGP_FILE_CACHE.clear()


def main():
    _, rank, size = get_mpi()
    # Force HTTP path by pointing at an empty file.
    empty_db = "/tmp/_empty_logp_db.dat"
    if rank == 0:
        open(empty_db, "w").close()

    # ----- Serial baseline (rank 0 only, single-threaded sequential calls) -----
    if rank == 0:
        reset_caches()
        t0 = time.perf_counter()
        for smi in FRAGMENTS:
            topology.smi2alogps(False, smi, None, 1, False, None, empty_db, False)
        serial = time.perf_counter() - t0
        print(f"Serial sequential ({len(FRAGMENTS)} fragments): {serial:.2f}s")

    # ----- Parallel prefetch -----
    reset_caches()
    comm, _, _ = get_mpi()
    if comm is not None:
        comm.Barrier()
    t0 = time.perf_counter()
    pairs = [(smi, False) for smi in FRAGMENTS]
    topology.prefetch_alogps(pairs, logp_file=empty_db, max_workers=8)
    if comm is not None:
        comm.Barrier()
    dt = time.perf_counter() - t0
    if rank == 0:
        print(f"prefetch_alogps  MPI ranks={size}, threads<=8: {dt:.2f}s")


if __name__ == "__main__":
    main()
