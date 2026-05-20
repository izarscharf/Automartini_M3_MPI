"""Micro-benchmark for the parallelized find_bead_pos hotspot.

Measures wall-clock time of the trial-enumeration loop only (the dominant cost
in AutoMartiniM3). Skips ALOGPS HTTP calls and downstream topology generation
so the timing reflects parallel CPU work.

Usage:
    # Serial:
    python benchmarks/bench_find_bead_pos.py
    # Parallel:
    mpirun -n 4 python benchmarks/bench_find_bead_pos.py
"""
import time

from rdkit import Chem
from rdkit.Chem import AllChem

from auto_martiniM3 import topology, optimization
from auto_martiniM3.mpi_utils import get_mpi, bcast


# Small to medium molecules, ordered roughly by heavy-atom count.
MOLECULES = [
    ("propane",       "CCC"),
    ("benzene",       "c1ccccc1"),
    ("toluene",       "Cc1ccccc1"),
    ("aspirin",       "CC(=O)OC1=CC=CC=C1C(=O)O"),
    ("ibuprofen",     "CC(C)Cc1ccc(cc1)C(C)C(=O)O"),
    ("caffeine",      "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"),
    ("naphthalene",   "c1ccc2ccccc2c1"),
    ("anthracene",    "c1ccc2cc3ccccc3cc2c1"),
]


def time_find_bead_pos(smiles: str):
    comm, rank, size = get_mpi()

    # Rank 0 builds the molecule (stochastic embedding), then broadcasts.
    if rank == 0:
        mol, _ = topology.gen_molecule_smi(smiles)
        mol = Chem.Mol(mol)
        AllChem.EmbedMolecule(mol)
        AllChem.MMFFOptimizeMolecule(mol, maxIters=1000, mmffVariant="MMFF94s")
    else:
        mol = None
    mol = bcast(mol)

    list_heavy_atoms, _ = topology.get_atoms(mol)
    conf, heavy_coords, all_coords = topology.get_heavy_atom_coords(mol)
    ring_atoms = topology.get_ring_atoms(mol)
    ring_flat = [a for r in ring_atoms for a in r]

    if comm is not None:
        comm.Barrier()
    t0 = time.perf_counter()
    optimization.find_bead_pos(
        mol, conf, list_heavy_atoms, heavy_coords, all_coords,
        ring_atoms, ring_flat, force_map=False,
    )
    if comm is not None:
        comm.Barrier()
    return time.perf_counter() - t0, len(list_heavy_atoms)


def main():
    _, rank, size = get_mpi()
    if rank == 0:
        print(f"# MPI ranks: {size}")
        print(f"{'molecule':<14}{'n_heavy':>8}{'wall_s':>12}")
    for name, smi in MOLECULES:
        try:
            dt, n_heavy = time_find_bead_pos(smi)
        except Exception as e:
            if rank == 0:
                print(f"{name:<14}{'?':>8}  FAILED: {e}")
            continue
        if rank == 0:
            print(f"{name:<14}{n_heavy:>8}{dt:>12.3f}")


if __name__ == "__main__":
    main()
