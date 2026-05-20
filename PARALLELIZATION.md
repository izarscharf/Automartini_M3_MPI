# Parallelization in this fork of Auto_MartiniM3

This document describes the performance work added in this fork. It explains
**what was slow**, **how the parallel paths are structured**, and **what
the measured speed-ups are**. It is meant as a companion to the development
log in [CLAUDE_DEVELOPMENT_LOG.md](CLAUDE_DEVELOPMENT_LOG.md), which captures
how the changes were arrived at.

The work was carried out with the assistance of Claude Code (Anthropic) under
interactive human guidance.

---

## 1. What was slow

A small profiling/code-reading pass identified two hot-spots that dominate
wall time for medium-sized molecules (≈ 13–20 heavy atoms):

### 1.1 Bead-placement search — `optimization.find_bead_pos`

Source: [auto_martiniM3/optimization.py](auto_martiniM3/optimization.py)

For a molecule with `N` heavy atoms, the function enumerates every possible
way to place `k` beads on the heavy-atom positions for every `k` between
`floor(N/4)` and `floor(N/2) + 1`. That is `Σ C(N, k)` trial combinations,
which already exceeds `10⁵` for `N = 15`.

For **each** trial combination the function calls:

1. `check_beads(...)` — `O(k²)` validity check (no two beads at the same
   atom; no inter-ring bonds between bead-atoms; terminal-bead heuristics).
2. `eval_gaussian_interac(...)` — energy scoring. Internally O(N²) RDKit
   `Chem.rdMolTransforms.GetBondLength` calls (each call is a Python ↔ C++
   round-trip).
3. `all_atoms_in_beads_connected(...)` → `voronoi_atoms_new/old(...)` —
   another O(N²) Voronoi partition pass.

Per-trial cost is on the order of milliseconds; per-molecule total is
seconds-to-minutes once `N ≳ 15`.

**Crucially, trials are independent.** They share read-only state
(`molecule`, `conformer`, `ring_atoms`, …) and only update a few summary
variables (`best_trial_comb`, `ene_best_trial`, `list_trial_comb`). This is
the textbook embarrassingly-parallel pattern.

### 1.2 ALOGPS HTTP queries — `topology.smi2alogps`

Source: [auto_martiniM3/topology.py](auto_martiniM3/topology.py)

For every CG bead, `topology.print_atoms` calls `smi2alogps` to obtain the
water/octanol partition free energy for the bead's SMILES fragment. The
free energy then determines the Martini 3 bead type. The original
implementation:

1. Re-opened and re-parsed `logP_smi.dat` (~125 lines) on **every** call.
2. If the fragment was not in that local database, made a blocking HTTP
   request to `http://vcclab.org/web/alogps/calc?SMILES=…` and parsed the
   HTML response with BeautifulSoup.
3. Was called **twice** per solver iteration (once in `trial=True` mode to
   gather bead types for the additivity check, once in `trial=False` mode
   to actually write the topology) — and also once from
   `check_additivity` for the whole-molecule SMILES.

The HTTP round-trip dominates for any molecule whose fragments are not
already cached in `logP_smi.dat`. Per-bead calls are independent.

---

## 2. Design

### 2.1 Module: `auto_martiniM3/mpi_utils.py`

A small wrapper that returns `(comm, rank, size)` from `mpi4py` when the
process was launched under `mpirun`, and `(None, 0, 1)` otherwise. The
serial path stays bit-identical to the original code — every parallel
branch checks `size > 1` before doing anything different. `mpi4py` is an
**optional** dependency: if it cannot be imported, the wrapper transparently
returns the serial defaults.

### 2.2 Parallelizing `find_bead_pos`

Source: [auto_martiniM3/optimization.py](auto_martiniM3/optimization.py),
function `find_bead_pos`.

Strategy:

* All ranks regenerate the same `itertools.combinations(list_heavy_atoms, num_beads)`
  list (cheap, deterministic, avoids broadcasting the index list).
* Each rank processes a **strided slice** of the trial indices:
  `range(rank, len(seq_one_beads), size)`. Stride > block partitioning
  because the acceptance fraction varies along the index range; strided
  slicing balances load implicitly.
* Each rank keeps local `list_trial_comb`, `combs`, `energies`, and a
  local-best `(ene, comb)` tuple.
* After each `num_beads` block, the ranks `gather` their local lists onto
  rank 0; rank 0 folds them into the global state and decides whether the
  early-exit condition (`last_best_trial_comb == best_trial_comb`) is met.
  That decision plus `ene_best_trial` and `best_trial_comb` are
  `bcast` back so every rank loops in lockstep.
* After the outermost loop the rank-0 `list_trial_comb` is broadcast so
  callers (potentially running on every rank) get an identical return
  value.

The conformer-generating step in `solver.Cg_molecule.__init__` is stochastic
(`AllChem.EmbedMolecule`) and was previously called independently on every
process when launched under MPI, yielding mismatched coordinates between
ranks. The fix: only rank 0 calls `EmbedMolecule` / `MMFFOptimizeMolecule`,
then broadcasts the resulting `Chem.Mol` (RDKit `Mol` objects pickle with
their conformers).

File writes (`.itp`, `.gro`, bartender input) and the “Converged” print are
gated behind `is_root()`.

### 2.3 Caching and parallelizing ALOGPS

Source: [auto_martiniM3/topology.py](auto_martiniM3/topology.py),
functions `smi2alogps`, `prefetch_alogps`, helpers `_load_logp_file`,
`_alogps_http_request`.

Three changes:

1. **Load `logP_smi.dat` once.** `_load_logp_file()` keeps a module-level
   dict keyed by filename and returns it on subsequent calls. Previously
   the file was re-opened and re-parsed on every `smi2alogps` invocation.
2. **Memoise ALOGPS results.** Module-level `_ALOGPS_CACHE` maps
   `(query_smi, is_mol)` → `(ΔG, origin)`. This collapses the duplicate
   work between trial-mode and final-mode passes of `print_atoms`, across
   multiple solver attempts, and across the `check_additivity` call.
3. **Batch and parallelize the HTTP calls.** `prefetch_alogps(smi_pairs)`
   does the following:
   * Deduplicates input, skips entries already cached, short-circuits any
     hit in the local `logP_smi.dat`.
   * Issues all remaining requests **from rank 0** through a
     `concurrent.futures.ThreadPoolExecutor` (default 8 workers).
   * Broadcasts the resulting dict to all ranks (`comm.bcast`) so every
     rank's cache ends up identical without any rank duplicating an HTTP
     call.

`print_atoms` was modified to do a small pre-pass that computes
`substruct2smi` for every bead, collects the un-charged fragment SMILES,
and calls `prefetch_alogps` once. The existing per-bead loop then runs
unchanged but hits the cache.

#### Why HTTP is not distributed across MPI ranks

An earlier version distributed the HTTP fetches across ranks the same way
`find_bead_pos` distributes its trials. Benchmarking showed this is
**counterproductive**:

* `vcclab.org` rate-limits per source IP.
* `mpirun -n N` on one host means all `N` ranks share that host's IP.
* `4 ranks × 8 threads = 32` concurrent requests from one IP → the server
  starts throttling, so the batch took **~5× longer** than `1 rank × 8 threads`.

The final design therefore uses **rank 0 + threads + broadcast**, which
gives a clean ~2× speed-up regardless of rank count and avoids antagonising
the public ALOGPS service.

---

## 3. Files changed / added

| File | Change |
|------|--------|
| [auto_martiniM3/mpi_utils.py](auto_martiniM3/mpi_utils.py) | **New.** Thin wrapper exposing `get_mpi()`, `is_root()`, `bcast()`, `barrier()` with serial fallback. |
| [auto_martiniM3/optimization.py](auto_martiniM3/optimization.py) | `find_bead_pos` is now MPI-aware: strided trial loop, per-rank locals, gather/bcast at each `num_beads` block. `voronoi_atoms_new` / `voronoi_atoms_old` fixed to return global-indexed `partitioning` (see §7). `all_atoms_in_beads_connected` updated accordingly. |
| [auto_martiniM3/topology.py](auto_martiniM3/topology.py) | `smi2alogps` rewritten to use a memoised cache and a single-load `logP_smi.dat`; new `prefetch_alogps()` and HTTP helper `_alogps_http_request`; `print_atoms` does a one-shot prefetch pass. |
| [auto_martiniM3/solver.py](auto_martiniM3/solver.py) | Stochastic embed/minimisation run only on rank 0 and the molecule broadcast; file writes and the converged-print gated on `is_root()`. |
| [auto_martiniM3/\_\_main\_\_.py](auto_martiniM3/__main__.py) | `.gro` writes gated on `is_root()`. |
| [benchmarks/bench_find_bead_pos.py](benchmarks/bench_find_bead_pos.py) | **New.** Times the bead-placement search hotspot over a fixed list of small/medium molecules. |
| [benchmarks/bench_alogps.py](benchmarks/bench_alogps.py) | **New.** Forces the HTTP path (by pointing the local DB at an empty file) and compares sequential vs. `prefetch_alogps`. |

The previously published serial API and CLI are unchanged.

---

## 4. Measured speed-ups

All numbers are wall-clock seconds on a single 8-core machine.

### 4.1 `find_bead_pos` (hot path of the solver)

Trial-enumeration time only. Measured by
[`benchmarks/bench_find_bead_pos.py`](benchmarks/bench_find_bead_pos.py).

| Molecule    | N\_heavy | Serial | `-n 2` | `-n 4` | `-n 8` (oversubscribed) |
|-------------|---------:|-------:|-------:|-------:|------------------------:|
| propane     |        3 |  0.009 |  0.004 |  0.003 |                   0.003 |
| benzene     |        6 |  0.028 |  0.016 |  0.010 |                   0.009 |
| toluene     |        7 |  0.034 |  0.026 |  0.015 |                   0.016 |
| naphthalene |       10 |  0.250 |  0.075 |  0.079 |                   0.078 |
| aspirin     |       13 |  0.98  |  0.58  |  0.40  |                    0.37 |
| caffeine    |       14 |  3.06  |  1.59  |  0.72  |                    1.13 |
| anthracene  |       14 |  1.83  |  0.94  |  0.66  |                    0.82 |
| ibuprofen   |       15 |  4.43  |  1.96  |  1.01  |                    1.26 |

* On the larger molecules we observe up to **4.4× speed-up at 4 ranks**.
* 8 ranks on an 8-core host regresses for some molecules — at that point
  the work per rank is small enough that MPI gather/bcast and process
  contention eat the savings.
* End-to-end wall time on ibuprofen went from **6.55 s → 5.00 s** with 4
  ranks; the remaining time is dominated by the ALOGPS path and one-time
  Python imports.

### 4.2 `prefetch_alogps` (HTTP batching)

Measured by [`benchmarks/bench_alogps.py`](benchmarks/bench_alogps.py) with
the local DB pointed at an empty file, so all 17 fragments go through HTTP.

| Configuration                              | Wall time |
|--------------------------------------------|----------:|
| Old serial (one-by-one)                    |   5.7 s   |
| `prefetch_alogps`, 1 rank, ≤ 8 threads     |   **2.7 s** (2.1×) |
| `prefetch_alogps`, 4 ranks, ≤ 8 threads on rank 0 |   2.8 s   |

The cache also eliminates redundant calls: a 5-bead molecule that
previously made up to `2 × 5 + 1 = 11` HTTP requests across one solver
attempt + check_additivity now makes at most 5 (and zero if all fragments
are in `logP_smi.dat`).

---

## 5. Correctness

* Tests under `auto_martiniM3/tests/` (`pytest -v auto_martiniM3/tests`)
  still pass.
* For aspirin and ibuprofen we diffed the `.itp` produced by the serial
  and the MPI runs (`mpirun -n 4`): the bead types, atom partitioning,
  bonds, angles, dihedrals, and exclusions are identical. The only
  difference is the user-supplied `--mol` molname column.
* The serial path is exercised whenever the program is invoked without
  `mpirun`, since `mpi_utils.get_mpi()` returns `(None, 0, 1)` in that
  case and every parallel branch is gated on `size > 1`.

---

## 6. What is **not** parallelized (and why)

* **Topology generation downstream of `find_bead_pos`** — `print_bonds`,
  `print_angles`, `print_dihedrals`, virtual-sites bookkeeping. These are
  cheap (microseconds), heavily intertwined, and gain little from
  parallelism.
* **The Voronoi partitioning calls inside the solver loop.** They are
  already short relative to the trial-enumeration loop and are themselves
  invoked from inside that loop, so they benefit indirectly from
  `find_bead_pos` parallelism.
* **Cross-IP MPI distribution of HTTP requests.** As described in §2.3,
  rate limiting at the ALOGPS service makes this actively harmful when
  multiple ranks share a host.

---

## 7. Bug fix — explicit-hydrogen / deuterium molecules

Discovered while attempting to run deuterium-labelled SDS
(`[2H]C([2H])([2H])...OS(=O)(=O)[O-]`) which crashed with `KeyError: 20`
inside `voronoi_atoms_new`.

### Root cause

`voronoi_atoms_new` and `voronoi_atoms_old` build `partitioning` as

```python
for j in range(len(heavyatom_coords)):   # j = local heavy-atom index 0..N-1
    partitioning[j] = bead
```

Every downstream caller (`substruct2smi`, `solver.Cg_molecule`,
`topology.print_atoms`, …) treats the keys of `partitioning` as **global**
RDKit atom indices and does things like `molecule.GetAtomWithIdx(key)`.

For molecules with no explicit H/D atoms, local and global indices are
identical and the code works by accident. For deuterium-labelled molecules
the heavy atoms sit at non-contiguous global indices (e.g. [1,4,7,…,41]
for SDS with `[2H]`) so:

* The hydrogen-assignment section checked `at1 in partitioning.keys()` with
  a global bond endpoint (`at1 = 19`) against local keys `{0..16}`. Since
  `19 > 16`, the check failed, the deuterium at global index 20 was never
  inserted into `aa_partitioning`, and the COG loop raised `KeyError: 20`.
* Even when no crash occurred (e.g. for D atoms whose global index happened
  to fall within `{0..N-1}`), the wrong local-keyed bead was returned,
  giving silently incorrect bead assignments and substructure SMILES.

### Fix

In both `voronoi_atoms_new` and `voronoi_atoms_old`
([optimization.py](auto_martiniM3/optimization.py)):

```python
# After running the Voronoi algorithm with local indices, convert
# partitioning to global atom indices before returning.
_heavy_global = [gi for gi in range(molecule.GetNumAtoms())
                 if molecule.GetAtomWithIdx(gi).GetAtomicNum() != 1]
partitioning = {_heavy_global[j]: bead for j, bead in partitioning.items()}
```

The hydrogen-assignment loop and COG calculation were updated to work with
the now-global `partitioning` directly (no further translation needed).

`all_atoms_in_beads_connected` was the one caller that *correctly*
converted global → local via `list_heavyatoms.index()` before lookup. Those
wrappers were removed now that the dict is already globally keyed:

```python
# Before:
voronoi[list_heavyatoms.index(cg_bead)]
# After:
voronoi[cg_bead]          # cg_bead is a global atom index from trial_comb
```

### Verification

Deuterium-labelled SDS now produces a valid 7-bead `.itp` in both serial
and MPI (`mpirun -n 4`) modes with identical output. The existing SMILES
test suite (aspirin, propane) continues to pass without change.

---

## 8. How to reproduce the numbers

```bash
conda activate automartiniM3
pip install -e .              # picks up mpi_utils.py and the new benchmark files
pip install mpi4py            # optional; only needed for the MPI path

# Hotspot 1
python benchmarks/bench_find_bead_pos.py
mpirun -n 2 python benchmarks/bench_find_bead_pos.py
mpirun -n 4 python benchmarks/bench_find_bead_pos.py

# Hotspot 2 (requires network access to vcclab.org)
python benchmarks/bench_alogps.py
mpirun -n 4 python benchmarks/bench_alogps.py
```
