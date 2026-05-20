# Claude-assisted development log

This document is a faithful record of how the parallelization work in this
fork was planned and implemented. It is intentionally narrative rather than
prescriptive: the goal is so that a reader can understand not just *what*
changed (see [PARALLELIZATION.md](PARALLELIZATION.md) for that) but **why**
each design choice was made and **what alternatives were considered and
rejected**.

The implementation was done in a single interactive session using
[Claude Code](https://www.anthropic.com/claude-code), the Anthropic command-line
agent. The human in the loop directed the work, asked clarifying questions,
made architectural calls, and validated each step. Claude Code was used to
do the code reading, edits, command execution, and benchmarking.

---

## Original prompt

> As an experienced Scientific Computing Engineer:
> Auto-MartiniM3 currently runs single threaded, it scores many possible
> versions of partitioning a molecule into beads and does many other tasks
> which are not inherently sequential. This makes it very slow for larger
> molecules.
>
> Your task is to find parallelizable steps in Auto-MartiniM3 and make them
> MPI compatible.
> 1. Look through the package and identify key computationally expensive
>    steps.
> 2. Plan the conversion to parallelization.
> 3. Rewrite the code so that it runs in parallel.
> 4. Benchmark the two versions on smaller molecules which will run in
>    a reasonable time on my local machine.
>
> Info: the conda automartiniM3 env already exists.

A follow-up later in the session asked to also "implement the
parallelization of the ALOGPS and fix the logP_smi.dat loading."

---

## Step 1 — Profiling by code reading

Rather than running a profiler we read the call graph from
`auto_martiniM3/__main__.py` downwards. The entry point delegates to
`solver.Cg_molecule`, which calls `optimization.find_bead_pos` and then
loops over candidate mappings calling several `topology.*` helpers.

Two hot-spots stood out almost immediately:

### Hot-spot 1: `optimization.find_bead_pos`

The structure of this function (paraphrased) is:

```python
for num_beads in range(min_beads, max_beads + 1):
    seq_one_beads = list(itertools.combinations(list_heavy_atoms, num_beads))
    for trial_comb in seq_one_beads:
        if check_beads(trial_comb, ...):
            energy = eval_gaussian_interac(trial_comb, ...)   # O(N²) RDKit calls
            if all_atoms_in_beads_connected(trial_comb, ...): # another O(N²)
                # update best, append to list_trial_comb
```

For a 15-heavy-atom molecule the total number of trials is on the order of
`10⁵`–`10⁶`. The inner work is RDKit-bound (lots of `GetBondLength` calls),
not numpy-bound, so multi-threading would be limited by the GIL and the
RDKit Python ↔ C++ overhead. **Multi-process (MPI) parallelism was the
natural fit.**

The most important observation was that **trials are independent**:
they share read-only state (molecule, conformer, ring atoms, …) and only
contribute to summary structures at the end of each `num_beads` block.

### Hot-spot 2: `topology.smi2alogps`

`print_atoms` makes one call per CG bead. Each call:

* Re-opens `logP_smi.dat` and scans it linearly.
* If no local hit, performs a blocking HTTP request to `vcclab.org`.

Both the file I/O and the HTTP latency are pure I/O and dominate the per-call
cost. The HTTP calls are independent across beads; they form a textbook
target for thread-pool concurrency.

### Side-band cost

`AllChem.EmbedMolecule` and `MMFFOptimizeMolecule` in
`Cg_molecule.__init__` are sub-second for the test molecules, so we didn't
parallelize them — but we *did* have to handle the fact that they are
**stochastic**: under `mpirun -n N`, each rank would otherwise run them
independently and end up with a different conformer. This had to be fixed
for the MPI path to give meaningful results.

---

## Step 2 — Design choices

### Why MPI rather than `multiprocessing` or `joblib`?

The user asked for MPI explicitly. There are also real reasons for that
choice:

* The mid-term aim is presumably to scale beyond one node (HPC use case).
  `mpi4py` plus `mpirun` carries over to a cluster directly.
* The trial-evaluation loop has a non-trivial reduction (track best ene
  globally + flag whether early-exit is reached). MPI's collective ops
  (`gather`, `bcast`) handle this naturally.
* `concurrent.futures.ProcessPoolExecutor` would have worked for a single
  node but would have required reinventing the reduction logic and would
  not have been a useful scaling path.

### Why preserve a serial fallback?

* The existing test suite, the published CLI, and the Sphinx docs all
  assume serial behaviour. Breaking them was not acceptable.
* On small molecules MPI overhead can be *larger* than the parallelizable
  work. The serial path stays the default.

This is why `mpi_utils.get_mpi()` returns `(None, 0, 1)` when `mpi4py`
cannot be imported or the process was not launched under `mpirun`, and why
every parallel branch is guarded by `size > 1`.

### Strided vs. block decomposition for trial enumeration

The `seq_one_beads` list is generated by `itertools.combinations` and the
acceptance fraction of `check_beads` varies along that index range
(rejections cluster). Two options:

1. **Block** — give rank `r` indices `[r * len/size, (r+1) * len/size)`.
2. **Strided** — give rank `r` indices `[r, r+size, r+2*size, …]`.

We picked **strided**, which spreads "easy" and "hard" trials evenly across
ranks and avoids the load imbalance we'd otherwise have. The per-iteration
gather is on lists of accepted trials only, which are small; the strided
pattern doesn't penalise memory access here because the underlying state
(molecule, conformer) is replicated everywhere.

### Determinism under MPI

`Cg_molecule.__init__` calls `AllChem.EmbedMolecule(molecule)` with no
seed. We considered two fixes:

* Pass a fixed `randomSeed=` argument. This changes upstream behaviour.
* **Have rank 0 do the embedding and broadcast the resulting RDKit `Mol`.**
  RDKit's `Mol` pickles include conformers, so `comm.bcast` does the job.
  This was the chosen approach.

### MPI distribution of ALOGPS — *rejected*

The first version distributed HTTP calls across ranks the same way it
distributed trials. Benchmarking showed it was actively bad:
`4 ranks × 8 threads = 32 concurrent requests` from one IP triggered
rate-limiting on `vcclab.org`, making the batch run ~5× **slower** than
`1 rank × 8 threads`.

The final design **issues all HTTP from rank 0 with a `ThreadPoolExecutor`
and broadcasts the results.** This gave a consistent ~2× speed-up
regardless of MPI rank count and stayed friendly to the public service.
The rejected version and the reasoning are kept in a comment block in
`prefetch_alogps`.

### Caching strategy

* `_LOGP_FILE_CACHE: dict[filename, dict[smi, logp]]` — loads the local
  database once per filename.
* `_ALOGPS_CACHE: dict[(query_smi, is_mol), (ΔG, origin)]` — collapses
  duplicate calls across:
  * trial-mode vs. final-mode passes of `print_atoms`
  * multiple `attempt` iterations of the solver loop
  * `check_additivity`'s whole-molecule call
* `prefetch_alogps` populates this cache; subsequent calls to
  `smi2alogps` are O(1) cache hits.

---

## Step 3 — Implementation

The patches landed in five files; see
[PARALLELIZATION.md §3](PARALLELIZATION.md#3-files-changed--added) for the
table. The key invariants we preserved are:

* The serial code path is **bit-identical** to the pre-fork behaviour.
* The MPI path produces topologies with **identical bead types,
  partitioning, bonds, angles, and dihedrals** to the serial path.
* Disk writes happen only on rank 0. Stdout chatter is rank-0-only.
* No external dependencies are required for the serial path. `mpi4py` is
  optional.

---

## Step 4 — Benchmark methodology

Two micro-benchmarks were added in `benchmarks/`:

* `bench_find_bead_pos.py` times *only* the trial-enumeration loop on a
  curated list of small/medium molecules ranging from propane (3 heavy
  atoms) to anthracene (14) and ibuprofen (15). It barriers before and
  after the call so MPI timing is fair.
* `bench_alogps.py` deliberately points the local DB at an empty file so
  every fragment goes through HTTP. It compares old-style sequential
  `smi2alogps` calls against `prefetch_alogps`.

End-to-end timing was also taken with the standard `time python -m
auto_martiniM3 --smi … --mol …` invocation for sanity.

The headline numbers are in
[PARALLELIZATION.md §4](PARALLELIZATION.md#4-measured-speed-ups).

---

## Lessons learned during the session

* **Profile by reading first.** The two hot-spots were obvious from a
  10-minute read of `solver.py`, `optimization.py`, and `topology.py`.
  Spinning up `cProfile` and waiting for a multi-minute run was
  unnecessary.
* **Bit-identical output is a strong test.** Diffing the serial and the
  4-rank `.itp` files (modulo the user-supplied `--mol` column) gave us
  high confidence that the parallel path had not silently changed
  results.
* **Naïve MPI distribution is not always a win.** The ALOGPS rate-limit
  experiment is a useful reminder that the right granularity depends on
  the bottleneck. We landed on threads + bcast, not ranks + allgather.
* **Cache aggressively for I/O-bound code.** Just loading
  `logP_smi.dat` once and memoising the HTTP responses removed a
  large fraction of the HTTP traffic on its own, before any
  parallelism was added.
* **Keep the serial path as the default.** MPI overhead can dominate for
  small inputs; making the user opt in with `mpirun` keeps the default
  experience fast and predictable.

---

## Chat-style summary of the working session

The session moved through roughly the following exchanges. This is a
condensed paraphrase, not a verbatim transcript.

> **User**: AutoMartiniM3 is single-threaded and slow on larger molecules.
> Find parallelizable steps, make them MPI-compatible, and benchmark.
>
> **Claude Code** (after reading the package): The dominant cost is
> `optimization.find_bead_pos` — it enumerates C(N, k) trial combinations
> per bead count and scores each one with O(N²) RDKit calls. Trials are
> independent → ideal for MPI. Secondary cost: ALOGPS HTTP queries in
> `topology.smi2alogps`. Plan: add an `mpi_utils` shim with a serial
> fallback; parallelize the trial loop with a strided decomposition; have
> rank 0 do the stochastic conformer embedding and broadcast the Mol.
>
> **(Implementation, install of `mpi4py`, test runs.)**
>
> **Claude Code**: 4-rank MPI gives a 4.4× speed-up on ibuprofen's
> trial loop. Serial path unchanged. Outputs match.
>
> **User**: Explain ALOGPS in detail and where parallelization could go.
>
> **Claude Code**: Walks through `smi2alogps` and `print_atoms`. Notes
> the per-call re-reading of `logP_smi.dat` and the sequential HTTP loop
> over beads. Identifies prefetch + cache as the highest-impact fix.
>
> **User**: Implement the ALOGPS parallelization and fix the
> `logP_smi.dat` loading.
>
> **Claude Code**: Adds module-level caches for the file and the results,
> writes `prefetch_alogps` with `ThreadPoolExecutor` plus MPI-rank
> distribution, wires it into `print_atoms`. Initial benchmarking shows
> 4-rank MPI distribution is **slower** than 1-rank-with-threads due to
> vcclab.org rate-limiting (32 concurrent connections from one IP).
> Pivots design: rank 0 does all HTTP with threads, broadcasts results.
> Final benchmark shows clean 2.1× speed-up independent of rank count.
>
> **User**: Document everything in the README and add a development log.
>
> **(This document.)**

---

## Acknowledgement

This fork's parallelization work was completed by **Izar Scharf** with the
assistance of Claude Code (Anthropic). The upstream Auto_MartiniM3 toolkit
and its scientific content are the work of Magdalena Szczuka, Tristan
Bereau, Kiran Kanekal, and Andrew Abi-Mansour, under the supervision of
Matthieu Chavent, Pierre Poulain, and Paulo C. T. Souza. See
[README.md](README.md#developers) for full credits.
