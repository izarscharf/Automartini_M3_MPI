Auto_MartiniM3
============

> ## :rocket: About this fork
>
> This repository is a **fork of [Auto_MartiniM3](https://github.com/Martini-Force-Field-Initiative/Automartini_M3)**
> created to **improve performance on larger molecules through parallelization**.
> The upstream code runs single-threaded and becomes very slow as the number of
> heavy atoms grows, because the bead-placement search enumerates
> `C(N_heavy, k)` trial combinations per bead count and scores each one
> independently. This fork adds an MPI-based parallel path for that search and
> a thread-pooled prefetch for the ALOGPS HTTP queries.
>
> The implementation work was carried out **with the help of Claude Code
> (Anthropic)** under interactive human guidance. See
> [PARALLELIZATION.md](PARALLELIZATION.md) for the technical writeup and
> [CLAUDE_DEVELOPMENT_LOG.md](CLAUDE_DEVELOPMENT_LOG.md) for the development
> log (problem analysis, design decisions, the chat transcript that produced
> the patches, and benchmark numbers).
>
> **Quick taste of the speed-ups** (measured on the local `bench_find_bead_pos.py`
> and `bench_alogps.py` micro-benchmarks):
>
> | Hotspot                 | Before  | After (parallel) |
> |-------------------------|---------|------------------|
> | `find_bead_pos` (ibuprofen, 15 heavy atoms) | 4.43 s | **1.01 s** with 4 MPI ranks (4.4×) |
> | `find_bead_pos` (caffeine, 14 heavy atoms)  | 3.06 s | **0.72 s** with 4 MPI ranks (4.2×) |
> | ALOGPS HTTP batch (17 fragments)            | 5.7 s  | **2.7 s** with thread-pooled prefetch (2.1×) |
>
> **Bug fix 1 — explicit-hydrogen / deuterium molecules:** The upstream
> `voronoi_atoms_new` / `voronoi_atoms_old` functions keyed `partitioning` on
> local heavy-atom indices (0..N-1) while all downstream callers expected global
> RDKit atom indices. For molecules with explicit `[2H]` labels this diverges,
> causing a `KeyError` crash or silently wrong bead assignments. Both functions
> now return global-indexed partitioning. Deuterated SDS runs correctly in both
> serial and MPI modes.
>
> **Bug fix 2 — OOM kill on larger molecules (≥ 20 heavy atoms):**
> `find_bead_pos` materialised the full `itertools.combinations` array with
> `np.array(list(...))` **on every MPI rank simultaneously**. For a 28-atom
> molecule this reaches 1–4.5 GB *per rank* before the OS OOM-killer fires
> (`SIGKILL`). Additionally, intermediate `list_combs` / `list_energies`
> accumulators were gathered across ranks but **never read** — dead code burning
> both RAM and MPI bandwidth. Fix: replaced the materialisation with a lazy
> `itertools.islice`-strided iterator (O(1) live memory per rank); removed the
> dead accumulators and their `comm.gather` calls. With the fix, 4 ranks running
> DOC (28 heavy atoms) hold ~90 MB each instead of crashing at ~4+ GB.
>
> Run the parallel pipeline with:
> ```bash
> mpirun -n 4 python -m auto_martiniM3 --smi "<SMILES>" --mol NAME
> ```
> Outputs are bit-identical to the serial path (verified by diffing the
> generated `.itp` files). Without `mpirun` the code transparently falls
> back to the original single-process behaviour.

## What is Auto_MartiniM3?

A toolkit that enables automatic generation of Martini force field for small organic molecules up to 25 heavy atoms, now in agreement with Martini 3 Force Field parameters. 
For details see the publication [here](https://doi.org/10.1021/acs.jctc.5c01178).

## Developers
* Magdalena Szczuka (University Toulouse 3, France)
* Tristan Bereau (University of Amsterdam, Netherlands)   
* Kiran Kanekal (Max Planck Institute for Polymer Research, Mainz, Germany)     
* Andrew Abi-Mansour (Molecular Sciences Software Institute, Virginia Tech, Blacksburg, US)

## Supervisors
* Matthieu Chavent (Centre de Biologie Intégrative (CBI)), University Toulouse 3, CNRS, France)
* Pierre Poulain (Université Paris Cité, France)
* Paulo C. T. Souza (Laboratoire de Biologie et Modélisation de la Cellule, UMR 5239, ENS Lyon, France)

## Installation with conda

 For enabling automatic mapping with `Auto-MartiniM3`, you need to clone this repository and create a conda environment.

```bash
git clone https://github.com/Martini-Force-Field-Initiative/Automartini_M3.git
cd Automartini_M3
conda env create -f environment.yaml
```

This will create a conda environment called `automartiniM3` which you can activate with

```bash
conda activate automartiniM3
```

## Testing

To run the test cases and validate your installation, you will need to have [pytest](https://docs.pytest.org/en/stable/getting-started.html) 
installed. If you installed `auto_martiniM3` with conda, then pytest should already be available in your environment.

To initiate the testing, run the following:
```bash
pytest -v tests
```

All tests should pass within few minutes. If any of the tests fail, please open an [issue](https://github.com/Martini-Force-Field-Initiative/Automartini_M3/issues).

## Command-line Interface
You can invoke `auto_martiniM3` from the command-line via:
```
python -m auto_martiniM3 [mode] [options]
```
By default, mode is set to 'run', which computes the MARTINI 3 force field for a given molecule.

To display the usage-information (help), either supply -h, --help, or nothing to auto_martiniM3:
 
```
usage: auto_martiniM3 [-h] [--mode {run}] [--sdf SDF | --smi SMI] [--logp LOGP] [--mol MOLNAME] [--aa AA] [-v]
                      [--fpred] [--bartender] [--simple] [--canon] [--ndx] [--map]

Generates Martini 3 force field for atomistic structures of small organic molecules

optional arguments:
  -h, --help     show this help message and exit
  --mode {run}   mode: run (compute FF)
  --sdf SDF      SDF file of atomistic coordinates
  --smi SMI      SMILES string of atomistic structure
  --logp LOGP    File with partial smiles and associated logP
  --mol MOLNAME  Name of CG molecule
  --aa AA        filename of all-atom structure .gro file
  -v, --verbose  increase verbosity
  --fpred        Atomic partitioning prediction
  --bartender    Bartender input file
  --simple       Simple model without dihedrals nor virtual sites
  --canon        Translate to RdKit canon structure
  --ndx          Write GROMACS index file (MOL.ndx) — one group per CG bead
  --map          Write VOTCA-CSG XML mapping file (MOL.map) for Fast-Forward

Developers:
===========
Magdalena Szczuka (magdalena.szczuka [at] univ-tlse3.fr)
Tristan Bereau (bereau [at] mpip-mainz.mpg.de)
Kiran Kanekal (kanekal [at] mpip-mainz.mpg.de)
Andrew Abi-Mansour (andrew.gaam [at] gmail.com)
```

## Example
To coarse-grain a molecule, simply provide its SMILES code (option `--smi SMI`) or a .SDF file (option `'--sdf file.sdf`). You also need to provide a name for the CG molecule (not longer than 5 characters) using the `--mol` option.  For instance, to coarse grain [aspirin](https://pubchem.ncbi.nlm.nih.gov/compound/2244#section=2D-Structure), you can either obtain/generate (e.g., from Open Babel) an SDF file:
```
python -m auto_martiniM3 --sdf aspirin.sdf --mol ASP 
```
(the name ASP is arbitrary) or use its SMILES code within double quotes
```
python -m auto_martiniM3 --smi "CC(=O)OC1=CC=CC=C1C(=O)O" --mol ASP 
```
In case no problem arises, it will output the gromacs ASP.itp file:
```
; GENERATED WITH Auto_Martini M3FF for ASP
; Developed by: Kiran Kanekal, Tristan Bereau, and Andrew Abi-Mansour
; updated to Martini 3 force field by Magdalena Szczuka
; supervised by Matthieu Chavent, Pierre Poulain and Paulo C. T. Souza 
; SMILES code : CC(=O)OC1=CC=CC=C1C(=O)O


[moleculetype]
; molname       nrexcl
  ASP          1

[atoms]
; id      type   resnr residue atom    cgnr    charge  mass ;  smiles    ; atom_num
   1       SN5a    1   ASP     N01       1        0    54   ;   CC=O     ; atoms: C0, C1, O2,          
   2       TP2a    1   ASP     P01       2        0    36   ;   CO       ; atoms: O3, C4,          
   3       TC5     1   ASP     C01       3        0    36   ;   C=C      ; atoms: C5, C6,          
   4       SC5     1   ASP     C02       4        0    54   ;   CC=C     ; atoms: C7, C8, C9,          
   5       SN6d    1   ASP     N02       5        0    54   ;   O=CO     ; atoms: C10, O11, O12, ; ALOGPS defined bead


[bonds]
;  i   j     funct   length   force.c.
   1   2     1       0.27       25000.00
   2   3     1       0.27       25000.00
   2   4     1       0.33       10000.00
   3   4     1       0.28       100000.00
   4   5     1       0.35       5000.00
#ifndef FLEXIBLE
[constraints]
#endif
;  i   j     funct   length

[angles]
;  i  j  k    funct  angle  force.c.
   1  2  5       1    95.8   100.0
   1  4  5       1    52.7    25.0

[dihedrals]
;  i  j  k  l  funct  angle  force.c.
   1  2  3  4    2    -135.5   25.0   
   2  3  4  5    2    -0.5     25.0   

[exclusions]
  1 4
```
The code will also output a corresponding `.gro` file for the coarse-grained coordinates.
Atomistic coordinates can be written using the `--aa output.gro` option.

If `--bartender` flag is used, additional file for further optimization of bonded parameters with Bartender [Pereira et al., 2024](https://pubs.acs.org/doi/10.1021/acs.jctc.4c00275) will be produced. You will find more information about Bartender in the [official tutorial](https://github.com/Martini-Force-Field-Initiative/Bartender.git). Examplary Bartender input file, created by Auto-MartiniM3 for aspirin and saved as ASP_bartender.inp :
```
# INPUT data for bonded parameter definition by BARTENDER for molecule ASP
BEADS
1 1,2,3,14,15,16
2 4,5
3 6,7,17,18
4 8,9,10,19,20
5 11,12,13,21
BONDS
1,2
2,3
2,4
3,4
4,5
ANGLES
1,2,5
1,4,5
IMPROPERS
1,2,3,4
2,3,4,5
```
## Fast-Forward pipeline outputs (fork addition)

This fork adds two extra output files that feed directly into the
[Fast-Forward](https://github.com/Martini-Force-Field-Initiative/Bartender)
/ VOTCA-CSG workflow for validating CG bonded parameters against a reference
AA simulation.

### `--ndx` — GROMACS index file

Writes `MOL.ndx` with one named group per CG bead containing the 1-based
heavy-atom serial numbers that match the AA `.gro` file (produced by `--aa`).
A trailing `[ System ]` group covers all heavy atoms.

```bash
python -m auto_martiniM3 --smi "CC(=O)OC1=CC=CC=C1C(=O)O" --mol ASP --aa ASP_aa.gro --ndx
```

Example `ASP.ndx`:
```
[ N01 ]
     1     2     3
[ P01 ]
     4     5
[ C01 ]
     6     7
[ C02 ]
     8     9    10
[ N02 ]
    11    12    13
[ System ]
     1     2     3     4     5     6     7     8     9    10    11    12    13
```

Use with GROMACS to extract per-bead centre-of-geometry positions from an AA trajectory:
```bash
gmx traj -f aa.xtc -n ASP.ndx -ox bead_cog.xvg -com -ng 5
```

### `--map` — VOTCA-CSG XML mapping file

Writes `MOL.map` in VOTCA-CSG format. Each `<cg_bead>` element lists the
Martini 3 bead type, a `COG` (centre-of-geometry) operator, and the heavy
atoms that map to it. Atom labels use the `RES:ElementGlobalIdx` convention
(e.g. `ASP:C0`) which cross-references the `; atoms:` column already in the
`.itp`.

```bash
python -m auto_martiniM3 --smi "CC(=O)OC1=CC=CC=C1C(=O)O" --mol ASP --map
```

Example `ASP.map` (fragment):
```xml
<cg_molecule>
  <name>ASP</name>
  <ident>ASP</ident>
  <topology>
    <cg_beads>
      <cg_bead>
        <name>N01</name>
        <type>SN5a</type>
        <mapping>COG</mapping>
        <beads>ASP:C0 ASP:C1 ASP:O2</beads>
      </cg_bead>
      ...
    </cg_beads>
  </topology>
</cg_molecule>
```

Use with `csg_map` to back-map an AA trajectory to CG coordinates:
```bash
csg_map --top aa.tpr --trj aa.xtc --out cg.xtc --cg ASP.map
```

Both flags can be combined with MPI:
```bash
mpirun -n 4 python -m auto_martiniM3 --smi "..." --mol ASP --aa ASP_aa.gro --ndx --map
```

### `itp_to_map.py` — convert an existing `.itp` without re-running

If you have already computed the CG topology and do not want to re-run
`auto_martiniM3` (e.g. for a large molecule), use the standalone converter.
It parses only the annotated `.itp` text — no RDKit or AutoMartiniM3
installation required.

```bash
python3 itp_to_map.py DOC.itp                      # writes DOC.ndx and DOC.map
python3 itp_to_map.py DOC.itp --out-dir /ff_run/   # write to a different directory
python3 itp_to_map.py DOC.itp --mol DOCX            # override molecule name
```

The converter reads the `; atoms: C0, C1, O2,` annotation on each `[atoms]`
line (written by AutoMartiniM3 since the original release) to reconstruct the
full bead-to-atom mapping. Output is byte-identical to what `--ndx` / `--map`
produce when running `auto_martiniM3` directly.

## Parallel execution (fork addition)

This fork adds MPI parallelism around the expensive bead-placement search and
thread-pooled batching around the ALOGPS HTTP queries. To use it:

```bash
# Install the extra dependency once (already covered if you reinstall with `pip install -e .`):
pip install mpi4py

# Then launch with mpirun. Pick a rank count up to (but not above) the number
# of physical cores on the machine. 4 is the sweet spot on a single 8-core box.
mpirun -n 4 python -m auto_martiniM3 --smi "<SMILES>" --mol NAME
```

The serial path is untouched — calling `python -m auto_martiniM3 ...` without
`mpirun` runs exactly as it always did. Internally the code asks `mpi4py`
whether it is running under an MPI launcher and only activates the parallel
paths when `size > 1`.

Two micro-benchmarks under `benchmarks/` cover the parallelized hot-spots:

```bash
# Bead-placement search (the dominant cost for medium/large molecules):
mpirun -n 4 python benchmarks/bench_find_bead_pos.py

# ALOGPS prefetch (the dominant cost when many bead fragments are not in
# the local logP_smi.dat database):
mpirun -n 4 python benchmarks/bench_alogps.py
```

See [PARALLELIZATION.md](PARALLELIZATION.md) for the design notes and
[CLAUDE_DEVELOPMENT_LOG.md](CLAUDE_DEVELOPMENT_LOG.md) for the development
history.

## Caveats

For frequently encountered problems, see [FEP](FEP.md).

