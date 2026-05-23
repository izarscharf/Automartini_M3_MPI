#!/usr/bin/env python3
"""
itp_to_map.py
=============
Convert an AutoMartiniM3-annotated .itp file to:

  <mol>.map   CGBuilder-format mapping file for Fast-Forward (default)
  <mol>.ndx   GROMACS index file for gmx traj, gmx rdf, etc.

No RDKit or AutoMartiniM3 installation required — only the .itp text is parsed.
The ''; atoms:'' column already written into the [atoms] section carries all
the bead-to-atom information needed to reconstruct both files.

Usage
-----
    python3 itp_to_map.py ASP.itp                    # CGBuilder format (default)
    python3 itp_to_map.py ASP.itp --votca             # VOTCA-CSG XML format
    python3 itp_to_map.py ASP.itp --mol MYNAME
    python3 itp_to_map.py ASP.itp --out-dir /path/to/outdir

Output
------
  MOL.ndx
      One [ BeadName ] group per CG bead with 1-based heavy-atom serial
      numbers that match the heavy-atom .gro file produced by AutoMartiniM3
      (--aa flag).  A trailing [ System ] group covers all heavy atoms.
      Use with:  gmx traj -f aa.xtc -n MOL.ndx -ox bead_pos.xvg

  MOL.map
      VOTCA-CSG XML mapping with COG (centre-of-geometry) operator.
      Atom labels have the form  RES:ElementGlobalIdx  (e.g. ASP:C0)
      which matches the '; atoms:' comment column in the .itp.
      Use with:  csg_map --top aa.tpr --trj aa.xtc --out cg.xtc --cg MOL.map
"""

import argparse
import re
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_itp(itp_path: Path):
    """Return (molname, smiles, beads) from an AutoMartiniM3 .itp file.

    beads: list of dicts with keys 'id', 'type', 'name', 'atoms'
           where 'atoms' is a list of 'ElementGlobalIdx' strings (e.g. ['C0','O3'])
    """
    text = itp_path.read_text()
    lines = text.splitlines()

    # -- SMILES from header comment ------------------------------------------
    smiles = "unknown"
    for line in lines:
        m = re.match(r';\s*SMILES[^:]*:\s*(.+)', line, re.IGNORECASE)
        if m:
            smiles = m.group(1).strip()
            break

    # -- Molecule name from [moleculetype] -----------------------------------
    molname = itp_path.stem
    in_moltype = False
    for line in lines:
        stripped = line.strip()
        if re.match(r'\[\s*moleculetype\s*\]', stripped, re.IGNORECASE):
            in_moltype = True
            continue
        if in_moltype:
            if stripped.startswith('['):
                break
            if stripped and not stripped.startswith(';'):
                molname = stripped.split()[0]
                break

    # -- Bead data from [atoms] ----------------------------------------------
    beads = []
    in_atoms = False
    for line in lines:
        stripped = line.strip()

        if re.match(r'\[\s*atoms\s*\]', stripped, re.IGNORECASE):
            in_atoms = True
            continue
        if in_atoms and stripped.startswith('['):
            break
        if not in_atoms or not stripped or stripped.startswith(';'):
            continue

        # Split the line at ';' boundaries:
        #   fields part    ; smiles frag ; atoms: C0, C1, O2, [; ALOGPS ...]
        parts = stripped.split(';')
        fields = parts[0].split()
        if len(fields) < 5:
            continue

        bead_id   = int(fields[0])
        bead_type = fields[1]
        bead_name = fields[4]   # 'atom' column

        # Find the 'atoms: ...' segment anywhere in the comment tail
        comment_tail = ';'.join(parts[1:])
        m = re.search(r'atoms:\s*([^;]+)', comment_tail)
        if not m:
            print(f"  WARNING: no '; atoms:' found for bead {bead_name} — skipping",
                  file=sys.stderr)
            continue

        raw = m.group(1).strip().rstrip(',')
        atom_tokens = [t.strip() for t in raw.split(',') if t.strip()]

        # Validate each token looks like an element + integer
        valid = []
        for tok in atom_tokens:
            if re.match(r'^[A-Za-z]+\d+$', tok):
                valid.append(tok)
            else:
                print(f"  WARNING: unrecognised atom token '{tok}' in bead "
                      f"{bead_name} — skipping token", file=sys.stderr)
        if not valid:
            continue

        beads.append({
            'id':    bead_id,
            'type':  bead_type,
            'name':  bead_name,
            'atoms': valid,
        })

    if not beads:
        sys.exit(f"ERROR: no parseable [atoms] entries found in {itp_path}")

    return molname, smiles, beads


# ---------------------------------------------------------------------------
# Index building
# ---------------------------------------------------------------------------

def build_serial_map(beads):
    """Map each 'ElementGlobalIdx' token to its 1-based GRO serial number.

    Heavy atoms appear in the AA .gro file in ascending global-index order.
    Collecting all unique global indices across every bead and sorting them
    reproduces that ordering, giving each atom its correct 1-based serial.
    """
    global_to_label = {}
    for bead in beads:
        for tok in bead['atoms']:
            m = re.match(r'[A-Za-z]+(\d+)$', tok)
            if m:
                gidx = int(m.group(1))
                global_to_label[gidx] = tok

    serial_map = {
        label: pos + 1
        for pos, (_, label) in enumerate(sorted(global_to_label.items()))
    }
    return serial_map


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def write_ndx(beads, serial_map, out_path: Path):
    """Write GROMACS .ndx index file."""
    lines = []
    for bead in beads:
        serials = [serial_map[a] for a in bead['atoms'] if a in serial_map]
        lines.append(f"[ {bead['name']} ]")
        for start in range(0, len(serials), 15):
            chunk = serials[start:start + 15]
            lines.append("  " + "  ".join(f"{s:4d}" for s in chunk))

    all_serials = sorted(serial_map.values())
    lines.append("[ System ]")
    for start in range(0, len(all_serials), 15):
        chunk = all_serials[start:start + 15]
        lines.append("  " + "  ".join(f"{s:4d}" for s in chunk))

    out_path.write_text("\n".join(lines) + "\n")


def write_map(beads, serial_map, molname, out_path: Path):
    """Write CGBuilder-format .map file (default)."""
    bead_names = [b['name'] for b in beads]
    # Build (serial, atomname, beadname) sorted by serial
    entries = []
    for bead in beads:
        for atom in bead['atoms']:
            if atom in serial_map:
                entries.append((serial_map[atom], atom, bead['name']))
    entries.sort(key=lambda x: x[0])

    lines = [
        "[molecule]",
        molname,
        "",
        "[ martini ]",
        " ".join(bead_names),
        "",
        "[ atoms ]",
    ]
    for serial, atom_name, bead_name in entries:
        lines.append(f"{serial:<6} {atom_name:<6} {bead_name}")
    lines.append("")
    out_path.write_text("\n".join(lines))


def write_map_votca(beads, molname, smiles, out_path: Path):
    """Write VOTCA-CSG XML .map file (--votca flag)."""
    res = molname[:4]
    lines = [
        '<?xml version="1.0"?>',
        f'<!-- AutoMartiniM3 VOTCA mapping file for {molname} -->',
        f'<!-- SMILES: {smiles} -->',
        f'<!-- Use with: csg_map --top aa.tpr --trj aa.xtc --out cg.xtc --cg {out_path.name} -->',
        '<cg_molecule>',
        f'  <name>{molname}</name>',
        f'  <ident>{molname}</ident>',
        '  <topology>',
        '    <cg_beads>',
    ]
    for bead in beads:
        bead_atoms = " ".join(f"{res}:{a}" for a in bead['atoms'])
        lines += [
            '      <cg_bead>',
            f'        <name>{bead["name"]}</name>',
            f'        <type>{bead["type"]}</type>',
            '        <mapping>COG</mapping>',
            f'        <beads>{bead_atoms}</beads>',
            '      </cg_bead>',
        ]
    lines += [
        '    </cg_beads>',
        '  </topology>',
        '</cg_molecule>',
        '',
    ]
    out_path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Convert an AutoMartiniM3 .itp to CGBuilder .map and GROMACS .ndx",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("itp", help="AutoMartiniM3 .itp file")
    ap.add_argument("--mol", dest="molname", default=None,
                    help="Override molecule name (default: read from .itp)")
    ap.add_argument("--out-dir", dest="out_dir", default=None,
                    help="Output directory (default: same directory as .itp)")
    ap.add_argument("--votca", dest="votca", action="store_true",
                    help="Write VOTCA-CSG XML format instead of CGBuilder format")
    args = ap.parse_args()

    itp_path = Path(args.itp)
    if not itp_path.exists():
        sys.exit(f"ERROR: file not found: {itp_path}")

    out_dir = Path(args.out_dir) if args.out_dir else itp_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Parsing {itp_path} ...")
    molname, smiles, beads = parse_itp(itp_path)
    if args.molname:
        molname = args.molname

    serial_map = build_serial_map(beads)
    print(f"  Molecule : {molname}")
    print(f"  Beads    : {len(beads)}")
    print(f"  Heavy atoms: {len(serial_map)}")
    print(f"  SMILES   : {smiles}")
    fmt = "VOTCA XML" if args.votca else "CGBuilder"
    print(f"  Map format : {fmt}")

    ndx_path = out_dir / f"{molname}.ndx"
    map_path = out_dir / f"{molname}.map"

    write_ndx(beads, serial_map, ndx_path)
    print(f"  Written  : {ndx_path}")

    if args.votca:
        write_map_votca(beads, molname, smiles, map_path)
    else:
        write_map(beads, serial_map, molname, map_path)
    print(f"  Written  : {map_path}")

    print("Done.")


if __name__ == "__main__":
    main()
