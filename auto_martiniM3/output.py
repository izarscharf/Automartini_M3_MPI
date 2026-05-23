"""
Created on March 13, 2019 by Andrew Abi-Mansour
Updated to Martini 3 force field on January 31, 2025 by Magdalena Szczuka

This is the::
    _   _   _ _____ ___     __  __    _    ____ _____ ___ _   _ ___   __  __ _____
   / \ | | | |_   _/ _ \   |  \/  |  / \  |  _ \_   _|_ _| \ | |_ _|  |  \/  |___ /  
  / _ \| | | | | || | | |  | |\/| | / _ \ | |_) || |  | ||  \| || |   | |\/| | |_ \  
 / ___ \ |_| | | || |_| |  | |  | |/ ___ \|  _ < | |  | || |\  || |   | |  | |___) | 
/_/  _\_\___/  |_| \___/   |_|  |_/_/   \_\_| \_\|_| |___|_| \_|___|  |_|  |_|____/    
                                                

A tool for automatic MARTINI 3 force field mapping and parametrization of small organic molecules

Developers::
        Magdalena Szczuka (magdalena.szczuka at univ-tlse3.fr)
        Tristan BEREAU (bereau at mpip-mainz.mpg.de)
        Kiran Kanekal (kanekal at mpip-mainz.mpg.de)
        Andrew Abi-Mansour (andrew.gaam at gmail.com)

AUTO_MARTINI M3 is open-source, distributed under the terms of the GNU Public
License, version 2 or later. It is distributed in the hope that it will
be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. You should have
received a copy of the GNU General Public License along with PyGran.
If not, see http://www.gnu.org/licenses . See also top-level README
and LICENSE files.
"""

from .common import *

logger = logging.getLogger(__name__)

from sys import exit


def output_ndx(list_heavy_atoms, atom_partitioning, cg_bead_names):
    """Write a GROMACS index file (.ndx) — one group per CG bead.

    Atom indices are 1-based serial numbers in the heavy-atom AA .gro file
    produced by output_aa (i.e. position in list_heavy_atoms + 1).  The file
    also contains a 'System' group covering all heavy atoms so it can be used
    directly as the index file for gmx traj / gmx rdf.
    """
    n_beads = len(cg_bead_names)
    lines = []

    for bead_idx in range(n_beads):
        indices = [
            local + 1
            for local, gidx in enumerate(list_heavy_atoms)
            if atom_partitioning.get(int(gidx)) == bead_idx
        ]
        lines.append(f"[ {cg_bead_names[bead_idx]} ]")
        for start in range(0, len(indices), 15):
            lines.append("  " + "  ".join(f"{i:4d}" for i in indices[start:start + 15]))

    # Convenience group containing every heavy atom
    lines.append("[ System ]")
    all_idx = list(range(1, len(list_heavy_atoms) + 1))
    for start in range(0, len(all_idx), 15):
        lines.append("  " + "  ".join(f"{i:4d}" for i in all_idx[start:start + 15]))

    return "\n".join(lines) + "\n"


def output_map(list_heavy_atoms, atom_partitioning, cg_bead_names, bead_types,
               molecule, molname, smiles):
    """Write a CGBuilder-format mapping file (.map) — default format.

    Sections:
      [molecule]   molecule name
      [ martini ]  space-separated list of CG bead names
      [ atoms ]    one line per heavy atom: serial  atomname  beadname

    Atom names use the convention ElementGlobalIdx (e.g. C0, O3) which
    matches the '; atoms:' column in the .itp.  Serial numbers are 1-based
    positions in the heavy-atom .gro file and match the .ndx file.

    Compatible with the Fast-Forward / CGBuilder / vermouth-martinize pipeline.
    """
    # Build (serial, atomname, beadname) triples sorted by serial
    entries = []
    for bead_idx, bead_name in enumerate(cg_bead_names):
        for local_i, gidx in enumerate(list_heavy_atoms):
            if atom_partitioning.get(int(gidx)) == bead_idx:
                serial = local_i + 1
                symbol = molecule.GetAtomWithIdx(int(gidx)).GetSymbol()
                atom_name = f"{symbol}{int(gidx)}"
                entries.append((serial, atom_name, bead_name))
    entries.sort(key=lambda x: x[0])

    lines = [
        "[molecule]",
        molname,
        "",
        "[ martini ]",
        " ".join(cg_bead_names),
        "",
        "[ atoms ]",
    ]
    for serial, atom_name, bead_name in entries:
        lines.append(f"{serial:<6} {atom_name:<6} {bead_name}")
    lines.append("")
    return "\n".join(lines)


def output_map_votca(list_heavy_atoms, atom_partitioning, cg_bead_names, bead_types,
                     molecule, molname, smiles):
    """Write a VOTCA-CSG XML mapping file — alternative format via --map-votca.

    Each <cg_bead> lists the atoms using RES:ElementGlobalIdx labels (e.g.
    ASP:C0) with a COG (centre-of-geometry) mapping operator.

    Use with:  csg_map --top aa.tpr --trj aa.xtc --out cg.xtc --cg MOL.map
    """
    res = molname[:4]
    lines = [
        '<?xml version="1.0"?>',
        f'<!-- AutoMartiniM3 VOTCA mapping file for {molname} -->',
        f'<!-- SMILES: {smiles} -->',
        f'<!-- Use with: csg_map --top aa.tpr --trj aa.xtc --out cg.xtc --cg {molname}.map -->',
        '<cg_molecule>',
        f'  <name>{molname}</name>',
        f'  <ident>{molname}</ident>',
        '  <topology>',
        '    <cg_beads>',
    ]

    n_beads = len(cg_bead_names)
    for bead_idx in range(n_beads):
        btype = bead_types[bead_idx] if bead_idx < len(bead_types) else "?"
        atom_labels = []
        for gidx in list_heavy_atoms:
            if atom_partitioning.get(int(gidx)) == bead_idx:
                symbol = molecule.GetAtomWithIdx(int(gidx)).GetSymbol()
                atom_labels.append(f"{res}:{symbol}{int(gidx)}")
        lines += [
            '      <cg_bead>',
            f'        <name>{cg_bead_names[bead_idx]}</name>',
            f'        <type>{btype}</type>',
            '        <mapping>COG</mapping>',
            f'        <beads>{" ".join(atom_labels)}</beads>',
            '      </cg_bead>',
        ]

    lines += [
        '    </cg_beads>',
        '  </topology>',
        '</cg_molecule>',
        '',
    ]
    return "\n".join(lines)


def output_gro(sites, site_names, molname):
    """Output GRO file of CG structure"""
    logger.debug("Entering output_gro()")
    num_beads = len(sites)
    gro_out = ""
    if len(sites) != len(site_names):
        logger.warning("Error. Incompatible number of beads and bead names.")
        exit(1)
    gro_out += "{:s} generated from auto_martiniM3\n".format(molname)
    gro_out += "{:5d}\n".format(num_beads)
    if len(molname)>4:molname=molname[:4]
    for i in range(num_beads):
        gro_out += "{:5d}{:<6s} {:3s}{:5d}{:8.3f}{:8.3f}{:8.3f}\n".format(
            1, #was i +1, but this is GRO file for one molecule, so all beads should be a part of the same molecule
            molname,
            site_names[i],
            i + 1,
            sites[i][0] / 10.0,
            sites[i][1] / 10.0,
            sites[i][2] / 10.0,
        )
    gro_out += "{:10.5f}{:10.5f}{:10.5f}\n".format(10.0, 10.0, 10.0)
    return gro_out
