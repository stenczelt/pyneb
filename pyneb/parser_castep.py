#!/bin/env python3

import numpy as np
import os, sys, warnings
import ase
import copy
from time import time
from pyneb.structureformat import supercell
import warnings
import ase.io


def deprecate(fun):
    warnings.warn(
        "This function ({} > {}) will be removed in future updates!".format(
            fun.__module__, fun.__name__
        ),
        DeprecationWarning,
    )
    return fun


# CODATA2002: default in CASTEP 5.01
# (-> check in more recent CASTEP in case of numerical discrepancies?!)
# taken from
#    http://physics.nist.gov/cuu/Document/all_2002.pdf
units_CODATA2002 = {
    "hbar": 6.58211915e-16,  # eVs
    "Eh": 27.2113845,  # eV
    "kB": 8.617343e-5,  # eV/K
    "a0": 0.5291772108,  # A
    "c": 299792458,  # m/s
    "e": 1.60217653e-19,  # C
    "me": 5.4857990945e-4,
}  # u

units_CODATA2002["t0"] = units_CODATA2002["hbar"] / units_CODATA2002["Eh"]
units_CODATA2002["Pascal"] = units_CODATA2002["e"] * 1e30


def convert_gridtocartesian(gridnum, cell, invNpoints):
    """
    convert grid numbers gridnum[0:3] to cartesian positions
    """
    cart = np.zeros(3, dtype=float)

    for ia in range(3):
        for ib in range(3):
            cart[ia] += (gridnum[ib] - 1.0) * invNpoints[ib] * cell[ib][ia]
    return cart


def castep_safety_checks(lines):
    for line in lines:
        # check that all SCF cycles converged
        assert "Warning: electronic minimisation did not converge" not in line, (
            "SCF cycle has not converged for " + fd.name
        )
        # check geometry optimization has converged
        assert "WARNING - Geometry optimization failed to converge" not in line, (
            "geometry optimization has not converged for " + fd.name
        )
        # check geometry optimisation has been performed if stated
        if clc_type == "geometry optimization":
            assert (
                "WARNING - there is nothing to optimise - skipping relaxation"
                not in line
            ), ("geometry optimization has not been performed for " + fd.name)

    # check units are as expected
    supp_units = {
        "length unit": [" A\n"],
        "energy unit": [" eV\n"],
        "force unit": [" eV/A\n"],
        "pressure unit": [" GPa\n"],
    }
    E2 = "calculation unit is not supported. Supported units are eV,A,GPa."
    for line in lines:
        if "output" in line and len(line.split()) > 3:
            if line.split()[1] == "length" and line.split()[2] == "unit":
                assert line.split(":")[-1] in supp_units["length unit"], E2
            elif "energy unit" in line:
                assert line.split(":")[-1] in supp_units["energy unit"], E2
            elif "force unit" in line:
                assert line.split(":")[-1] in supp_units["force unit"], E2
            elif "pressure unit" in line:
                assert line.split(":")[-1] in supp_units["pressure unit"], E2


def _castep_castep_get_indices(clc_type, lines):

    idx = [
        iv
        for iv, v in enumerate(lines)
        if v == " +-------------------------------------------------+"
    ]
    if len(idx) > 3:
        warnings.warn("Found more than one calculation, reading only the last one...")
    # print("idx {}".format(idx))
    readstart = idx[-3]
    readend = len(lines)
    return readstart, readend, len(idx) > 3


def castep_completemeness_checks(
    fd,
    sedc_energy_total,
    sedc_free_energy_total,
    sedc_energy_0K,
    free_energy_total,
    energy_total,
    energy_0K,
    supercells,
    atoms,
    species,
    forces,
    clc_type,
):
    E1 = "unexpected behaviour in " + fd.name + " , check file for corruption"

    num_ens = [len(energy_total), len(energy_0K), len(free_energy_total)]
    num_ens_dc = [
        len(sedc_energy_total),
        len(sedc_free_energy_total),
        len(sedc_energy_0K),
    ]
    num_config = [len(forces), len(supercells), len(atoms), len(species)]
    if not len(set(num_ens)) == 1:
        raise AssertionError(
            "Mismatching numbers of energies:\n"
            + "Final energy, E ({})\n".format(len(energy_total))
            + "NB est. 0K energy (E-0.5TS) ({})\n".format(len(energy_0K))
            + "Final free energy (E-TS) ({})".format(len(free_energy_total))
        )
    if num_ens_dc[-1] > 0 and not len(set(num_ens_dc)) == 1:
        raise AssertionError(
            "Mismatching numbers of dispersion corrected energies:\n"
            + "Dispersion corrected final energy ({})\n".format(len(sedc_energy_total))
            + "Dispersion corrected final free energy ({})\n".format(
                len(sedc_free_energy_total)
            )
            + "NB dispersion corrected est. 0K energy ({})\n".format(
                len(sedc_energy_0K)
            )
        )
    if clc_type != "geometry optimization":
        if not len(set(num_config)) == 1:
            raise AssertionError(
                "Mismatching configuration:\n"
                + "Forces ({}) Supercells ({}) Atoms ({}) Species ({})".format(
                    *num_config
                )
            )


def castep_get_unit_cells(lines, idx_unit_cell):
    num = len(idx_unit_cell)
    unitcells = [[]] * num
    for i, ix in enumerate(idx_unit_cell):
        unitcells[i] = [
            [float(_c) for _c in _line.split()[0:3]] for _line in lines[ix + 3 : ix + 6]
        ]
    return unitcells


def castep_get_cell_content(lines, idx_cell_content):
    num = len(idx_cell_content)
    atoms, species = [[]] * num, [[]] * num
    for i, ix in enumerate(idx_cell_content):
        idx_tmp = []
        c = 0
        for j, line in enumerate(lines[ix:]):
            if "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" in line:
                idx_tmp.append(j)
                c += 1
            if c == 2:
                break

        config_lines = lines[ix + idx_tmp[0] + 4 : ix + idx_tmp[1]]
        config_lines = [list(filter(None, v.split(" "))) for v in config_lines]
        species[i] = [v[1] for v in config_lines]
        atoms[i] = np.array([v[3:6] for v in config_lines], dtype=float)
    return atoms, species

def read_castep_castep(fd, index=None, safety_checks=False, check_completeness=True):
    """parse a .castep file of supported calculation type.

    Parameters
    ----------
    lines: list of str

    Returns
    -------
    - the list of supercell structures appropriate to the calculation type.
        - single point energy   : a single structure
        - geometry optimization : the final optimised structure only
        - molecular dynamics    : Nsteps many structures

    Notes
    -----
    Supported calculations types:
        - single point energy
        - geometry optimisation
        - molecular dynamics

    """
    ####
    # Under which circumstances appear dispersion corrected energies?
    ####

    # supported calculation types for reading
    supported_MD_envs = set(["nvt", "nvp", "nve"])
    supp_clcs = set(
        ["single point energy", "geometry optimization", "molecular dynamics"]
    )
    lines = list(map(lambda x: x.rstrip("\n"), fd.readlines()))
    # determine calculation type
    for i, line in enumerate(lines):
        if "type of calculation" in line:
            clc_type = " ".join(list(filter(None, line.split(" ")))[4:])
            assert (
                clc_type in supp_clcs
            ), "Assertion failed - found unexpected '{}' calculation type. Expected one of {}".format(
                clc_type, supp_clcs
            )
            break
    if i == len(lines):
        raise ValueError(
            "Tough luck... corrupted file ({}) found no 'type of calculation' line!".format(
                fd.name
            )
        )
    # print("calculation type {}".format(clc_type))

    # check how many calculations are present in case of single point calculation
    run_idx = [iv for iv, v in enumerate(lines) if "Run started" in v]
    cntr = len(run_idx)

    if cntr > 1:
        warnings.warn(
            "Warning : more than 1 calculation is present in {}, are considering only the last calculation".format(
                fd.name
            )
        )
    if cntr == 0:
        raise ValueError(
            "Missing calculation or corrupted file in {}...".format(fd.name)
        )

    if safety_checks:
        castep_safety_checks(lines)

    # get indices for processing
    readstart, readend, multiple_calc = _castep_castep_get_indices(clc_type, lines)
    # print("readstart {} readend {}".format(readstart,readend))

    # create a list of material properties for each configuration
    energy_total = []  # 1. total energy
    free_energy_total = []  # 2. total free energy
    energy_0K = []  # 3. 0K total energy estimate
    sedc_energy_total = []  # 4. sedc corrected total energy
    sedc_free_energy_total = []  # 5. sedc corrected free energy
    sedc_energy_0K = []  # 6. sedc corrected 0K energy estimate
    MP_energy_total = []  # 7. Makov-Payne finite basis set corrected total energy
    supercells = []  # 8. supercell vectors in cartesian
    atoms = []  # 9. fractional coordinates of atoms
    species = []  # 10. atom types
    forces = []  # 11. atom cartesian force components /(eV/A)
    stress = []  # 12. stress tensor / (GPa)
    spacegroup = None  # 13. space group : take number and Hermann-Mauguin notation
    charge = None  # 14. atomic charge / (|e-|)
    enthalpy = []  # 15. enthalpy H = U + PV

    # for ia in range(readstart,readend):
    ia = int(readstart)
    idx_unit_cell = []
    idx_forces = []
    idx_cell_content = []
    idx_stress = []
    idx_charge = []
    num_cutoff = 0  # for some reason castep may do a small cutoff convergence test leading to additional superfluous energy values
    while ia < readend:

        line = lines[ia]
        if "Final energy, E" in line:
            energy_total.append(float(line.split()[-2]))
        elif "Final free energy (E-TS)" in line:
            free_energy_total.append(float(line.split()[-2]))
        elif "NB est. 0K energy (E-0.5TS)" in line:
            energy_0K.append(float(line.split()[-2]))
        elif "Dispersion corrected final energy" in line:
            sedc_energy_total.append(float(line.split()[-2]))
        elif "Dispersion corrected final free energy" in line:
            sedc_free_energy_total.append(float(line.split()[-2]))
        elif "NB dispersion corrected est. 0K energy" in line:
            sedc_energy_0K.append(float(line.split()[-2]))
        elif "Total energy corrected for finite basis set" in line:
            MP_energy_total.append(float(line.split()[-2]))
        elif (
            "Unit Cell" in line
        ):  # bloody hell, castep outputs multiple unit cells for NPT but only single for NVT...
            idx_unit_cell.append(ia)
        elif (
            "********** Forces ********" in line
            or line == " ***************** Symmetrised Forces *****************"
        ):
            # elif line == " *********************************** Forces ***********************************" \
            idx_forces.append(ia)
        elif "Cell Contents" in line:
            idx_cell_content.append(ia)
        elif (
            line
            == " ************************ Molecular Dynamics Parameters ************************"
        ):
            for ib, line in enumerate(lines[ia:]):
                if "ensemble" in line:
                    break
            md_env = list(filter(None, lines[ib + ia].split(" ")))[-1].lower()
        elif (
            line == " *********** Symmetrised Stress Tensor ***********"
            or line == " ***************** Stress Tensor *****************"
        ):
            idx_stress.append(ia)
        elif "Final Enthalpy" in line:
            enthalpy.append(float(line.split()[4]))
        elif "Space group of crystal" in line:
            spacegroup = {
                "number": int((line.split()[5]).split(":")[0]),
                "Hermann-Mauguin": (line.split()[6]).split(",")[0],
            }
        elif "Calculating finite basis set correction with" in line:
            num_cutoff = int(list(filter(None, line.split(" ")))[6])

        ia += 1

    # print("energy_total {} num_cutoff {}".format(len(energy_total),num_cutoff))
    if num_cutoff > 0:
        energy_total = energy_total[num_cutoff - 1 :]
        free_energy_total = free_energy_total[num_cutoff - 1 :]
        energy_0K = energy_0K[num_cutoff - 1 :]

    supercells = castep_get_unit_cells(lines, idx_unit_cell)
    atoms, species = castep_get_cell_content(lines, idx_cell_content)

    # use ASE to read forces & stress correctly
    warnings.filterwarnings("ignore")
    ase_atoms = ase.io.read(fd.name, format="castep-castep")
    forces = [ase_atoms.get_forces()]
    stress = [ase_atoms.get_stress(voigt=False)]

    if clc_type == "molecular dynamics":
        # in case of MD the last forces are repeated -> scratch those
        forces = forces[:-1]
        # also if only one stress tensor appears for MD then that's also only for the last configuration...
        if len(stress) == 1:
            stress = []
    # in case of NVT MD castep prints only one simulation box, hence need to multiply it...
    const_box_cases = set(["nvt"])
    if clc_type == "molecular dynamics" and md_env in const_box_cases:
        for i in range(len(atoms) - 1):
            supercells.append(supercells[0])

    if check_completeness:
        castep_completemeness_checks(
            fd,
            sedc_energy_total,
            sedc_free_energy_total,
            sedc_energy_0K,
            free_energy_total,
            energy_total,
            energy_0K,
            supercells,
            atoms,
            species,
            forces,
            clc_type,
        )

    # assign all attributes to in-house data structure
    structures = []

    if clc_type == "single point energy":
        tmp = supercell()
        tmp["cell"] = supercells[-1]
        tmp["positions"] = atoms[-1]
        tmp["species"] = species[-1]

        if len(forces) > 0:
            tmp["forces"] = forces[-1]
        if len(stress) > 0:
            tmp["stress"] = stress[-1]
        if len(sedc_energy_0K) > 0:
            tmp["energy"] = sedc_energy_0K[-1]
        else:
            tmp["energy"] = energy_0K[-1]
        if spacegroup is not None:
            tmp["spacegroup"] = spacegroup
        structname = fd.name.split("/")[-1]
        tmp["name"] = structname
        if len(enthalpy) > 0:
            tmp["enthalpy"] = enthalpy[-1]
        # add contributing file name
        tmp["files"] = [fd.name.split("/")[-1]]
        structures += [copy.deepcopy(tmp)]
    else:
        for ia in range(len(supercells)):
            tmp = supercell()
            tmp["cell"] = supercells[ia]

            # fractional atomic positions
            if clc_type != "geometry optimization":
                # final positions are not given for geom. opt.
                tmp["positions"] = atoms[ia]
                tmp["species"] = species[ia]

            if len(forces) > 0:
                tmp["forces"] = forces[ia]

            # TO CHECK! is the handling of the stress really ok this way?!
            if len(stress) == 1 and ia == len(energy_total) - 1:
                tmp["stress"] = stress[0]

            elif len(stress) > 1:
                tmp["stress"] = stress[ia]

            if len(sedc_energy_0K) > 0:
                tmp["energy"] = sedc_energy_0K[ia]
            else:
                tmp["energy"] = energy_0K[ia]

            if spacegroup is not None:
                tmp["spacegroup"] = spacegroup

            if len(energy_total) > 1:
                fname = fd.name.split("/")[-1].split(".")
                fname = ".".join(fname[:-1]) + "-{}".format(ia) + "." + fname[-1]
                # print("energy {}: {}".format(fname,energy_total))
                structname = fname
            else:
                structname = fd.name.split("/")[-1]

            tmp["name"] = structname

            if charge is not None:
                tmp["charge"] = charge

            if len(enthalpy) > 0:
                tmp["enthalpy"] = enthalpy[ia]

            # add contributing file name
            tmp["files"] = [fd.name.split("/")[-1]]
            structures += [copy.deepcopy(tmp)]

    return structures


class parse:
    """Class to interface castep parsing with general parsing (end user) class

    Supported file types are self.implemented_file_types

    Example
    -------
    #initialise instance
    castep_parser = parse(/full/dir/path/<file_name>.<file_type> ,file_type=<file_type>)

    #parse
    "/full/dir/path/<file_name>.<file_type>" : castep_parser.run()

    #extract supercell
    supercell = castep_parser.get_supercells()
    """

    def __init__(self, path, file_type):
        assert file_type == "castep"

        self.file_type = file_type  # file type
        self.path = path  # full path to file
        self.supercells = None  # supercell

    def run(self):
        """
        parse self.path
        """
        with open(self.path, "r") as f:
            self.supercells = read_castep_castep(f)
        # assert isinstance(self.supercells,list),'implementation error with parser'

    def get_supercells(self):
        """
        Output:
            - a list of supercell structures
        """
        return self.supercells
