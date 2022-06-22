"""
routines for dealing with parsing an ase Atoms object to a .cell file
"""

ALLOWED_CELL_KEYWORDS = [
    "lattice_cart",
    "lattice_abc",
    "positions_frac",
    "positions_abs",
    "symmetry_generate",
    "symmetry_ops",
    "symmetry_tol",
    "ionic_constraints",
    "fix_com",
    "cell_constraints",
    "external_pressure",
    "fix_all_ions",
    "fix_all_cell",
    "species_mass",
    "species_pot",
    "ionic_velocities",
    "species_lcao_states",
    "kpoints_list",
    "kpoints_mp_grid",
    "kpoints_mp_spacing",
    "kpoints_mp_offset",
    "kpoint_list",
    "kpoint_mp_grid",
    "kpoint_mp_spacing",
    "kpoint_mp_offset",
    "bs_kpoint_path",
    "bs_kpoint_path_spacing",
    "bs_kpoint_list",
    "bs_kpoint_mp_grid",
    "bs_kpoint_mp_spacing",
    "bs_kpoint_mp_offset",
    "bs_kpoints_path",
    "bs_kpoints_path_spacing",
    "bs_kpoints_list",
    "bs_kpoints_mp_grid",
    "bs_kpoints_mp_spacing",
    "bs_kpoints_mp_offset",
    "phonon_supercell_matrix",
    "phonon_kpoint_path",
    "phonon_kpoint_path_spacing",
    "phonon_kpoint_list",
    "phonon_kpoint_mp_grid",
    "phonon_kpoint_mp_offset",
    "phonon_kpoint_mp_spacing",
    "phonon_gamma_directions",
    "phonon_kpoints_path",
    "phonon_kpoints_path_spacing",
    "phonon_kpoints_list",
    "phonon_fine_kpoint_list",
    "phonon_fine_kpoint_path",
    "phonon_fine_kpoint_path_spacing",
    "phonon_fine_kpoint_mp_grid",
    "phonon_fine_kpoint_mp_spacing",
    "phonon_fine_kpoint_mp_offset",
    "optics_kpoints_list",
    "optics_kpoints_mp_grid",
    "optics_kpoints_mp_spacing",
    "optics_kpoints_mp_offset",
    "optics_kpoint_list",
    "optics_kpoint_mp_grid",
    "optics_kpoint_mp_spacing",
    "optics_kpoint_mp_offset",
    "magres_kpoint_list",
    "magres_kpoint_path",
    "magres_kpoint_path_spacing",
    "magres_kpoint_mp_grid",
    "magres_kpoint_mp_spacing",
    "magres_kpoint_mp_offset",
    "positions_frac_product",
    "positions_abs_product",
    "positions_frac_intermediate",
    "positions_abs_intermediate",
    "fix_vol",
    "species_gamma",
    "species_q",
    "supercell_kpoints_list",
    "supercell_kpoints_mp_grid",
    "supercell_kpoints_mp_spacing",
    "supercell_kpoints_mp_offset",
    "supercell_kpoint_list",
    "supercell_kpoint_mp_grid",
    "supercell_kpoint_mp_spacing",
    "supercell_kpoint_mp_offset",
    "supercell_matrix",
    "nonlinear_constraints",
    "external_efield",
    "positions_noise",
    "cell_noise",
    "hubbard_u",
    "hubbard_alpha",
    "atomic_init",
    "quantisation_axis",
    "quantization_axis",
    "jcoupling_site",
    "chemical_potential",
    "elnes_kpoint_list",
    "elnes_kpoint_mp_grid",
    "elnes_kpoint_mp_spacing",
    "elnes_kpoint_mp_offset",
    "snap_to_symmetry",
    "spectral_kpoint_path",
    "spectral_kpoint_path_spacing",
    "spectral_kpoint_list",
    "spectral_kpoint_mp_grid",
    "spectral_kpoint_mp_spacing",
    "spectral_kpoint_mp_offset",
    "spectral_kpoints_path",
    "spectral_kpoints_path_spacing",
    "spectral_kpoints_list",
    "spectral_kpoints_mp_grid",
    "spectral_kpoints_mp_spacing",
    "spectral_kpoints_mp_offset",
    "sedc_custom_params",
]

PROTECTED_CELL_KEYWORDS = [
    "lattice_cart",
    "lattice_abc",
    "positions_frac",
    "positions_abs",
]


def fetchkeywords(_cellfile):
    """
    strip lattice vector and atom position coordinates
    """

    # keys not to copy

    with open(_cellfile, "r") as f:
        flines = f.readlines()

    # list of lines of keyword extracts to copy
    keylines = []

    for _key in ALLOWED_CELL_KEYWORDS:
        if _key in PROTECTED_CELL_KEYWORDS:
            continue

        instances = []

        for i, line in enumerate(flines):
            if _key in line.lower():
                instances.append(i)

        if len(instances) == 0:
            continue
        elif len(instances) == 1:
            keylines.append(flines[instances[0]] + "\n")
        elif len(instances) == 2:
            for i in range(instances[1] - instances[0] + 1):
                keylines.append(flines[instances[0] + i])
            keylines.append("\n")
        elif len(instances) > 2:
            raise ValueError(
                "{} keyword found {} many times. Parsing error.".format(
                    _key, len(instances)
                )
            )

    return keylines


def comparekeywords(lines):
    """
    compre 2 lists of lines to be identical when space delimited
    """
    from copy import deepcopy

    # list of dictionary of keywords
    dictlist = []

    for i in range(2):
        # list of lines belonging to file i
        flines = lines[i]

        # list of lines of keyword extracts to copy
        keylines = {}

        for _key in ALLOWED_CELL_KEYWORDS:

            instances = []
            tmplines = []

            for i, line in enumerate(flines):
                if _key in line.lower():
                    instances.append(i)

            if len(instances) == 0:
                continue
            elif len(instances) == 1:
                tmplines.append(flines[instances[0]] + "\n")
            elif len(instances) == 2:
                for i in range(instances[1] - instances[0] + 1):
                    tmplines.append(flines[instances[0] + i])
                tmplines.append("\n")
            elif len(instances) > 2:
                raise ValueError(
                    "{} keyword found {} many times. Parsing error.".format(
                        _key, len(instances)
                    )
                )

            keylines.update({_key: tmplines})

        dictlist.append(deepcopy(keylines))

    # make sets out of present keywords, the compare
    sets = [set([]), set([])]
    for i in range(2):
        for _key in dictlist[i]:
            sets[i].update(set([_key]))

    if sets[0] != sets[1]:
        raise ValueError("initial .cell seedfiles do not have identical keyword values")

    # now have list of dictionaries, compare keys
    for _key in dictlist[0]:
        for i in range(len(dictlist[0][_key])):
            if (
                len(dictlist[0][_key][i].split()) != 0
                and len(dictlist[1][_key][i].split()) != 0
            ):
                if (
                    dictlist[0][_key][i].lower().split()
                    != dictlist[1][_key][i].lower().split()
                ):
                    raise ValueError(
                        "initial .cell seedfiles do not have identical keyword values: \n {}\n{}".format(
                            dictlist[0][_key][i], dictlist[1][_key][i]
                        )
                    )


def checkparamfiles(param1, param2):
    """
    compare 2 .param files allowing for deviations in space delimiting

    check for singlepoint and atomic force calculations
    """
    with open(param1, "r") as f:
        flines1 = f.readlines()
    with open(param2, "r") as f:
        flines2 = f.readlines()

    flines = [flines1, flines2]

    # list for dictionaries of .param keyword:value pairs
    dicts = [{}, {}]

    for i in range(2):
        for _l in flines[i]:
            if len(_l) != 0:
                # create dictionary of .param {keywords:value} for both files, store these dicts in a list
                if _l.lstrip().startswith("#"):
                    # this is a comment, should also check ! and
                    continue
                dicts[i].update(
                    {
                        _l.split(":")[0]
                        .split()[0]
                        .lower(): _l.split(":")[1]
                        .split()[0]
                        .lower()
                    }
                )

    # create sets of keywords
    setlist = [set([]), set([])]

    for i in range(2):
        for _key in dicts[i]:
            setlist[i].update(set([_key]))

    # check keyword:values pairs are the same
    assert setlist[0] == setlist[1], ".param files {}, {} are not identical".format(
        param1, param2
    )
    assert all(
        [dicts[1][_key] == dicts[0][_key] for _key in dicts[0]]
    ), ".param files {}, {} are not identical".format(param1, param2)

    # check that calculation type is singlepoint
    assert "task" in dicts[0], "calculation type not specified in {},{}".format(
        param1, param2
    )
    assert (
        dicts[0]["task"] == "singlepoint"
    ), "calculation task specified must be a singlepoint"

    # check that forces are being computed
    assert (
        "calculate_stress" in dicts[0]
    ), '"calculate_stress : true" must be included in {}, {} to output forces'.format(
        param1, param2
    )
    assert (
        dicts[0]["calculate_stress"] == "true"
    ), '"calculate_stress : true" must be included in {}, {} to output forces'.format(
        param1, param2
    )


def checkimages(atoms1, atoms2):
    """
    check atoms objects to see that they are not identical!
    """
    import numpy as np

    different = False

    for i, _atom1 in enumerate(atoms1):
        if np.array_equal(_atom1.position, atoms2[i].position) is not True:
            different = True

    assert different, "initial configurations appear to be identical!"


def adoptcellorder(castep_atoms, cell_atoms):
    """
    switch the order of atoms in a castep Atoms object to be the same order
    as that in a cell Atoms object
    """
    from numpy import zeros as zeros
    from numpy import isclose as isclose
    from ase.atom import Atom
    from ase.atoms import Atoms

    # tolerence
    tol = 0.0001

    # sanity check
    check = [0] * len(castep_atoms.positions)

    atoms_list = [None for i in range(len(castep_atoms.positions))]

    forces = zeros((len(cell_atoms.positions), 3), dtype=float, order="C")

    for i, _cellpos in enumerate(cell_atoms.get_scaled_positions(wrap=True)):
        for j, _cstppos in enumerate(castep_atoms.get_scaled_positions(wrap=True)):
            # need to account for cases when atom can have fractional coordinate of 0 or 1
            if all(
                [
                    any(
                        [
                            isclose(_cstppos[k] - 1, _cellpos[k], atol=tol),
                            isclose(_cstppos[k] + 0, _cellpos[k], atol=tol),
                            isclose(_cstppos[k] + 1, _cellpos[k], atol=tol),
                        ]
                    )
                    for k in range(3)
                ]
            ):
                # if all([isclose(_cstppos[k],_cellpos[k],atol=0.0001) for k in range(3)]):
                # copy atom forces
                forces[i][:] = castep_atoms.forces[j][:]

                # copy atom object
                atoms_list[i] = cell_atoms[i]

                # sanity check
                check[i] += 1

    assert all([_c == 1 for _c in check]), "error reordering .castep atoms"

    # create new Atoms objects, borrow constraints from .cell file
    return Atoms(
        atoms_list,
        cell=cell_atoms.cell,
        pbc=True,
        castep_neb=True,
        system_energy=castep_atoms.energy,
        system_forces=forces,
        constraint=cell_atoms.constraints,
    )
