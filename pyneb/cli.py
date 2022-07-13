"""Alternative CLI, using click & multiple endpoints"""
import json
import re
import warnings
from pathlib import Path
from typing import List, Tuple, Union

import ase.io
import click
import numpy as np
from ase.calculators.castep import Castep, CastepParam
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io.castep import read_param, write_param
from ase.neb import BaseSplineMethod, NEB
from ase.optimize import BFGS
from ase.optimize.precon import SplineFit

SUPP_OPT_METHODS = ("bfgs", "lbfgs", "mdmin", "fire")

# supported initial math estimation methods
SUPP_INI_METHODS = ("linear", "idpp")

# file for NEB metadata
optfile = "neb_metadata"

# dir name for NEB files
nebdir = "pyneb-internal-files/"


@click.group("pyneb")
def main():
    pass


@main.command("init")
@click.argument("num_images", type=click.INT)
@click.argument("seed", type=click.STRING)
# @click.argument("interpolation_method", type=click.types.Choice(SUPP_INI_METHODS))
# @click.argument("opt_method", type=click.types.Choice(SUPP_OPT_METHODS))
def initial(
    num_images: int,
    seed: str = "neb-calc",
    interpolation_method: str = "idpp",
    opt_method: str = "BFGS",
):
    """Initialise the NEB calculation


    files needed:
        SEED_start.cell
        SEED_start.param (-> copied to SEED_end.param)
        SEED_end.cell

    notes
    - we don't need to copy the start & end files too many times,
    those should be done once and read back at every iteration


    steps:
    1. parameter reading & assertions
    2. read the inputs, start & end
        checks:
        - same cell
        - exactly 2 structures
        - same number of atoms & species
        - endpoints are minima (need forces...)
    """

    if num_images < 3:
        raise ValueError("Need at least 3 images for a string/band")

    # filenames
    cell_start = Path(f"{seed}_start.cell")
    param_file = Path(f"{seed}_start.param")
    cell_end = Path(f"{seed}_end.cell")
    for fn in [cell_start, param_file, cell_end]:
        if not fn.exists():
            raise FileNotFoundError(fn.name)

    # read endpoints
    at_start = ase.io.read(cell_start)
    at_end = ase.io.read(cell_end)

    # add interim images
    images = [at_start]
    for _ in range(num_images):
        # copy & keep the .cell keys
        new_image = at_start.copy()
        new_image.calc = at_start.calc
        images.append(new_image)
    images.append(at_end)

    # construct NEB
    neb = NEB(images)
    neb.interpolate(method=interpolation_method, mic=True)

    # read params
    param = read_param(param_file).param

    # end point needs param file
    write_param(f"{seed}_end.param", param, force_write=True)

    # write the interim images (endpoints are the input)
    ase.io.write(f"{seed}_band0.xyz", images)
    for i in range(1, num_images + 1):
        ase.io.write(f"{seed}_0-{i:0>2}.cell", images[i], magnetic_moments="initial")
        write_param(f"{seed}_0-{i:0>2}.param", param, force_write=True)


def read_de_dlog(fn: Union[Path, str]):
    with Path(fn).open("r") as file:
        lines = file.readlines()

    pattern = re.compile(
        r"(?:finite basis dEtot/dlog\(Ecut\)|user-supplied dEtot/dlog\(Ecut\) ) =\s+(-\d+\.\d+)\s?eV"
    )
    for line in lines:
        m = pattern.search(line)
        if m:
            return float(m.group(1))
    return None


def read_castep_outputs(fn: Union[Path, str]) -> Tuple[ase.Atoms, Castep]:
    """Reads CASTEP output and wraps the results into SPC

    we need to keep the calculator though, because that supplies
    the castep cell/param keys
    """
    # read
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        atoms: ase.Atoms = ase.io.read(fn)

    # Castep calculator: disallow calculation, it should have been done already
    def disallow_calculation(*args, **kwargs):
        raise NotImplementedError

    castep_calc: Castep = atoms.calc
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        castep_calc.update = disallow_calculation

    # change calculator
    spc = SinglePointCalculator(
        atoms,
        energy=atoms.get_potential_energy(),
        forces=atoms.get_forces(),
    )
    atoms.calc = spc

    # preserve cell & param keys
    atoms_cell = ase.io.read(str(fn).replace(".castep", ".cell"))
    param = read_param(str(fn).replace(".castep", ".param"))
    castep_calc.cell = atoms_cell.calc.cell
    castep_calc.param = param.param

    return atoms, castep_calc


def set_reuse(params: CastepParam, checkpoint: str):
    """Set reuse of checkpoint file in parameters"""
    params.continuation = None
    params.reuse = checkpoint


def set_finite_basis(params: CastepParam, castep_fn: Union[Path, str]):
    if castep_fn == "":
        # we are starting from scratch
        params.finite_basis_corr = 2
        params.basis_de_dloge = None
        return

    de_dlog = read_de_dlog(castep_fn)

    if de_dlog is None:
        return

    params.finite_basis_corr = 1
    params.basis_de_dloge = de_dlog


def read_images(seed: str, last_step: int) -> Tuple[List[ase.Atoms], List[Castep]]:
    # ends
    cell_start = Path(f"{seed}_start.castep")
    cell_end = Path(f"{seed}_end.castep")

    # interim ones
    interim = Path(".").glob(f"{seed}_{last_step}-*.castep")

    # read the whole band
    at0, calc0 = read_castep_outputs(cell_start)
    images = [at0]
    castep_calculators = [calc0]
    for pth in sorted(interim):
        print(pth)
        at, calc = read_castep_outputs(pth)
        images.append(at)
        castep_calculators.append(calc)
    at1, calc1 = read_castep_outputs(cell_end)
    images.append(at1)
    castep_calculators.append(calc1)

    return images, castep_calculators


class AdjustableImagePositionStringMethod(BaseSplineMethod):
    """String method, with adjustable position of images on the Spline path interpolation"""

    def __init__(self, neb: NEB, lambdas: Union[List[float], np.array]):
        super().__init__(neb)
        assert (
            len(lambdas) == neb.nimages
        ), f"{len(lambdas)}, {neb.nimages}, {len(neb.images)}, {lambdas}"
        self.lambdas = np.array(lambdas)

    def adjust_positions(self, positions):
        # fit cubic spline to positions, reinterpolate to equispace images
        # note this uses the preconditioned distance metric.
        fit = self.neb.spline_fit(positions)
        new_positions = fit.x(self.lambdas[1:-1]).reshape(-1, 3)
        return new_positions


@main.command("step")
@click.argument("last_step", type=click.INT)
@click.argument("seed", type=click.STRING)
@click.option("--add-images", is_flag=True)
@click.option("--reuse", "-r", is_flag=True)
@click.option("--finite-basis", "-fb", is_flag=True)
def step(
    last_step: int,
    seed: str = "neb-calc",
    add_images: bool = False,
    reuse: bool = False,
    finite_basis: bool = False,
):
    """NEB step: interpret current results & write new band

    steps:
    1. need to read the saved settings (neb_dir/...)

    """

    # read
    images, castep_calculators = read_images(seed, last_step)

    # read reaction coordinate if present
    lambdas_on_string = list(np.linspace(0.0, 1.0, len(images)))
    last_info_file = Path(f"{seed}_band{last_step}.info.json")
    if last_info_file.is_file():
        with last_info_file.open(mode="r") as file:
            data = json.load(file)
            lambdas_on_string = data["reaction_coordinate"]

    # save the energies
    energies = [at.get_potential_energy() for at in images]

    # actual NEB
    neb = NEB(
        images,
        method="string",
    )
    neb.method = AdjustableImagePositionStringMethod(neb, lambdas_on_string)
    neb_opt = BFGS(neb)

    # perform 1 step, generating next iteration
    for im in neb.images[1:-1]:
        im.calc.ignored_changes = {"positions"}
    neb_opt.run(fmax=0.05, steps=1)

    # checkpoint & carry of finite basis params
    # checkpoints are the same as before
    checkpoint_names = [f"{seed}_{last_step}-{i:0>2}.check" for i in range(len(images))]
    checkpoint_names[0] = ""
    checkpoint_names[-1] = ""

    # finite basis filenames unchanged
    finite_basis_filenames = [
        f"{seed}_{last_step}-{i:0>2}.castep" for i in range(len(images))
    ]
    finite_basis_filenames[0] = ""
    finite_basis_filenames[-1] = ""

    # adding new images
    if add_images:
        full_string_positions = neb.get_positions()
        spline_fit: SplineFit = neb.spline_fit(full_string_positions)

        # find index of the supposed TS
        ts_index = energies.index(max(energies))
        if ts_index in {0, len(images) - 1}:
            raise ValueError(
                "No transition state found, the maximum of "
                "the band is the start or end structure"
            )

        # lambdas for new images -> positions
        bracket_lambdas = np.linspace(
            lambdas_on_string[ts_index - 1],
            lambdas_on_string[ts_index + 1],
            5,
        )
        positions = spline_fit.x(bracket_lambdas[[1, 3]]).reshape(-1, 3)

        # new image before & after TS
        image_before = images[ts_index].copy()
        image_before.set_positions(positions[: len(image_before)])
        image_after = images[ts_index].copy()
        image_after.set_positions(positions[len(image_after) :])

        # insert them on the right places
        images.insert(ts_index, image_before)
        checkpoint_names.insert(ts_index, checkpoint_names[ts_index])
        castep_calculators.insert(ts_index, castep_calculators[ts_index])
        finite_basis_filenames.insert(ts_index, "")
        lambdas_on_string.insert(ts_index, bracket_lambdas[1])

        images.insert(ts_index + 2, image_after)
        checkpoint_names.insert(ts_index + 2, checkpoint_names[ts_index + 1])
        castep_calculators.insert(ts_index + 2, castep_calculators[ts_index + 1])
        finite_basis_filenames.insert(ts_index + 2, "")
        lambdas_on_string.insert(ts_index + 2, bracket_lambdas[3])

    # write the NEW interim images
    current_step = last_step + 1
    ase.io.write(f"{seed}_band{current_step}.xyz", images)
    for i in range(1, len(images) - 1):
        # set the calculator back -> keep the .cell keys
        atoms = images[i]
        calc = castep_calculators[i]
        atoms.calc = calc

        ase.io.write(
            f"{seed}_{current_step}-{i:0>2}.cell",
            atoms,
            magnetic_moments="initial",
        )

        if reuse:
            set_reuse(calc.param, checkpoint_names[i])
        if finite_basis:
            set_finite_basis(calc.param, finite_basis_filenames[i])

        write_param(
            f"{seed}_{current_step}-{i:0>2}.param",
            calc.param,
            force_write=True,
        )

    # save the current lambdas
    current_info_file = Path(f"{seed}_band{current_step}.info.json")
    with current_info_file.open(mode="w") as file:
        data = {"reaction_coordinate": lambdas_on_string}
        json.dump(data, file, indent=4, sort_keys=True)


@main.command("refine")
@click.argument("last_step", type=click.INT)
@click.argument("seed", type=click.STRING)
@click.argument("num_images", type=click.INT)
@click.option("--reuse", "-r", is_flag=True)
@click.option("--finite-basis", "-fb", is_flag=True)
def refine(
    last_step: int,
    num_images: int,
    seed: str = "neb-calc",
    reuse: bool = True,
    finite_basis: bool = True,
):
    """Refine around the TS guess

    Having bracketed the TS, we can interpolate around it

    """

    # read
    previous_images, castep_calculators = read_images(seed, last_step)

    # find the TS guess
    energies = [at.get_potential_energy() for at in previous_images]
    ts_index = energies.index(max(energies))

    if ts_index in {0, len(previous_images) - 1}:
        raise ValueError(
            "No transition state found, the maximum of "
            "the band is the start or end structure"
        )

    # NEB object
    neb = NEB(
        previous_images,
        method="string",
    )

    # bracket the TS
    # use the original spline, but choose only the part between
    # TS +- 1 images, then parametrise N images with that
    full_string_positions = neb.get_positions()
    spline_fit: SplineFit = neb.spline_fit(full_string_positions)
    original_lambdas = np.linspace(0.0, 1.0, neb.nimages)
    bracket_lambdas = np.linspace(
        original_lambdas[ts_index - 1],
        original_lambdas[ts_index + 1],
        num_images + 2,  # we are dropping the ends
    )
    positions = spline_fit.x(bracket_lambdas[1:-1]).reshape(-1, 3)

    # set positions on new images
    images = [previous_images[ts_index].copy() for _ in range(num_images)]
    n_atoms = len(images[0])
    n1 = 0
    for image in images[1:-1]:
        n2 = n1 + n_atoms
        image.set_positions(positions[n1:n2])
        n1 = n2

    # # write
    ase.io.write(f"{seed}_refined{last_step}.xyz", images)
    calc = castep_calculators[ts_index]
    for i, atoms in enumerate(images):
        # set the calculator back -> keep the .cell keys
        atoms.calc = calc

        ase.io.write(
            f"{seed}_refine_{last_step}-{i:0>2}.cell",
            atoms,
            magnetic_moments="initial",
        )

        if reuse:
            set_reuse(calc.param, f"{seed}_{last_step}-{i:0>2}.check")
        if finite_basis:
            set_finite_basis(calc.param, f"{seed}_0-{i:0>2}.castep")

        write_param(
            f"{seed}_refine_{last_step}-{i:0>2}.param",
            calc.param,
            force_write=True,
        )


if __name__ == "__main__":
    main()
