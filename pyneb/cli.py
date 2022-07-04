"""Alternative CLI, using click & multiple endpoints"""
import re
import warnings
from pathlib import Path
from typing import List, Tuple, Union

import ase.io
import click
from ase.calculators.castep import Castep, CastepParam
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io.castep import read_param, write_param
from ase.neb import NEB
from ase.optimize import BFGS

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

    pattern = re.compile(r"finite basis dEtot/dlog\(Ecut\) =\s+(-\d+\.\d+)\s?eV")
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


@main.command("step")
@click.argument("last_step", type=click.INT)
@click.argument("seed", type=click.STRING)
@click.option("--reuse", "-r", is_flag=True)
@click.option("--finite-basis", "-fb", is_flag=True)
def step(
    last_step: int,
    seed: str = "neb-calc",
    reuse: bool = False,
    finite_basis: bool = False,
):
    """NEB step: interpret current results & write new band

    steps:
    1. need to read the saved settings (neb_dir/...)

    """

    # read
    images, castep_calculators = read_images(seed, last_step)

    # actual NEB
    neb = NEB(
        images,
        method="string",
    )
    neb_opt = BFGS(neb)

    # perform 1 step, generating next iteration
    for im in neb.images[1:-1]:
        im.calc.ignored_changes = {"positions"}
    neb_opt.run(fmax=0.05, steps=1)

    # write the NEW interim images
    num_images = len(images) - 2

    current_step = last_step + 1
    ase.io.write(f"{seed}_band{current_step}.xyz", images)
    for i in range(1, num_images + 1):
        # set the calculator back -> keep the .cell keys
        calc = castep_calculators[i]
        atoms = images[i]
        atoms.calc = calc

        ase.io.write(
            f"{seed}_{current_step}-{i:0>2}.cell",
            atoms,
            magnetic_moments="initial",
        )

        if reuse:
            set_reuse(calc.param, f"{seed}_{last_step}-{i:0>2}.check")
        if finite_basis:
            set_finite_basis(calc.param, f"{seed}_0-{i:0>2}.castep")

        write_param(
            f"{seed}_{current_step}-{i:0>2}.param",
            calc.param,
            force_write=True,
        )


@main.command("refine")
@click.argument("last_step", type=click.INT)
@click.argument("seed", type=click.STRING)
@click.option("num_images", type=click.INT)
def step(
    last_step: int,
    num_images: int,
    seed: str = "neb-calc",
):
    """Refine around the TS guess

    Having bracketed the TS, we can interpolate around it

    """

    # read
    images, castep_calculators = read_images(seed, last_step)

    # find the TS guess
    ts_index = images.index(min(images, key=lambda x: x.get_potential_energy()))

    print(ts_index)

    # actual NEB
    neb = NEB(
        images,
        method="string",
    )


if __name__ == "__main__":
    main()
