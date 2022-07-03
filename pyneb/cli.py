"""Alternative CLI, using click & multiple endpoints"""
from pathlib import Path

import ase.io
import click
from ase.neb import NEB

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
        images.append(at_start.copy())
    images.append(at_end)

    # construct NEB
    neb = NEB(images)
    neb.interpolate(method=interpolation_method, mic=True)

    # write the interim images (endpoints are the input)
    ase.io.write(f"{seed}_band0.xyz", images)
    for i in range(1, num_images + 1):
        ase.io.write(f"{seed}_0-{i:0>2}.cell", images[i])

    raise NotImplementedError


@main.command("step")
def step():
    """NEB step: interpret current results & write new band

    steps:
    1. need to read the saved settings (neb_dir/...)

    """
    raise NotImplementedError
