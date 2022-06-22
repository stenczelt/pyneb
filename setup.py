import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name="pyneb",
    version="0.1.0",
    packages=setuptools.find_packages(),
    install_requires=["numpy", "ase", "docopt"],
    entry_points="""
    [console_scripts]
    pyneb=pyneb.cli:main
    """,
    description="Hybrid Molecular Dynamics for Quantum Mechanics codes with Force Fields",
    author="Tamas K. Stenczel, Andrew (GitHub: @andrew31416)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
    ],
    license_files=("LICENSE",),
    zip_safe=True,
)
