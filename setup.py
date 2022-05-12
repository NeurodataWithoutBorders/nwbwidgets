from setuptools import setup

d = {}
exec(open("nwbwidgets/version.py").read(), None, d)
version = d["version"]

with open("README.md") as f:
    long_description = f.read()

setup(
    author="Ben Dichter",
    author_email="ben.dichter@catalystneuro.com",
    version=version,
    classifiers=[
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Framework :: Jupyter",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    description="This is nwbwidgets, widgets for viewing the contents of a "
    "NWB-file in Jupyter Notebooks using ipywidgets.",
    install_requires=[
        "pynwb",
        "ipympl",
        "matplotlib",
        "numpy",
        "ipyvolume",
        "ndx_grayscalevolume",
        "plotly",
        "scikit-image",
        "tqdm>=4.36.0",
        "ndx-icephys-meta",
        "ipysheet",
        "zarr",
        "ccfwidget",
        "tifffile",
        "ndx-spectrum",
        "trimesh",
        "dandi"
    ],
    license="MIT",
    keywords=["jupyter", "hdf5", "notebook", "nwb"],
    long_description=long_description,
    long_description_content_type="text/markdown",
    name="nwbwidgets",
    packages=[
        "nwbwidgets",
        "nwbwidgets/utils",
        "nwbwidgets/analysis",
        "nwbwidgets/controllers",
    ],
    python_requires=">=2.7",
    setup_requires=["setuptools>=38.6.0", "setuptools_scm"],
    url="https://github.com/NeurodataWithoutBorders/nwb-jupyter-widgets",
)
