import os
from pathlib import Path

from setuptools import setup

root = Path(__file__).parent

d = {}
exec(open("nwbwidgets/version.py").read(), None, d)
version = d["version"]

with open("README.md") as f:
    long_description = f.read()

with open(os.path.join(root, "requirements.txt")) as f:
    requirements = f.readlines()

setup(
    author="Ben Dichter",
    author_email="ben.dichter@catalystneuro.com",
    version=version,
    classifiers=[
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Framework :: Jupyter",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    description="This is nwbwidgets, widgets for viewing the contents of a "
    "NWB-file in Jupyter Notebooks using ipywidgets.",
    install_requires=requirements,
    extras_require={
        "human_electrodes": ["nilearn", "trimesh"],
        "mouse_electrodes": ["ccfwidget", "aiohttp"],
        "full": ["ccfwidget", "aiohttp"],
    },
    license="BSD",
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
    python_requires=">=3.7",
    setup_requires=["setuptools>=38.6.0", "setuptools_scm"],
    url="https://github.com/NeurodataWithoutBorders/nwb-jupyter-widgets",
)
