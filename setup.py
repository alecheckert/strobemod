#!/usr/bin/env python
"""
setup.py -- install the strobemodels package in the current
environment and also unzip some of the larger data files

"""
import os
import setuptools
from zipfile import ZipFile

# Unzip large data files
REPO_DIR = os.path.split(os.path.abspath(__file__))[0]
DATA_DIR = os.path.join(REPO_DIR, "strobemodels", "data")
targets = [
    os.path.join(DATA_DIR, "fbm_defoc_splines.zip"),
    os.path.join(DATA_DIR, "free_abel_transform.zip")
]
for t in targets:
  print("unzipping {}...".format(t))
  out_path = os.path.split(t)[0]
  with ZipFile(t) as f:
    f.extractall(out_path)

# Install
setuptools.setup(
    name="strobemodels",
    version="1.0",
    author="Alec Heckert",
    author_email="aheckert@berkeley.edu",
    description="Diffusion modeling for stroboscopic particle tracking",
    packages=setuptools.find_packages(),
)
