#!/usr/bin/env python
"""
setup.py -- install the strobemodels package in the current
environment and also unzip some of the larger data files

"""
import os
import setuptools
from zipfile import ZipFile

# Build the gs_dp_diff program for evaluation of 
# a Dirichlet process mixture model
BIN_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "bin")
SRC_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "src")
try:
    if not os.path.isdir(BIN_DIR):
        os.mkdir(BIN_DIR)
    if os.path.isfile("/usr/local/bin/gs_dp_diff"):
        os.remove("/usr/local/bin/gs_dp_diff")
    os.system("make -f {}".format(os.path.join(SRC_DIR, "makefile")))
    os.rename(
        os.path.join(SRC_DIR, "gs_dp_diff"),
        os.path.join(BIN_DIR, "gs_dp_diff")
    )
    os.system("ln -s {} /usr/local/bin/gs_dp_diff".format(os.path.join(BIN_DIR, "gs_dp_diff")))
except:
    print("WARNING: gs_dp_diff not installed")

# Unzip large data files
REPO_DIR = os.path.split(os.path.abspath(__file__))[0]
DATA_DIR = os.path.join(REPO_DIR, "strobemodels", "data")
targets = [
    os.path.join(DATA_DIR, "fbm_defoc_splines.zip"),
    os.path.join(DATA_DIR, "abel_transform_dz-0.5um.zip"),
    os.path.join(DATA_DIR, "abel_transform_dz-0.6um.zip"),
    os.path.join(DATA_DIR, "abel_transform_dz-0.7um.zip"),
    os.path.join(DATA_DIR, "free_abel_transform.zip"),
    os.path.join(DATA_DIR, "abel_transforms_range-20um.zip")
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
